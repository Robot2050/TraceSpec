"""DLite evaluator: history-free speculative decoding for TraceSpec.

DLite (SJTU-DAI-Sys) drafts a whole block at once from contextual hidden
states (CHS) taken from selected target layers at the last committed
position, without maintaining any draft-side KV history. This evaluator
keeps the harness's shared speculative-decoding loop
(`agentspec.vendor.deepspec.eval.base_evaluator.generate_decoding_sample`)
so verification semantics and metrics (acceptance_lengths /
proposal_lengths / accepted_draft_lengths / verify_count) are defined
identically for DLite, DSpark, DFlash and EAGLE3.

Algorithm mapping onto the generic loop:

* `init_context` — after prefill, take CHS at the last prompt position
  (`gather_pivot_hidden_states(..., -1, ...)`), exactly like DLite's own
  `spec_generate` prefill stage. Students keep no target-history buffer,
  so `recent_condition_hidden` stays empty (it is only used for shape
  bookkeeping by `build_inference_context`).
* `propose` — build the anchor + MASK query embeddings, run the DLite
  draft backbone (joint attention over CHS context and draft queries,
  `is_causal=False`), then serially sample `block_size - 1` tokens with
  the low-rank RNN sequential head. The proposal is always the full
  block; draft probs are exposed for rejection sampling at T > 0
  (equivalent to DLite's "rejection" mode; at T = 0 the generic loop
  degenerates to greedy matching, DLite's "match" mode).
* `update` — refresh CHS at the verify-window position of the last
  committed token (`accepted_draft_tokens`), which is the pivot the next
  anchor token will condition on (anchor at position P, CHS at P - 1 —
  the same off-by-one DLite itself uses).

Note: unlike DSpark, DLite checkpoints legitimately include the *final*
target decoder layer in `target_layer_ids` (layer 35 for a 36-layer
target). DLite's `gather_pivot_hidden_states` applies the embedding
offset itself and matches the checkpoint's training pipeline, so the
`assert_no_final_target_layer` guard from the DSpark evaluator must
NOT be applied here.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agentspec.vendor.deepspec.eval.base_evaluator import (
    BaseEvaluator,
    DraftProposal,
    VerificationResult,
    generate_decoding_sample,
)
from agentspec.vendor.deepspec.utils.sampling import logits_to_probs
from agentspec.vendor.dlite.modeling import DLiteDraftModel
from agentspec.vendor.dlite.utils import gather_pivot_hidden_states


class Qwen3DLiteEvaluator(BaseEvaluator):
    """Speculative decoding with the vendored DLite draft (Qwen3 targets)."""

    EVAL_ATTN_IMPLEMENTATION = "sdpa"
    draft_model_cls = DLiteDraftModel

    @property
    def max_proposal_tokens(self) -> int:
        # DLite proposes block_size - 1 draft tokens after the anchor token.
        return int(self.draft_model.block_size) - 1

    def build_models(self):
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        # DLite draft checkpoints may include the final target layer in
        # target_layer_ids — see the module docstring for why the DSpark
        # assert is intentionally not applied here.
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, draft_model, tokenizer

    def _init_context(
        self,
        *,
        initial_output,
        **kwargs,
    ) -> SimpleNamespace:
        draft_model = self.draft_model
        # CHS at the last prompt position — the pivot the first sampled
        # token (anchor of the first draft block) conditions on.
        target_hidden = gather_pivot_hidden_states(
            initial_output.hidden_states,
            draft_model.target_layer_ids,
            -1,
            draft_model.config.num_target_layers,
        )
        recent_condition_hidden = draft_model.initialize_inference_condition(
            initial_output.hidden_states,
        )
        return SimpleNamespace(
            target_hidden=target_hidden,
            recent_condition_hidden=recent_condition_hidden,
        )

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids: torch.Tensor,
        position_ids: torch.Tensor,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        draft_model = self.draft_model
        temperature = float(self.args.temperature)

        # Anchor token + MASK slots (the generic loop's output_ids buffer is
        # not pre-filled with mask ids, so build the block explicitly).
        draft_input_ids = torch.full(
            (output_ids.size(0), draft_model.block_size),
            int(draft_model.mask_token_id),
            dtype=torch.long,
            device=output_ids.device,
        )
        draft_input_ids[:, 0] = output_ids[:, start]

        noise_embedding = draft_model.build_inference_query_embeddings(
            self.target_model.get_input_embeddings(),
            draft_input_ids,
        )
        current_target_hidden = draft_model.build_inference_current_chs(
            context.target_hidden,
        )
        context_positions, draft_block_positions = draft_model.build_inference_context(
            context.recent_condition_hidden,
            current_target_hidden,
            start,
        )
        full_rotary = torch.cat([context_positions, draft_block_positions], dim=-1)
        block_hidden = draft_model(
            target_hidden=current_target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_block_positions,
            rotary_position_ids=full_rotary,
            is_causal=False,
        )
        draft_hidden = draft_model._prediction_hidden(block_hidden)
        sampled_tokens, draft_logits = draft_model.sample_draft_tokens(
            draft_hidden=draft_hidden,
            first_prev_token_ids=draft_input_ids[:, 0],
            temperature=temperature,
        )

        verify_input_ids = torch.cat(
            [draft_input_ids[:, :1], sampled_tokens],
            dim=1,
        )
        # Full-vocab probs for the generic loop's rejection-sampling
        # verification (one-hot at T=0 → greedy match, DLite "match" mode).
        draft_probs = logits_to_probs(draft_logits, temperature)
        return DraftProposal(
            draft_token_count=int(draft_model.proposal_length),
            verify_input_ids=verify_input_ids,
            draft_probs=draft_probs,
        )

    def _update(
        self,
        context: SimpleNamespace,
        verification: VerificationResult,
    ) -> None:
        draft_model = self.draft_model
        hidden_states = verification.target_output.hidden_states
        # The verify window covers positions [start, start + K]; the last
        # committed token sits at window index `accepted_draft_tokens`, and
        # the next anchor (the bonus token) conditions on its hidden states.
        pivot_index = min(
            int(verification.accepted_draft_tokens),
            hidden_states[0].shape[1] - 1,
        )
        context.target_hidden = gather_pivot_hidden_states(
            hidden_states,
            draft_model.target_layer_ids,
            pivot_index,
            draft_model.config.num_target_layers,
        )

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        return generate_decoding_sample(
            target_model=self.target_model,
            input_ids=input_ids,
            max_new_tokens=int(self.args.max_new_tokens),
            max_proposal_tokens=self.max_proposal_tokens,
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            init_context=self._init_context,
            propose=self._propose,
            update=self._update,
        )
