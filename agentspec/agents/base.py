"""Loop-agent base class: a Plan → Act → Reflect → Answer state machine.

The episode is NOT a rigid plan → (act)* → answer pipeline but a small state
machine:

```
        ┌────────┐
        │  Plan  │◀────────────────────┐
        └───┬────┘                     │
            ↓                          │ REPLAN (reflect decision
        ┌────────┐                     │  or stall intervention)
   ┌──→ │  Act   │                     │
   │    └───┬────┘                     │
   │        ↓ (observation)            │
   │  ┌───────────┐    CONTINUE   ─────┘
   │  │  Reflect  │ ──────────────→ Act
   │  │ evaluate  │    REPLAN     ──→ Plan
   │  │ + decide  │    FINISH     ──→ Answer
   │  └───────────┘
   │ answer block / submit / step limit / stall ──→ Answer
   └─ (Act re-entry is the common path)
```

Every model call is tagged with a mode (plan / act / reflect / answer) and
recorded with its context length, accepted-length stats and wall time, which
is what the evaluation aggregates. Subclasses only define the prompts, the
action format and the tool execution.

Reflect is defined as *evaluate + decide*, not bare "thinking": each reflect
call asks the model to assess the trajectory and end with a decision line
(`DECISION: CONTINUE | REPLAN | FINISH`, parsed leniently, default CONTINUE).
The decision drives the next state, so re-planning mid-episode is a normal
transition rather than an exception.

Two deterministic stall guards sit underneath the model-driven transitions
(greedy decoding otherwise locks into degenerate loops):

- invalid streak — consecutive unparseable `act` responses get escalating,
  counter-stamped feedback, a format-focused reflect at INVALID_REFLECT_AT,
  and a forced re-plan intervention at INVALID_ABORT_AT;
- redundant streak — consecutive zero-information repeats (duplicate reads,
  exhausted searches; the "cycling over the same few commands" pattern) get a
  loop-focused reflect at REDUNDANT_REFLECT_AT and the same intervention at
  REDUNDANT_REPLAN_AT.

An intervention = one deep-analysis reflect call followed by a mandated
re-plan. Early termination (`stuck_format` / `stuck_loop`) happens only when
the episode stalls again after its MAX_REPLANS interventions are spent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import re
import time

from ..backends.base import BaseBackend
from ..config import (
    AgentConfig,
    MODE_ACT,
    MODE_ANSWER,
    MODE_PLAN,
    MODE_REFLECT,
)
from ..datasets import AgentTask
from ..metrics import EpisodeRecord, EpisodeRecorder

# Prompts appended mid-episode to elicit the non-act modes.
PLAN_PROMPT = (
    "Before taking any action, briefly analyze the task and write a step-by-step "
    "plan. Do NOT call any tool in this response."
)
DECISION_INSTRUCTION = (
    "End your response with exactly one decision line:\n"
    "DECISION: CONTINUE — keep executing the current plan;\n"
    "DECISION: REPLAN — the current approach is exhausted, write a new plan;\n"
    "DECISION: FINISH — the evidence is sufficient, produce the final answer."
)
REFLECT_PROMPT = (
    "Evaluate the trajectory so far: what is established, what worked or "
    "failed, and what is still missing. Do NOT call any tool in this "
    "response.\n\n" + DECISION_INSTRUCTION
)
FINAL_PROMPT = (
    "The investigation ends here. Based on everything gathered so far, "
    "produce your final output in the required format."
)

# Lenient reflect-decision line; the LAST match wins (the model may mention
# the options verbatim while reasoning).
_DECISION_RE = re.compile(
    r"decision\s*[:\-]?\s*\*{0,2}\s*`?\s*(continue|re-?plan|finish|answer)",
    re.IGNORECASE,
)

DECISION_CONTINUE = "continue"
DECISION_REPLAN = "replan"
DECISION_FINISH = "finish"


@dataclass
class Action:
    """Parsed from an `act` response."""

    kind: str          # tool name, or "finish", or "invalid"
    payload: str       # tool argument / final output


_logger = logging.getLogger(__name__)


class LoopAgent(ABC):
    # Escalation for consecutive unparseable `act` responses: a forced
    # reflect call at INVALID_REFLECT_AT, a re-plan intervention at
    # INVALID_ABORT_AT.
    INVALID_REFLECT_AT = 3
    INVALID_ABORT_AT = 5
    # Escalation for consecutive *redundant* steps (actions that repeat
    # earlier ones and therefore return no new information, e.g. a duplicate
    # read or an exhausted search): a loop-focused forced reflect at
    # REDUNDANT_REFLECT_AT, a re-plan intervention at REDUNDANT_REPLAN_AT.
    # Termination (`stuck_loop`) happens only when the cycle resumes after
    # the intervention budget is spent.
    REDUNDANT_REFLECT_AT = 4
    REDUNDANT_REPLAN_AT = 8
    # Forced re-plan interventions per episode, shared by the format-stall
    # and no-progress-cycle paths. Model-decided re-plans (via the reflect
    # decision) are a normal transition and do not consume this budget.
    MAX_REPLANS = 1
    # Fractions of max_steps at which a remaining-budget note is appended.
    BUDGET_NUDGE_FRACTIONS = (0.5, 0.85)

    def __init__(self, backend: BaseBackend, cfg: AgentConfig):
        self.backend = backend
        self.cfg = cfg
        self._invalid_streak = 0
        self._redundant_streak = 0
        self._replan_count = 0


    # ---- subclass interface -------------------------------------------------

    @abstractmethod
    def initial_messages(self, task: AgentTask) -> list[dict]:
        """System + first user message for the episode."""

    @abstractmethod
    def parse_action(self, text: str) -> Action:
        """Parse one `act` response into a tool call / finish / invalid."""

    @abstractmethod
    def execute(self, action: Action) -> str:
        """Run a tool call; returns the observation text."""

    def set_task_context(self, task: AgentTask) -> None:
        """Optional hook: prepare per-episode tool state (repo, corpus...)."""

    def teardown_task_context(self) -> None:
        """Optional hook: release per-episode tool state."""

    def plan_prompt(self) -> str:
        """Prompt used to elicit the opening plan. Subclasses may override."""
        return PLAN_PROMPT

    def loop_start_prompt(self) -> str | None:
        """Optional nudge appended right after the plan, marking the end of
        the analysis phase and the start of execution. Without it, greedy
        models tend to continue "planning" inside the action format (e.g.
        comment-only bash blocks). Subclasses may override."""
        return None

    def format_guidance(self) -> str:
        """Concrete action-format reminder (with an example) appended to
        every invalid-response feedback message. Subclasses should override
        with their exact action syntax."""
        return "Respond with exactly one action block in the required format."

    def invalid_feedback(self, streak: int, detail: str) -> str:
        """Observation returned for an unparseable `act` response. The streak
        counter is embedded so identical mistakes never produce identical
        contexts (which would lock greedy decoding into the same output)."""
        message = f"{detail} {self.format_guidance()}"
        if streak > 1:
            message += f" [consecutive format errors: {streak}]"
        if streak >= self.INVALID_ABORT_AT - 1:
            message += (
                " Last chance: one more unparseable response ends this episode."
            )
        return message

    def invalid_reflect_prompt(self) -> str:
        """Prompt for the forced reflect call fired at INVALID_REFLECT_AT
        consecutive unparseable responses."""
        return (
            f"Your last {self.INVALID_REFLECT_AT} responses could not be parsed. "
            f"{self.format_guidance()} In one sentence, state your next concrete "
            "step and the exact action block you will output for it. "
            "Do NOT call any tool in this response.\n\n" + DECISION_INSTRUCTION
        )

    def reflect_prompt(self) -> str:
        """Prompt used to elicit reflection (evaluate + decide). Subclasses
        may override but should keep the DECISION line protocol."""
        return REFLECT_PROMPT

    def final_prompt(self) -> str:
        """Prompt for the closing output when the episode ended WITHOUT a
        submission (step limit / context limit / stall). Subclasses may
        override."""
        return FINAL_PROMPT

    def finalize_prompt(self) -> str:
        """Closing prompt when the agent already submitted mid-loop: the
        submission sits in context as the last assistant turn, and the
        closing answer-mode call produces the final polished version.
        Subclasses may override."""
        return (
            "You have submitted your answer above. Now write the final "
            "version: state the answer directly, then the key evidence "
            "supporting it, and note any remaining uncertainty."
        )

    def forced_reflect_prompt(self, step: int) -> str | None:
        """Subclass hook evaluated after each step's observation: return a
        prompt to force an out-of-schedule reflect-mode call (e.g. after a
        search streak), or None to leave the schedule untouched."""
        return None

    def step_was_redundant(self, action: "Action", observation: str) -> bool:
        """Subclass hook: True when a valid action repeated an earlier one and
        the observation therefore carries no new information (duplicate read,
        exhausted search, ...). Drives the redundant-cycle escalation."""
        return False

    def loop_reflect_prompt(self) -> str:
        """Prompt for the forced reflect fired at REDUNDANT_REFLECT_AT
        consecutive redundant steps."""
        return (
            f"Your last {self.REDUNDANT_REFLECT_AT} actions repeated earlier "
            "ones and returned no new information — you are going in circles. "
            "Stop and take stock: what do you actually know, what is the one "
            "genuinely different thing left to try, and is it time to finish? "
            "Do NOT call any tool in this response.\n\n" + DECISION_INSTRUCTION
        )

    def replan_reflect_prompt(self) -> str:
        """Deep-analysis prompt for the re-plan intervention (a reflect call
        after which a re-plan is mandated, so no DECISION line is asked)."""
        return (
            "You have been repeating actions that return no new information. "
            "Analyze honestly before continuing: which assumption turned out "
            "wrong, what evidence are you missing, and what have you been "
            "avoiding to look at? Do NOT call any tool in this response."
        )

    def replan_prompt(self) -> str:
        """Fresh-plan prompt used on every PLAN re-entry (model-decided or
        mandated)."""
        return (
            "Based on the analysis above, write a NEW plan for the remaining "
            "steps. It must NOT repeat the actions already exhausted — name "
            "concrete, previously untried steps, and state when you will "
            "finish. Do NOT call any tool in this response."
        )

    def reset_escalation_state(self) -> None:
        """Reset escalation streaks on PLAN re-entry, giving the agent a fresh
        slate under the new plan. Subclasses extend with their own streak
        state (call super())."""
        self._invalid_streak = 0
        self._redundant_streak = 0

    def budget_nudge_prompt(self, remaining_steps: int) -> str:
        """Remaining-budget note appended at BUDGET_NUDGE_FRACTIONS of the
        step limit. Subclasses may override with action-specific wording."""
        return (
            f"[system] Step budget: {remaining_steps} tool call(s) left. "
            "Converge: wrap up the current line of investigation and move "
            "toward your final output."
        )

    # ---- shared state machine ------------------------------------------------

    @staticmethod
    def parse_reflect_decision(text: str) -> str:
        """Extract the reflect decision line; defaults to CONTINUE when
        absent, so weak models simply keep acting."""
        matches = _DECISION_RE.findall(text)
        if not matches:
            return DECISION_CONTINUE
        last = matches[-1].lower().replace("-", "")
        if last.startswith("replan"):
            return DECISION_REPLAN
        if last in (DECISION_FINISH, "answer"):
            return DECISION_FINISH
        return DECISION_CONTINUE

    def run(self, task: AgentTask, episode_id: str) -> EpisodeRecord:
        recorder = EpisodeRecorder(episode_id, task.dataset, task.task_id)
        try:
            # Context setup itself can fail (e.g. the SWE worktree cannot be
            # created at the base commit); keep it inside the try so one broken
            # instance is recorded as an error episode instead of killing the shard.
            _logger.info(f"[episode {episode_id}] setting task context...")
            self.set_task_context(task)
            _logger.info(f"[episode {episode_id}] task context ready; starting loop")
            messages = self.initial_messages(task)
            output = ""
            finish_reason = "max_steps"
            submitted_via_action = False
            self._invalid_streak = 0
            self._redundant_streak = 0
            self._replan_count = 0
            nudge_thresholds = sorted(
                {int(self.cfg.max_steps * f) for f in self.BUDGET_NUDGE_FRACTIONS}
            )

            state = MODE_PLAN
            ever_planned = False      # PLAN re-entry uses replan_prompt()
            mandate_replan = False    # intervention reflect → PLAN regardless of decision
            reflect_override: str | None = None  # prompt for the pending REFLECT
            num_steps = 0

            while state != MODE_ANSWER:

                # ---------------- PLAN ----------------
                if state == MODE_PLAN:
                    replanning = ever_planned
                    prompt = self.replan_prompt() if replanning else self.plan_prompt()
                    _logger.info(
                        f"[episode {episode_id}] {'re-plan' if replanning else 'planning'} call..."
                    )
                    # Re-plans are plan-mode calls; like the opening plan they
                    # are recorded with step=-1 so the by-step tables and the
                    # context×progress matrix stay clean.
                    text = self._generate(
                        recorder, messages + [{"role": "user", "content": prompt}],
                        step=-1, mode=MODE_PLAN,
                    )
                    messages.append({"role": "user", "content": prompt})
                    messages.append({"role": "assistant", "content": text})
                    if replanning:
                        recorder.episode.replans += 1
                        self.reset_escalation_state()
                    else:
                        nudge = self.loop_start_prompt()
                        if nudge is not None:
                            messages.append({"role": "user", "content": nudge})
                    ever_planned = True
                    state = MODE_ACT
                    continue

                # ---------------- REFLECT ----------------
                if state == MODE_REFLECT:
                    prompt = reflect_override or self.reflect_prompt()
                    reflect_override = None
                    _logger.info(f"[episode {episode_id}] reflection call...")
                    text = self._generate(
                        recorder, messages + [{"role": "user", "content": prompt}],
                        step=num_steps - 1, mode=MODE_REFLECT,
                    )
                    messages.append({"role": "user", "content": prompt})
                    messages.append({"role": "assistant", "content": text})
                    if mandate_replan:
                        # Stall intervention: the re-plan is mandated, not
                        # decided by the model.
                        mandate_replan = False
                        state = MODE_PLAN
                        continue
                    decision = self.parse_reflect_decision(text)
                    _logger.info(f"[episode {episode_id}] reflect decision: {decision}")
                    if decision == DECISION_REPLAN:
                        state = MODE_PLAN
                    elif decision == DECISION_FINISH:
                        finish_reason = "submitted"
                        state = MODE_ANSWER
                    else:
                        state = MODE_ACT
                    continue

                # ---------------- ACT ----------------
                if num_steps >= self.cfg.max_steps:
                    state = MODE_ANSWER
                    continue
                step = num_steps          # 0-indexed act step
                num_steps += 1
                _logger.info(f"[episode {episode_id}] step {num_steps}/{self.cfg.max_steps}")
                messages = self._trim_to_context_limit(messages, recorder)
                if self._over_context_limit(messages):
                    _logger.warning(f"[episode {episode_id}] context limit exceeded; stopping")
                    finish_reason = "context_limit"
                    state = MODE_ANSWER
                    continue

                if nudge_thresholds and num_steps == nudge_thresholds[0]:
                    nudge_thresholds.pop(0)
                    remaining = self.cfg.max_steps - num_steps
                    _logger.info(f"[episode {episode_id}] budget nudge: {remaining} steps left")
                    messages.append(
                        {"role": "user", "content": self.budget_nudge_prompt(remaining)}
                    )

                _logger.info(f"[episode {episode_id}] step {num_steps}: generating act")
                response_text = self._generate(recorder, messages, step=step, mode=MODE_ACT)
                _logger.info(f"[episode {episode_id}] step {num_steps}: act response length={len(response_text)}")
                messages.append({"role": "assistant", "content": response_text})
                action = self.parse_action(response_text)
                _logger.info(f"[episode {episode_id}] step {num_steps}: parsed action kind={action.kind}")

                if action.kind == "finish":
                    output = action.payload
                    submitted_via_action = True
                    finish_reason = "submitted"
                    _logger.info(f"[episode {episode_id}] submitted")
                    state = MODE_ANSWER
                    continue

                if action.kind == "invalid":
                    self._invalid_streak += 1
                    _logger.warning(
                        f"[episode {episode_id}] unparseable response "
                        f"(streak={self._invalid_streak})"
                    )
                    observation = self.invalid_feedback(self._invalid_streak, action.payload)
                else:
                    self._invalid_streak = 0
                    _logger.info(f"[episode {episode_id}] step {num_steps}: executing tool")
                    observation = self.execute(action)
                    _logger.info(f"[episode {episode_id}] step {num_steps}: observation length={len(observation)}")
                    if self.step_was_redundant(action, observation):
                        self._redundant_streak += 1
                        _logger.warning(
                            f"[episode {episode_id}] redundant step "
                            f"(streak={self._redundant_streak})"
                        )
                    else:
                        self._redundant_streak = 0
                messages.append({"role": "user", "content": observation})

                # Stall interventions: a forced deep-analysis reflect followed
                # by a mandated re-plan. Terminate only when the episode stalls
                # again after the intervention budget is spent.
                stall_reason = None
                if self._invalid_streak >= self.INVALID_ABORT_AT:
                    stall_reason = "format"
                elif self._redundant_streak >= self.REDUNDANT_REPLAN_AT:
                    stall_reason = "loop"
                if stall_reason is not None:
                    if self._replan_count >= self.MAX_REPLANS:
                        _logger.warning(
                            f"[episode {episode_id}] aborting loop: still stalling "
                            f"({stall_reason}) after {self._replan_count} re-plan(s)"
                        )
                        finish_reason = f"stuck_{stall_reason}"
                        state = MODE_ANSWER
                        continue
                    self._replan_count += 1
                    _logger.warning(
                        f"[episode {episode_id}] {stall_reason} stall: re-plan "
                        f"intervention #{self._replan_count}/{self.MAX_REPLANS}"
                    )
                    reflect_override = self.replan_reflect_prompt()
                    mandate_replan = True
                    state = MODE_REFLECT
                    continue

                # Reflect triggers, in priority order (at most one per step).
                if self._invalid_streak == self.INVALID_REFLECT_AT:
                    _logger.info(f"[episode {episode_id}] forced reflection call (format-triggered)...")
                    reflect_override = self.invalid_reflect_prompt()
                    state = MODE_REFLECT
                    continue
                if self._redundant_streak == self.REDUNDANT_REFLECT_AT:
                    _logger.info(f"[episode {episode_id}] forced reflection call (loop-triggered)...")
                    reflect_override = self.loop_reflect_prompt()
                    state = MODE_REFLECT
                    continue
                # A subclass-requested forced reflection (e.g. after a search
                # streak) takes precedence over the periodic schedule.
                forced_prompt = self.forced_reflect_prompt(step)
                if forced_prompt is not None:
                    _logger.info(f"[episode {episode_id}] forced reflection call (streak-triggered)...")
                    reflect_override = forced_prompt
                    state = MODE_REFLECT
                    continue
                if (
                    self.cfg.reflect_every > 0
                    and num_steps % self.cfg.reflect_every == 0
                    and num_steps < self.cfg.max_steps
                ):
                    state = MODE_REFLECT
                    continue
                # default: keep acting

            # ---------------- ANSWER (closing phase, every episode) ----------------
            recorder.episode.num_steps = num_steps
            # Submitted episodes already carry a draft answer (the act response
            # that contained the answer block) in context; the closing call
            # finalizes it. Non-submitted episodes get the fallback prompt.
            # Keeping the closing call unconditional makes the answer-mode
            # stats cover all episodes instead of only failed ones.
            closing_prompt = (
                self.finalize_prompt() if submitted_via_action else self.final_prompt()
            )
            _logger.info(f"[episode {episode_id}] closing answer call (reason={finish_reason})...")
            final_messages = messages + [{"role": "user", "content": closing_prompt}]
            final_messages = self._trim_to_context_limit(final_messages, recorder)
            closing_text = self._generate(recorder, final_messages, step=num_steps, mode=MODE_ANSWER)
            if finish_reason == "max_steps":
                finish_reason = "forced_answer"
            if not submitted_via_action:
                # For non-submitted episodes the closing call IS the answer;
                # for submitted ones `output` stays the payload the agent
                # submitted (e.g. SWE keeps its captured diff trace).
                output = closing_text
            _logger.info(f"[episode {episode_id}] finishing with reason={finish_reason}")
            return recorder.finish(finish_reason=finish_reason, output=output)
        except Exception as exc:  # keep one broken episode from killing the run
            _logger.exception(f"[episode {episode_id}] exception in agent loop")
            return recorder.finish(finish_reason="error", output=f"{type(exc).__name__}: {exc}")
        finally:
            _logger.info(f"[episode {episode_id}] tearing down task context")
            self.teardown_task_context()

    # ---- helpers --------------------------------------------------------------

    def _generate(
        self,
        recorder: EpisodeRecorder,
        messages: list[dict],
        *,
        step: int,
        mode: str,
    ) -> str:
        _logger.info(f"[generate] mode={mode} step={step} calling backend.generate...")
        t0 = time.perf_counter()
        result = self.backend.generate(
            messages, max_new_tokens=self.cfg.max_new_tokens
        )
        elapsed = time.perf_counter() - t0
        _logger.info(
            f"[generate] mode={mode} step={step} done: "
            f"ctx_tokens={result.num_input_tokens} out_tokens={result.num_output_tokens} "
            f"verify_count={result.verify_count} accepted_sum={result.accepted_sum} "
            f"time={elapsed:.2f}s (backend reported {result.gen_time_s:.2f}s)"
        )
        recorder.add_generation(
            step=step,
            mode=mode,
            num_input_tokens=result.num_input_tokens,
            num_output_tokens=result.num_output_tokens,
            verify_count=result.verify_count,
            accepted_sum=result.accepted_sum,
            proposal_sum=result.proposal_sum,
            gen_time_s=result.gen_time_s,
            backend=self.backend.name,
        )
        return result.text

    def _over_context_limit(self, messages: list[dict]) -> bool:
        ctx = self.backend.count_tokens(messages)
        over = ctx > self.cfg.max_context_tokens
        if over:
            _logger.warning(f"[context] ctx_tokens={ctx} > limit={self.cfg.max_context_tokens}")
        return over

    def _trim_to_context_limit(
        self, messages: list[dict], recorder: EpisodeRecorder
    ) -> list[dict]:
        """Drop the oldest middle turns (keep system + first task message and
        the newest turns) until under the context limit. A marker note is left
        at the cut point so the model can tell the history has a gap."""
        if not self._over_context_limit(messages):
            return messages
        pinned_head = 2 if messages and messages[0]["role"] == "system" else 1
        messages = list(messages)
        dropped = 0
        while len(messages) > pinned_head + 1 and self._over_context_limit(messages):
            del messages[pinned_head]
            dropped += 1
            recorder.episode.truncations += 1
        if dropped:
            messages.insert(
                pinned_head,
                {
                    "role": "user",
                    "content": (
                        f"[system] {dropped} earlier turn(s) of this conversation "
                        f"were omitted to fit the context window."
                    ),
                },
            )
        return messages
