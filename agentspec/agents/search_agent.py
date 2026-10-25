"""BrowseComp-plus agent: deep-search over a local document corpus.

ReAct-style loop with three actions (search / read / answer), backed by a
disk-resident FTS5 index with BM25 ranking.  On top of the generic LoopAgent
this agent maintains search state and escalates against the "search-only"
death spiral that makes every episode run to the step limit:

- after `REPLAN_GUIDANCE_AFTER_SEARCH_STREAK` consecutive searches the base
  loop fires a forced reflect-mode call (via `forced_reflect_prompt`), so the
  model must stop and reconsider what is known / missing;
- after `FORCE_READ_AFTER_SEARCH_STREAK` consecutive searches further search
  calls are blocked and the most relevant unread document is force-read with
  guidance to answer.

Two robustness measures keep greedy decoding from wasting the episode:

- `parse_action` accepts several fallback action shapes (XML-ish tags, bare
  fences, inline `search: ...` lines) instead of only exact tagged fences —
  a single malformed response used to lock the episode into an
  unparseable-repeat loop;
- near-duplicate queries are no longer hard-blocked: the search still runs
  (already-read documents are excluded, so the observation carries fresh
  material) but is prefixed with a no-rephrase warning. A hard block left the
  context unchanged and the model simply re-issued the same query forever,
  which is why earlier runs executed under two real searches per episode.
"""
from __future__ import annotations

import logging
import re

from ..backends.base import BaseBackend
from ..config import AgentConfig
from ..datasets import AgentTask
from ..tools.search_tools import SearchTool
from .base import DECISION_INSTRUCTION, Action, LoopAgent

_logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous deep-research analyst. You answer open-ended questions that require multi-step evidence gathering over a local corpus of ~100k web documents (keyword search plus full-document reads).

Method:
1. Decompose the question into the concrete facts you must establish.
2. Search with focused keyword queries; each query must attack a genuinely different angle or missing fact — near-duplicate rewrites of an earlier query are flagged and return no fresh material once you have read the top hits.
3. Read the most promising documents instead of grinding out more searches; repeated searching without reading is blocked.
4. Cross-check important facts against more than one document when possible, and keep track of what each document established.
5. Answer only when the evidence covers every part of the question.

In every response, first write 1-3 sentences of reasoning: what the evidence shows so far, what is still missing, and why the next action is the best one. Be specific to the question — no generic filler. Then output exactly ONE action as a single code block (the reasoning stays OUTSIDE the block; the block contains only the query, docid, or answer text):

```search
<keyword query>
```
to search the corpus, or

```read
<docid>
```
to read the full text of one document, or

```answer
<the final answer, followed by the key evidence supporting it>
```
when you are confident you have found the answer.

Important: after a few consecutive searches you will be required to read one of the returned documents or give the final answer. Repeated searches without reading are blocked, and rephrasing an earlier query returns a warning instead of new results."""

_BLOCK_RE = re.compile(r"```(search|read|answer)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
# Fallback shapes, tried in order when no tagged fence is present: XML-ish
# tags, a bare fence whose first word is the action, and an inline
# `action: payload` line. They only salvage otherwise-unparseable responses.
_XML_BLOCK_RE = re.compile(r"<(search|read|answer)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_BARE_BLOCK_RE = re.compile(r"```\s*\n?\s*(search|read|answer)\s*[:\n](.*?)```", re.DOTALL | re.IGNORECASE)
_INLINE_RE = re.compile(r"^\s*\*{0,2}(search|read|answer)\*{0,2}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _query_tokens(query: str) -> frozenset:
    """Normalized token set of a search query, for duplicate detection."""
    return frozenset(re.findall(r"[a-z0-9]+", query.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SearchAgent(LoopAgent):
    """Deep-search agent with planning, forced reflection, and forced reads."""

    # Force a reflect-mode call after this many consecutive searches.
    REPLAN_GUIDANCE_AFTER_SEARCH_STREAK = 3
    # Block searching and force-read the top unread document after this streak.
    FORCE_READ_AFTER_SEARCH_STREAK = 5
    # Flag a search whose token-set Jaccard similarity with any earlier query
    # of the same episode reaches this threshold. Flagged queries still run
    # (read documents are excluded); the result is prefixed with a
    # no-rephrase warning, and the streak keeps growing toward the forced
    # reflect / force-read escalation.
    DUP_QUERY_JACCARD = 0.7

    _DOCID_RE = re.compile(r"\(docid=([^)]+)\)")

    def __init__(self, backend: BaseBackend, cfg: AgentConfig, db_path: str):
        super().__init__(backend, cfg)
        self.db_path = db_path
        self.tool: SearchTool | None = None
        self._reset_search_state()

    def _reset_search_state(self) -> None:
        self.search_count = 0
        self.read_count = 0
        self.search_streak = 0
        self.last_search_docids: list[str] = []
        self.seen_docids: set[str] = set()
        self.past_queries: list[frozenset] = []
        self.query_history: list[str] = []
        self._streak_reflect_fired = False
        self._last_step_redundant = False

    def set_task_context(self, task: AgentTask) -> None:
        if self.tool is None:
            self.tool = SearchTool(
                self.db_path,
                top_k=self.cfg.search_top_k,
                max_chars=self.cfg.observation_max_chars,
            )
        self._reset_search_state()

    def teardown_task_context(self) -> None:
        # The connection is shared across episodes in this process; keep it.
        pass

    def initial_messages(self, task: AgentTask) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {task.query}"},
        ]

    def plan_prompt(self) -> str:
        return (
            "Before taking any action, analyze the question and write a concise "
            "investigation plan: decompose it into the facts you must establish, "
            "decide which key terms to search first, which kinds of documents "
            "you expect to read, and what evidence will be sufficient to answer. "
            "Do NOT call any tool in this response."
        )

    def reflect_prompt(self) -> str:
        return (
            f"{self._state_summary()}"
            "Evaluate your progress: what has each document established, what "
            "is still missing, and how close is the evidence to covering every "
            "part of the question? Do NOT call any tool in this response."
            "\n\n" + DECISION_INSTRUCTION
        )

    def _state_summary(self) -> str:
        """Compact investigation ledger prepended to reflect prompts, so the
        model plans against what it has already tried instead of looping."""
        queries = "; ".join(f'"{q}"' for q in self.query_history[-6:]) or "(none yet)"
        docs = ", ".join(sorted(self.seen_docids)) or "(none yet)"
        return (
            f"[investigation so far] queries issued: {queries} | "
            f"documents read: {docs}\n\n"
        )

    def final_prompt(self) -> str:
        return (
            "The investigation ends here. Based on the evidence gathered "
            "so far, give your best final answer in an ```answer ... ``` block: "
            "state the answer directly, then the key evidence supporting it, "
            "and note any remaining uncertainty."
        )

    def finalize_prompt(self) -> str:
        return (
            "You have submitted your answer above. Write the final version in "
            "an ```answer ... ``` block: state the answer directly, then the "
            "key evidence supporting it, and note any remaining uncertainty."
        )

    def forced_reflect_prompt(self, step: int) -> str | None:
        """Fire at most one reflect call per search streak once the streak
        reaches the replan threshold; the flag resets when a read breaks the
        streak (so a renewed spiral triggers it again)."""
        if (
            self.search_streak >= self.REPLAN_GUIDANCE_AFTER_SEARCH_STREAK
            and not self._streak_reflect_fired
        ):
            self._streak_reflect_fired = True
            _logger.info(
                f"[searchagent] streak={self.search_streak}: forcing reflection"
            )
            return self.reflect_prompt()
        return None

    def step_was_redundant(self, action: Action, observation: str) -> bool:
        """Blocked duplicate reads and exhausted searches carry zero
        information; everything else (including near-duplicate searches, which
        still return fresh snippets) is progress."""
        return self._last_step_redundant

    def loop_reflect_prompt(self) -> str:
        return (
            f"{self._state_summary()}"
            f"Your last {self.REDUNDANT_REFLECT_AT} actions repeated earlier "
            "ones and produced no new information — you are going in circles. "
            "Decide now: search a genuinely different angle, read an unread "
            "document, or give the final answer with ```answer ... ```. "
            "Do NOT call any tool in this response."
        )

    def budget_nudge_prompt(self, remaining_steps: int) -> str:
        return (
            f"[system] Step budget: {remaining_steps} action(s) left. "
            "Converge: if the evidence already covers the question, answer now "
            "with ```answer ... ```; otherwise spend the remaining budget only "
            "on the single most decisive missing fact."
        )

    def replan_reflect_prompt(self) -> str:
        return (
            f"{self._state_summary()}"
            "You have been repeating searches/reads without gaining new "
            "information. Analyze honestly before continuing: which "
            "interpretation of the question might be wrong, which entity, "
            "alias, time frame or facet have you not tried, and which "
            "retrieved-but-unread documents remain promising? Do NOT call "
            "any tool in this response."
        )

    def replan_prompt(self) -> str:
        return (
            "Write a NEW investigation plan for the remaining steps: the "
            "specific new query angles you will try (different entities, "
            "aliases or facets — not rephrasals), the unread documents you "
            "will read, or state that the evidence is sufficient and you will "
            "answer next. Do NOT call any tool in this response."
        )

    def reset_escalation_state(self) -> None:
        """Fresh slate after a re-plan: the search streak and its reflect
        cadence restart under the new plan (already-read documents stay
        excluded — re-reading them would still return nothing new)."""
        super().reset_escalation_state()
        self.search_streak = 0
        self._streak_reflect_fired = False

    def loop_start_prompt(self) -> str | None:
        return (
            "Your plan is noted. Now execute the first investigation step: "
            "1-3 sentences of reasoning, then your first ```search ... ``` "
            "block with the single most informative keyword query."
        )

    def format_guidance(self) -> str:
        return (
            "Respond with 1-3 sentences of reasoning, then exactly ONE action "
            "block and nothing else:\n"
            "```search\n<keyword query>\n``` — search the corpus;\n"
            "```read\n<docid>\n``` — read one retrieved document;\n"
            "```answer\n<answer + key evidence>\n``` — finish."
        )

    def parse_action(self, text: str) -> Action:
        """Pure parsing; all streak policy lives in `execute`."""
        for pattern in (_BLOCK_RE, _XML_BLOCK_RE, _BARE_BLOCK_RE):
            match = pattern.search(text)
            if match is not None:
                break
        else:
            match = _INLINE_RE.search(text)
        if match is None:
            return Action("invalid", "[format error] No recognizable action block found.")

        kind, payload = match.group(1).lower(), match.group(2).strip()
        if kind == "answer":
            return Action("finish", payload)
        return Action(kind, payload)

    def execute(self, action: Action) -> str:
        assert self.tool is not None
        self._last_step_redundant = False

        if action.kind == "search":
            # Hard limit: block the search and force-read the top unread doc.
            if self.search_streak >= self.FORCE_READ_AFTER_SEARCH_STREAK:
                docid = self._next_unread_docid()
                if docid is not None:
                    _logger.info(
                        f"[searchagent] streak={self.search_streak}: search blocked, "
                        f"forcing read of {docid}"
                    )
                    return self._read_with_guidance(docid)
                return (
                    "[system] You have searched repeatedly and no unread retrieved "
                    "documents remain. Please provide the final answer with "
                    "```answer ... ```."
                )

            # Near-duplicate of an earlier query: still execute it (already
            # read documents are excluded, so the observation carries fresh
            # material), but prefix a no-rephrase warning. Hard-blocking the
            # query left the context unchanged and locked greedy decoding
            # into re-issuing the identical query forever.
            tokens = _query_tokens(action.payload)
            near_dup = any(
                _jaccard(tokens, past) >= self.DUP_QUERY_JACCARD
                for past in self.past_queries
            )
            result = self.tool.search(action.payload, exclude=self.seen_docids)
            if result.startswith("[no new results"):
                # Every top hit was already read: the observation carries no
                # new material, so the step feeds the cycle guard.
                self._last_step_redundant = True
            self.search_count += 1
            self.search_streak += 1
            self.past_queries.append(tokens)
            self.query_history.append(action.payload)
            self.last_search_docids = self._extract_docids(result)
            _logger.info(
                f"[searchagent] search #{self.search_count}, streak={self.search_streak}, "
                f"near_dup={near_dup}, docids={self.last_search_docids[:5]}"
            )
            if near_dup:
                result = (
                    "[system] This query is very similar to one you already ran; "
                    "the results below exclude documents you have read. Do NOT "
                    "rephrase this query again — ```read <docid>``` one of the "
                    "documents below, search a genuinely different angle, or give "
                    "the final answer with ```answer ... ```.\n\n"
                ) + result
            return result

        if action.kind == "read":
            docid = action.payload
            if docid in self.seen_docids:
                _logger.info(f"[searchagent] duplicate read blocked: {docid}")
                self._last_step_redundant = True
                return (
                    f"[system] You already read document {docid}; its content is "
                    f"in the conversation above. Re-reading adds nothing — search "
                    f"a genuinely new angle, read an unread document, or give the "
                    f"final answer with ```answer ... ```."
                )
            self.read_count += 1
            self.seen_docids.add(docid)
            self._reset_streak()
            return self.tool.read(docid)

        return f"[unknown action: {action.kind}]"

    def _reset_streak(self) -> None:
        self.search_streak = 0
        self._streak_reflect_fired = False

    def _extract_docids(self, text: str) -> list[str]:
        return self._DOCID_RE.findall(text)

    def _next_unread_docid(self) -> str | None:
        for docid in self.last_search_docids:
            if docid not in self.seen_docids:
                return docid
        return None

    def _read_with_guidance(self, docid: str) -> str:
        """Force-read a document and append guidance to answer afterwards."""
        text = self.tool.read(docid)
        self.read_count += 1
        self.seen_docids.add(docid)
        self._reset_streak()
        return (
            f"{text}\n\n"
            f"[system] You have searched repeatedly, so the system has read the "
            f"most relevant document for you (docid={docid}). "
            f"Based on this document and your previous searches, please provide "
            f"the final answer with ```answer ... ```, or search for one specific "
            f"missing piece of information if absolutely necessary."
        )

    def close(self) -> None:
        if self.tool is not None:
            self.tool.close()
            self.tool = None
