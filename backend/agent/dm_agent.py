"""
DM agent — a two-model LangGraph StateGraph: a mechanics node (tool-calling
loop, low temperature) hands off to a narrator node (prose only, no tools,
higher temperature) once its tool calls are resolved for the turn.

Lifecycle
---------
Call `agent_lifespan()` as a FastAPI lifespan context manager. It creates the
Postgres connection pool, sets up the LangGraph checkpoint tables, and stores
the ready checkpointer in module-level state so `get_agent()` can build
per-session agents without touching the DB each time.

Streaming
---------
`stream_response()` is an async generator that yields plain-text tokens as they
arrive from the narrator node only, suitable for piping into an SSE response.

Session management
------------------
`get_thread_messages()` retrieves the full message history for a thread.
`summarize_session()` generates a narrative chronicle, key events list, an
adventure-progress note, and any newly-stated entity relationships (Stage
1.5's incremental relation graph) from that history, suitable for storing as
a session record and indexing in HistoryStore for future RAG retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from backend.config import settings
from backend.data.spells import ALL_SPELLS, SPELL_MENUS
from backend.llm import vllm_chat
from backend.locks import campaign_read_lock, campaign_write_lock
from backend.models import Campaign, CombatantType, InitiativeEntry
from backend.agent.prompts import (
    OOC_MARKER, VERIFIED_CHANGES_MARKER, VERIFIED_ROLLS_MARKER, get_mechanics_system_prompt,
    get_narrator_system_prompt,
)
from backend.agent.session_zero_prompt import get_session_zero_mechanics_prompt, get_session_zero_narrator_prompt
from backend.stores.campaign_store import CampaignStore
from backend.stores.draft_store import DraftStore
from backend.stores.graph_store import RelationGraphStore
from backend.stores.history_store import HistoryStore
from backend.stores.lore_store import LoreStore
from backend.stores.rules_store import RulesStore
from backend.tools import chargen, companion, dice, rules
from backend.tools import campaign as campaign_tools  # aliased — `campaign` is used throughout this file as a Campaign instance
from backend.tools.combat import build_encounter_context
from backend.tools.npc import build_traveling_npcs_context
from backend.tools.registry import get_npc_prep_tools, get_tools, get_world_prep_tools
from backend.tools.resolution import resolve_pending_action_impl
from backend.tools._helpers import find_char

log = logging.getLogger(__name__)

# ── module-level singletons (set by lifespan) ─────────────────────────────────

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


@asynccontextmanager
async def agent_lifespan():
    """Async context manager for use in FastAPI lifespan."""
    global _pool, _checkpointer

    conninfo = settings.database_url.replace("postgresql+psycopg://", "postgresql://")

    _pool = AsyncConnectionPool(conninfo=conninfo, kwargs={"autocommit": True}, open=False)
    await _pool.open()

    _checkpointer = AsyncPostgresSaver(_pool)
    await _checkpointer.setup()

    try:
        yield
    finally:
        await _pool.close()
        _pool = None
        _checkpointer = None


# ── models ─────────────────────────────────────────────────────────────────────

def _get_model() -> ChatOpenAI:
    """Single-model instance used by world-prep, party-fill, and session
    summarization — structured/tool-driven passes with no dedicated
    narration step. Session 0 chargen used to be in this list too, but moved
    to its own mechanics/narrator split (get_session_zero_agent) 2026-07-04
    after a live conversation caught this combined-role approach narrating
    fake tool calls — see get_session_zero_agent's docstring and
    _detect_fake_tool_call.

    Was hardcoded to "qwen2.5:14b" (a separate, smaller model from
    settings.mechanics_model) until 2026-07-03 — switched after a live Session
    0 test surfaced a severe instance of a quirk already documented elsewhere
    in this file (see run_fill_party's comment about qwen2.5:14b appending a
    fenced json block after its real summary, "a second, never-actually-
    executed tool call it's narrating rather than invoking"): across an entire
    multi-turn chargen conversation, the model wrote convincing prose and fake
    ```json``` blocks that looked like tool calls but never made a single real
    one, then confidently declared the character "successfully created" while
    DraftStore stayed completely empty and finalize_character was never truly
    called. settings.mechanics_model has been extensively live-tested
    elsewhere in this app for reliable, genuine tool-calling discipline
    (including self-correcting after guardrail rejections across multi-turn
    combat) — standardizing on one validated model closes this failure
    class rather than working around it per call site. Note this model
    choice didn't fully close the gap for Session 0 either, hence the
    2026-07-04 structural split — see above. (2026-07-13: model/client
    swapped from Ollama's gemma4:26b-mlx to vLLM-metal's Qwen3-30B-A3B-4bit,
    see vllm-migration-plan.md — this function's own role/rationale is
    otherwise unchanged.)"""
    return vllm_chat(temperature=0.7)


def _get_mechanics_model() -> ChatOpenAI:
    """Low-temperature tool-calling model for the in-game agent.
    Cross-cutting client policy is owned and documented by backend/llm.py's
    vllm_chat() (2026-07-13, vllm-migration-plan.md — swapped from Ollama's
    ollama_chat() for real forced tool-calling via tool_choice="required")."""
    return vllm_chat(temperature=0.1)


def _get_narrator_model() -> ChatOpenAI:
    """Higher-temperature prose model for the in-game agent. No tool access.

    Same underlying model as the mechanics node (settings.mechanics_model),
    just a different temperature/prompt — benchmarked against a smaller
    dedicated narrator model (gemma4:12b-mlx, back when the chat backend was
    Ollama) and found to be both faster at raw generation and to incur no
    residency/swap cost either way, so a second model wasn't worth the extra
    resident memory and the added config surface. (2026-07-13: model/client
    swapped to vLLM-metal's Qwen3-30B-A3B-4bit, see vllm-migration-plan.md —
    the narrator has no tool_choice concerns either way, since it never had
    tools bound.)"""
    return vllm_chat(temperature=0.8)


# ── message trimmer ────────────────────────────────────────────────────────────

# Raw (non-system) messages kept in the mechanics model's context. Originally 30,
# carried over unchanged from the old single-model agent — but in the current
# two-node graph, every mechanics tool call burns 2 raw messages (the tool-call
# AIMessage + its ToolMessage result), so a handful of narrative turns with a
# few tool calls each could exhaust a 30-message window fast, well before a
# session felt long. gemma4:26b-mlx has 32k tokens of context to spare, so 30
# was far more conservative than the model actually requires — raised to 100.
_MAX_MESSAGES = 100


def _make_state_modifier(system_prompt: str):
    """Returns a state modifier that prepends the system prompt and trims
    old messages so the context window stays manageable for local models."""
    system_msg = SystemMessage(content=system_prompt)

    def modifier(state: dict) -> list[BaseMessage]:
        msgs = state["messages"]
        if len(msgs) > _MAX_MESSAGES:
            trimmed = msgs[-_MAX_MESSAGES:]
            # Drop any ToolMessages at the start of the window that lost their
            # paired AIMessage(tool_calls) during the trim. Ollama rejects a
            # ToolMessage with no preceding assistant tool-call in scope.
            while trimmed and isinstance(trimmed[0], ToolMessage):
                trimmed = trimmed[1:]
        else:
            trimmed = msgs
        result = [system_msg] + trimmed
        # Injected by mechanics_node's guardrail (see _detect_missing_followup) when
        # a turn stopped short mid-resolution during combat — a one-time corrective
        # nudge for the retry, not a persistent instruction.
        note = state.get("correction_note")
        if note:
            result.append(HumanMessage(content=f"[SYSTEM CHECK — internal, not player dialogue]\n{note}"))
        return result

    return modifier


def _make_mechanics_modifier(system_prompt: str, campaign_id: str, store: CampaignStore):
    """Async-aware sibling of _make_state_modifier, used ONLY for the mechanics
    node. Does everything the plain modifier does, plus live-injects the active
    encounter's state (round, initiative, monster stats, any pending reaction)
    on every single mechanics-node invocation — not once per turn, but on every
    loop iteration within a turn, so it reflects tool calls made earlier in the
    SAME turn too. Replaces get_active_encounter as a callable tool: correctness
    no longer depends on the model remembering to call it every turn.

    Narrator/session-zero/world-prep keep using the plain synchronous
    _make_state_modifier unchanged — none of them need live encounter state, and
    _make_state_modifier is also handed to create_react_agent's `prompt` hook
    elsewhere, which isn't verified to support an async callable, so that path
    is left untouched."""
    system_msg = SystemMessage(content=system_prompt)

    async def modifier(state: dict) -> list[BaseMessage]:
        msgs = state["messages"]
        if len(msgs) > _MAX_MESSAGES:
            trimmed = msgs[-_MAX_MESSAGES:]
            while trimmed and isinstance(trimmed[0], ToolMessage):
                trimmed = trimmed[1:]
        else:
            trimmed = msgs
        result = [system_msg] + trimmed

        campaign = await store.load(campaign_id)
        if campaign and campaign.active_encounter and campaign.active_encounter.is_active:
            ctx = build_encounter_context(campaign)
            if ctx:
                result.append(HumanMessage(content=f"[LIVE ENCOUNTER STATE — internal, not player dialogue]\n{ctx}"))
        if campaign:
            tctx = build_traveling_npcs_context(campaign)
            if tctx:
                result.append(HumanMessage(content=f"[TRAVELING NPCs — internal, not player dialogue]\n{tctx}"))

        note = state.get("correction_note")
        if note:
            result.append(HumanMessage(content=f"[SYSTEM CHECK — internal, not player dialogue]\n{note}"))
        return result

    return modifier


_INTERNAL_DIRECTIVE_MARKER = "[SESSION START"  # see build_session_kickoff_message


def _narrative_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep only human turns and DM narrative — drop tool calls/results and
    tool-call-only AIMessages. Shared by format_transcript, the narrator
    node's context window, and chat hydration on page reload, since all
    three need the same "story so far" view. Also drops the synthetic
    session-kickoff directive itself (build_session_kickoff_message) — it's
    marked internal/not-player-dialogue precisely so it's never shown to the
    player, but it's posted through the same message pipeline as a real
    player turn and gets persisted like one, so it has to be filtered out
    here rather than relying on callers to know to skip it."""
    result = []
    for m in messages:
        if not (isinstance(m, HumanMessage) or isinstance(m, AIMessage)):
            continue
        text = _extract_text(m.content).strip()
        if not text:
            continue
        if isinstance(m, HumanMessage) and text.startswith(_INTERNAL_DIRECTIVE_MARKER):
            continue
        result.append(m)
    return result


_NARRATOR_MAX_TURNS = 20  # narrative-only messages kept in the narrator's context


def _make_narrator_modifier(system_prompt: str, campaign_id: str, store: CampaignStore):
    """Returns a state modifier for the narrator node: narrative-only history
    (no raw tool traffic) plus this turn's mechanics resolution report,
    appended as a trailing directive. Async (unlike the plain
    _make_state_modifier) so it can live-inject which NPCs are traveling with
    the party (build_traveling_npcs_context) — the same reminder the mechanics
    node gets, needed here too since the narrator's own context window
    (_NARRATOR_MAX_TURNS) is narrative-only and can trim a companion's
    introduction out of view well before the campaign record forgets them."""
    system_msg = SystemMessage(content=system_prompt)

    async def modifier(state: dict) -> list[BaseMessage]:
        narrative = _narrative_messages(state["messages"])[-_NARRATOR_MAX_TURNS:]
        notes = state.get("mechanics_notes") or "No mechanical changes this turn."
        result = [system_msg] + narrative

        campaign = await store.load(campaign_id)
        if campaign:
            tctx = build_traveling_npcs_context(campaign)
            if tctx:
                result.append(HumanMessage(content=f"[TRAVELING NPCs — internal, not player dialogue]\n{tctx}"))

        directive = HumanMessage(
            content=(
                "[MECHANICAL RESOLUTION — internal, not player dialogue]\n"
                f"{notes}\n\nNarrate this turn for the player now."
            )
        )
        result.append(directive)
        return result

    return modifier


class DMState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mechanics_notes: str
    correction_note: str  # set by mechanics_node's guardrail for a one-shot retry nudge
    correction_count: int  # caps guardrail retries per PLAYER TURN, not per tool-call cycle —
                            # reset to 0 at the top of every stream_response invocation; a plain
                            # "haven't retried yet this cycle" check isn't enough, since a model
                            # that stops short after nearly every real tool call could otherwise
                            # get re-triggered on every single loop iteration and never terminate
    narrator_correction_note: str  # Session 0 only — separate budget/field from correction_note.
    narrator_correction_count: int  # Sharing the mechanics-side counter risks starvation: if a
                                     # mechanics-side correction already spent the turn's one retry
                                     # before the narrator ever runs, the narrator-side check
                                     # (structurally different failure class, more serious — wrong
                                     # game data reaching the player, not leaked debug syntax) would
                                     # have zero budget left exactly when it's needed.
    tool_error_count: int  # caps mechanics<->tools retries specifically for a bad/hallucinated
                            # tool call (unknown name or malformed args) — a separate budget from
                            # correction_count/narrator_correction_count, which cover different
                            # failure classes. Without this, a model that keeps emitting bad tool
                            # calls has no backstop but recursion_limit (60), which surfaces as an
                            # unhandled GraphRecursionError -> a generic "Lost connection to the
                            # model backend" error that silently drops the turn.
    lore_guardrail_count: int  # caps retries for Stage 2's lore guardrails (fabricated citation,
                                # abstention violation, spoiler leak) — a separate budget from
                                # correction_count for the same starvation-avoidance reason
                                # tool_error_count/narrator_correction_count already document: a
                                # combat/loot correction spending the turn's one correction_count
                                # retry shouldn't leave zero budget for a lore check later the
                                # same turn.
    stalled_turn_guardrail_count: int  # caps retries for _detect_stalled_non_player_turn_followup —
                                        # a separate budget from correction_count, split out after
                                        # observed live (2026-07-09) that it starved exactly the way
                                        # lore_guardrail_count's own doc comment predicts: one player
                                        # message can auto-continue through several combatants'
                                        # turns in a row (see the Combat prompt section), so an
                                        # earlier, unrelated correction_count-spending correction
                                        # (e.g. a missing combat-roll-backing catch a few combatants
                                        # earlier in the same response) left zero budget by the time
                                        # this genuinely distinct failure — a DM companion's turn
                                        # later in the same sequence — needed its own retry. Further
                                        # closed 2026-07-11 by narrator_node itself resetting this
                                        # (and the other three counters) to 0 on every loop-back to
                                        # mechanics for the next combatant — each combatant's turn is
                                        # now its own mechanics->narrator round-trip (see narrator_node
                                        # and TURN_BOUNDARY), so every counter genuinely starts fresh
                                        # per combatant rather than being shared across a whole
                                        # multi-combatant response. This field/split stays as a
                                        # backstop for the remaining case: several distinct guardrail
                                        # failures on the SAME combatant's turn within its own budget.
                                        # 2026-07-11: the "reset every combatant" part of this now
                                        # lives in stream_response's own per-combatant loop (each
                                        # iteration is its own astream_events call with these counters
                                        # freshly zeroed in its input) rather than in narrator_node —
                                        # see stream_response's docstring for why a shared
                                        # recursion_limit across a whole multi-combatant response had
                                        # the same starvation problem one level up.


# Every per-combatant-turn guardrail budget above, zeroed. stream_response
# spreads this into each loop iteration's input so every combatant's turn
# starts with fresh budgets (see its docstring for the starvation this
# prevents). Add any new DMState counter that must reset per combatant HERE
# — this constant is the single definition; the loop's two input literals
# both build from it. narrator_correction_count is deliberately absent: it
# self-resets in narrator_node.
_FRESH_GUARDRAIL_BUDGETS = {
    "correction_count": 0, "tool_error_count": 0,
    "lore_guardrail_count": 0, "stalled_turn_guardrail_count": 0,
}


# ── agent factory ──────────────────────────────────────────────────────────────

def _handle_any_tool_error(e: Exception) -> str:
    """LangGraph's ToolNode default (_default_handle_tool_errors) only
    converts ToolInvocationError (a Pydantic arg-validation failure) into a
    corrective ToolMessage — any other exception raised from inside a
    tool's own body (e.g. roll_notation's ValueError on bad dice notation,
    _helpers.py) re-raises and crashes the whole turn instead of giving the
    model a chance to self-correct. Annotating this handler's parameter as
    plain `Exception` (not ToolInvocationError) makes LangGraph's
    _infer_handled_types catch every exception type here, not just
    validation failures — verified against the installed
    langgraph.prebuilt.tool_node source."""
    if isinstance(e, ToolInvocationError):
        return e.message
    return f"Error: {e}\nPlease fix your mistakes and try the tool call again."


_TOOL_EXECUTION_TIMEOUT_S = 90.0


def _make_tool_node(tools: list[BaseTool], campaign_id: str) -> ToolNode:
    """Wrap tools in a ToolNode that serializes MUTATING tool calls per
    campaign, while letting read-only ones (_LOOKUP_ONLY_TOOLS) run
    concurrently with each other.

    LangGraph gathers parallel tool calls with asyncio.gather; the campaign
    store uses delete-all/reinsert so the last writer wins — a real writer
    still needs exclusive access (campaign_write_lock). But a single shared
    mutex for every tool, reads included, was confirmed live (2026-07-12) to
    turn one hung read-only call (search_rules, stuck in local Chroma/BM25
    retrieval well after its own Ollama embed call had already returned)
    into an apparent freeze of the OTHER two, otherwise-instant read-only
    calls in the same batch (get_campaign_summary, get_current_location) —
    they were never actually stuck themselves, just queued behind a lock a
    hung reader was never going to release. campaign_read_lock lets any
    number of reads proceed together; only an actual write contends with a
    read (see backend/locks.py for the full incident writeup).

    Also bounds every tool call's execution at _TOOL_EXECUTION_TIMEOUT_S —
    same discipline this codebase already applies to every direct Ollama
    client call (backend/llm.py's CHAT_TIMEOUT_S/EMBED_TIMEOUT_S) extended
    to a tool's own body, since the 2026-07-12 hang landed in local
    Chroma/BM25 code AFTER its Ollama call had already completed — nothing
    upstream would have caught it. A timeout here turns an indefinite
    silent stall into a normal, catchable tool error the existing
    _handle_any_tool_error/guardrail-retry machinery already knows how to
    react to. Note: cancelling asyncio.wait_for does NOT kill the
    underlying OS thread if the sync tool body is running via
    asyncio.to_thread (LangChain's default) — this bounds the turn, not a
    guaranteed clean kill of the stuck work, the same tradeoff already
    accepted for Ollama calls elsewhere in this codebase.
    """

    async def _serialize(request, execute):
        tool_name = request.tool_call["name"]
        lock = campaign_read_lock(campaign_id) if tool_name in _LOOKUP_ONLY_TOOLS else campaign_write_lock(campaign_id)
        async with lock:
            return await asyncio.wait_for(execute(request), timeout=_TOOL_EXECUTION_TIMEOUT_S)

    return ToolNode(tools, awrap_tool_call=_serialize, handle_tool_errors=_handle_any_tool_error)


_STATE_CHANGE_TOOLS = {
    "update_character_hp", "update_monster_hp", "add_condition", "remove_condition",
    "advance_initiative", "end_encounter",
}


def _messages_since_last_turn_boundary(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages appended since the current combatant-turn's resolution
    started — walks backward until (excluding) the most recent HumanMessage
    OR tool-call-free AIMessage (a completed narrator reply), whichever is
    more recent. This is the mechanics loop's own scratch-work: tool-call
    AIMessages and their ToolMessage results.

    Named/scoped this way (not just "since last human") because narrator_node
    can now loop back to mechanics multiple times per player message — once
    per combatant's turn, see get_agent()'s narrator_node — and each of its
    AIMessages becomes a PERMANENT part of `messages` (a real chat bubble the
    player already saw), not scratch. The old "since last human" boundary
    would sweep up a prior combatant's already-final narrator reply and try
    to RemoveMessage it, corrupting history. A tool-call-free AIMessage can
    only ever be a completed narrator reply here: mechanics_node's own
    non-tool-calling response (the resolution report) is captured into
    `notes` via _extract_text and never itself appended to `messages` before
    reaching narrator, so this can't accidentally stop mid-turn."""
    since: list[BaseMessage] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage) and not m.tool_calls:
            break
        since.append(m)
    since.reverse()
    return since


def _tools_called_since_last_turn_boundary(messages: list[BaseMessage]) -> set[str]:
    """Tool names called since the current combatant-turn's resolution
    started. Used by _detect_missing_followup to see what the mechanics
    model has already done this turn without needing extra state to track
    it."""
    called: set[str] = set()
    for m in _messages_since_last_turn_boundary(messages):
        if isinstance(m, AIMessage) and m.tool_calls:
            called.update(c["name"] for c in m.tool_calls)
    return called


_MAX_TOOL_ERROR_RETRIES = 2


def _last_tool_batch_had_error(messages: list[BaseMessage]) -> bool:
    """True if the most recent tools-node round (the trailing run of
    ToolMessages at the end of `messages`) contains an error result —
    either ToolNode's unknown-tool-name rejection or a validation/execution
    failure caught by _handle_any_tool_error, both marked status="error".
    Stops at the first non-ToolMessage walking backward, since that's the
    AIMessage(tool_calls) that started this batch."""
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            if getattr(m, "status", "success") == "error":
                return True
            continue
        break
    return False


def _orphaned_interrupted_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Detects a PRIOR turn that was interrupted mid-execution — a process
    crash/restart landing between the mechanics node checkpointing a
    tool-call decision and the tools node actually executing it. That leaves
    scratch (the run of messages between the turn's HumanMessage and the
    next one) ending in an AIMessage with unresolved tool_calls: no
    ToolMessage results, no narrator reply ever followed. Left in place,
    this turn's mechanics call would hand the model (and Ollama's chat API)
    a malformed sequence — an assistant tool_calls message with no matching
    tool results before the next human turn — which errors outright rather
    than degrading gracefully. Returns [] for a turn that resolved normally
    (scratch ending in a tool-call-free narrator AIMessage) or if there's no
    prior turn to check."""
    humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(humans) < 2:
        return []
    # Everything after the last turn boundary before the latest human. The
    # boundary rule (HumanMessage OR completed narrator reply) is OWNED by
    # _messages_since_last_turn_boundary — reusing it rather than re-walking
    # here matters because stream_response's multi-combatant loop can leave
    # several completed narrator AIMessages (real chat bubbles the player
    # already saw) between two HumanMessages, and a cleanup using a stale
    # hand-rolled copy of the rule would RemoveMessage those replies too,
    # silently deleting them from the persisted transcript.
    scratch = _messages_since_last_turn_boundary(messages[:humans[-1]])
    if scratch and isinstance(scratch[-1], AIMessage) and scratch[-1].tool_calls:
        return scratch
    return []


def _recover_orphaned_turn(messages: list[BaseMessage]) -> Command | None:
    """The recovery half of _orphaned_interrupted_turn: the RemoveMessage
    cleanup Command both mechanics nodes must issue, before anything else,
    for a thread that died mid-turn. One implementation so a future recovery
    tweak (extra state reset, logging) can't land in only one node."""
    orphaned = _orphaned_interrupted_turn(messages)
    if not orphaned:
        return None
    return Command(
        goto="mechanics",
        update={"messages": [RemoveMessage(id=m.id) for m in orphaned if m.id]},
    )


def _turn_was_explicitly_ended(messages: list[BaseMessage]) -> bool:
    """True if this turn's scratch already advanced (or asked to advance) the
    initiative pointer — either a direct advance_initiative call, or a
    resolve_attack/resolve_saving_throw call with end_turn=True. Distinct from
    _tools_called_since_last_turn_boundary because this needs each call's ARGS, not
    just its name. Doesn't specially recognize a resolve_pending_action call
    honoring a stored attacker_wanted_end_turn — that intent lives in
    persisted PendingAction state, not in this turn's own call args; accepted
    as a minor blind spot since that path is rare and resolve_pending_action's
    own docstring already tells the model when a turn still needs finishing."""
    for m in _messages_since_last_turn_boundary(messages):
        if not (isinstance(m, AIMessage) and m.tool_calls):
            continue
        for c in m.tool_calls:
            if c["name"] == "advance_initiative":
                return True
            if c["name"] in ("resolve_attack", "resolve_saving_throw") and c.get("args", {}).get("end_turn"):
                return True
    return False


_RESOLUTION_TOOLS = {"resolve_attack", "resolve_saving_throw", "cast_spell", "resolve_pending_action"}


def _detect_missing_followup(state: DMState, campaign: Campaign) -> str | None:
    """Catches the mechanics model stopping mid-resolution during combat —
    observed live: rolling an attack/damage and narrating a hit or a turn
    passing without ever calling the tool that actually makes it true, or in
    the worst case resolving an action with no tool calls — no roll — at all.
    Only checked during an active encounter, since that's the only place this
    failure was observed and a "no tool calls this turn" turn is completely
    normal otherwise (pure roleplay, a question, etc.).

    resolve_attack/resolve_saving_throw/cast_spell/resolve_pending_action
    apply their own damage/condition state changes atomically by construction
    — the "rolled but didn't apply" half of the original bug can't happen for
    a turn using them. What's left to catch for those tools is narrower: did
    the turn actually advance (see _turn_was_explicitly_ended)? The older
    "bare roll_dice with no _STATE_CHANGE_TOOLS follow-up" check stays for
    roll_dice's own fallback use (a raw roll not tied to any resolution
    tool). Like every detector in the chain, reads the caller's one shared
    live snapshot (see mechanics_node) rather than loading its own."""
    if not campaign.active_encounter or not campaign.active_encounter.is_active:
        return None
    called = _tools_called_since_last_turn_boundary(state["messages"])
    if not called:
        return (
            "You're resolving a turn during an active encounter but made no tool calls at "
            "all. Never fabricate a combat outcome — call resolve_attack/resolve_saving_throw/"
            "resolve_check/cast_spell (or roll_dice as a last resort) for the action being "
            "attempted, before writing your resolution report."
        )
    if called & _RESOLUTION_TOOLS:
        if not _turn_was_explicitly_ended(state["messages"]):
            return (
                "You resolved an action this turn during an active encounter, but the "
                "turn was never advanced — no advance_initiative call, and no "
                "resolve_attack/resolve_saving_throw call had end_turn=True. If the "
                "acting combatant's whole turn (including any bonus action) is fully "
                "resolved, call advance_initiative now, or redo the last resolution with "
                "end_turn=True. If more of this turn's action economy is still coming "
                "(a bonus-action attack, a Hasted extra action), proceed as before."
            )
        return None
    if "roll_dice" in called and not (called & _STATE_CHANGE_TOOLS):
        return (
            "You rolled dice this turn during an active encounter but didn't follow "
            "through — no HP/condition change and the initiative order wasn't advanced. "
            "If the roll succeeded, apply the resulting damage/effect now "
            "(update_character_hp / update_monster_hp / add_condition); once the acting "
            "combatant's action is fully resolved, call advance_initiative. If truly "
            "nothing changed as a result, proceed as before."
        )
    return None


_COMBAT_ROLL_TOOLS = {
    "resolve_attack", "resolve_saving_throw", "cast_spell",
    "resolve_pending_action", "roll_dice",
}

# Superset for the narrator-side VERIFIED ROLLS capture (mechanics_node and
# the chargen mechanics node): every tool whose return string embeds a
# roll_notation()/d20 breakdown the narrator must relay verbatim.
# Deliberately wider than _COMBAT_ROLL_TOOLS, which pairs with
# _COMBAT_ROLL_MENTION_RE to answer a different question — "does a die roll
# back this turn's combat narration?" — where a mere ability check must NOT
# count as backing for a narrated attack. Skill checks and death saves are
# exactly the roll types most easily misreported by the mechanics model's
# free-text notes, so leaving them out of the capture (as originally
# shipped) silently dropped their real breakdowns while the tools most
# likely to be double-checked kept theirs.
_VERIFIED_ROLL_TOOLS = _COMBAT_ROLL_TOOLS | {"resolve_check", "resolve_death_save"}

# Chargen additionally captures roll_ability_scores — chargen-only (not an
# in-game resolution tool, so not in _VERIFIED_ROLL_TOOLS itself) but with
# the exact same failure mode: its per-score breakdown (e.g. "Roll 1:
# [4, 5, 2, 6] → drop 2 → 15") is just as easy for the narrator to silently
# paraphrase away.
_CHARGEN_VERIFIED_ROLL_TOOLS = _VERIFIED_ROLL_TOOLS | {"roll_ability_scores"}


def _verified_rolls_note(scratch: list[BaseMessage], tools: set[str] = _VERIFIED_ROLL_TOOLS) -> str:
    """The VERIFIED_ROLLS_MARKER block appended to a mechanics resolution
    report: verbatim tool output for this turn's dice, captured before the
    scratch purge deletes it — otherwise the only numbers that ever reach
    the narrator are whatever the mechanics model's own free-text notes
    claim happened, a paraphrase of a paraphrase with nothing to check it
    against. roll_notation() (backend/tools/_helpers.py) computes the real
    breakdown deterministically and every capture-listed tool's return
    string already includes it (e.g. "damage [1] +4 = 5 piercing").
    Confirmed live, 2026-07-11: with no such backstop there was no way to
    tell whether a suspiciously low damage total the player questioned was
    an actual unlucky roll or the mechanics/narrator hop silently dropping
    or inventing one. Returns "" when nothing captured ran this turn.

    Both narrator prompts key on the literal marker phrase (prompts.py's
    VERIFIED_ROLLS_MARKER); this helper is the ONLY code that emits it — it
    used to be copy-pasted into both mechanics nodes, which meant a wording
    tweak in one silently broke the verbatim-relay contract for the other."""
    rolls = [
        _extract_text(m.content) for m in scratch
        if isinstance(m, ToolMessage) and m.content and m.name in tools
    ]
    if not rolls:
        return ""
    return (
        f"\n\n{VERIFIED_ROLLS_MARKER} (exact tool output — the narrator "
        "must use these numbers verbatim in its 🎲 lines, never recompute, "
        "round, or otherwise alter them):\n"
        + "\n".join(f"- {r}" for r in rolls)
    )

# Read-only/lookup tools — the ONLY tools excluded from the verified-state-
# changes capture below. Deliberately an exclusion list rather than an
# allowlist of mutating tools: every mutating tool in backend/tools/*.py
# already belongs in the capture (its return string is the real, deterministic
# fact — an HP total, an item grant, an encounter closing), and a new tool
# added later is far more likely to mutate state than to be a pure lookup, so
# defaulting new tools INTO the capture is the safe direction for this list to
# be wrong in. An allowlist defaults new tools OUT, silently recreating the
# exact "narrated but not verified" hole this mechanism exists to close (see
# _verified_state_changes_note's docstring for the incident that motivated
# generalizing this beyond dice rolls).
_LOOKUP_ONLY_TOOLS = {
    "get_party_status", "get_character", "get_unassigned_loot", "get_npc",
    "get_campaign_summary", "get_active_quests", "search_campaign_history",
    "search_rules", "search_adventure_literal", "lookup_entity", "search_lore",
    "get_current_location", "get_travel_estimate",
}

_STATE_CHANGE_EXCLUDE_TOOLS = _VERIFIED_ROLL_TOOLS | _LOOKUP_ONLY_TOOLS


def _verified_state_changes_note(scratch: list[BaseMessage]) -> str:
    """The VERIFIED_CHANGES_MARKER block appended to a mechanics resolution
    report: verbatim tool output for every NON-roll mutation this turn — HP,
    conditions, items, currency, encounter open/close, NPC/quest state, and
    so on. Generalizes _verified_rolls_note's proven pattern (verbatim tool
    output as the narrator's only ground truth, instead of trusting the
    mechanics model's own free-text notes) beyond dice specifically.

    Motivating incident, 2026-07-12: the mechanics model narrated Tarvokk
    executing three bound, captured goblins — flavor 🎲 dice text and all —
    with no update_monster_hp call behind it. The narrator, which has no
    tools and cannot check anything, dutifully wrote prose describing three
    deaths that never happened; the campaign's real state still had all
    three goblins alive days later. A regex guardrail
    (_detect_missing_monster_death_followup) now also nudges the mechanics
    model to make the call in the first place, but that's a retry budget of
    one per player turn — this note is the backstop for when that retry
    doesn't land: the narrator is instructed (see prompts.py) to only
    confirm a mechanical outcome that actually appears here, so even an
    unfixed fabrication in the mechanics model's own notes can't reach the
    player as if it were real."""
    changes = [
        _extract_text(m.content) for m in scratch
        if isinstance(m, ToolMessage) and m.content and m.name not in _STATE_CHANGE_EXCLUDE_TOOLS
    ]
    if not changes:
        return ""
    return (
        f"\n\n{VERIFIED_CHANGES_MARKER} (exact tool output — ground truth for "
        "every non-roll mechanical change this turn: HP, conditions, items, "
        "currency, encounter status, and similar. Never narrate a mechanical "
        "outcome — a death, an item gained, combat ending — that isn't listed "
        "here, and never alter these facts when writing prose):\n"
        + "\n".join(f"- {c}" for c in changes)
    )


_COMBAT_ROLL_MENTION_RE = re.compile(
    r"\battack roll\b|\bdamage roll\b|\bsaving throw\b|\brolls?\s+(?:to hit|damage)\b|"
    r"\b\d+\s*(?:slashing|piercing|bludgeoning|fire|cold|lightning|acid|poison|psychic|"
    r"radiant|necrotic|force|thunder)\s+damage\b|"
    # Prose narrating a physical strike with no accompanying number or roll
    # phrase — e.g. "Elara stepped in and stabbed the goblin" — the patterns
    # above only fire on an explicit "attack roll"/"X damage" mention, so a
    # vague action verb like this slipped through with zero resolve_attack
    # call behind it (a DM companion's whole turn narrated with no dice at
    # all). Unambiguous strike verbs (stab/slash/pierce/lunge/skewer/gash)
    # fire on their own; ambiguous ones (strike/hit/swing/drive/plunge) only
    # fire alongside "at/into/through" a target, since those verbs are
    # common in non-combat, non-literal prose too ("the realization hits
    # you", "the storm strikes the coast"). Only the FIRST preposition after
    # the verb is tested — the tempered gap (?:(?!prep)[^.]) can't skip past
    # one to a later one, otherwise "swings into place at the end"
    # backtracks to "at the end" and defeats the object check — and the
    # negative lookahead after it drops the two idiomatic objects most
    # plausible in a real resolution report: a scene line's "swings into
    # place/position" and a rules quote's "hits at half/full damage". Known
    # residual false positives ("plunges into the river", "struck at dawn")
    # are accepted: they cost at most one guardrail retry per player turn
    # (correction_count), and every tighter form tried loses real recall —
    # e.g. requiring a capitalized target drops "swings his club at the
    # goblin", the common generic-monster case.
    r"\b(?:stabs?|stabbed|slashes?|slashed|slices?|sliced|pierces?|pierced|"
    r"lunges?|lunged|skewers?|skewered|gashes?|gashed)\b|"
    r"\b(?:strikes?|struck|hits?|swings?|swung|drives?|drove|plunges?|plunged)\b"
    r"(?:(?!\b(?:at|into|through)\b)[^.]){0,30}"
    r"\b(?:at|into|through)\b(?!\s+(?:place|position|half|full)\b)",
    re.IGNORECASE,
)


def _detect_missing_combat_roll_followup(notes: str, called: set[str]) -> str | None:
    """Catches the mechanics model's own resolution report narrating a combat
    roll (an attack roll, damage, a saving throw) with NO backing
    resolve_attack/resolve_saving_throw/cast_spell/resolve_pending_action/
    roll_dice call this turn. Observed live: an entire multi-message "combat"
    (attack rolls, damage, a fabricated "Round 1 / Initiative Order" block)
    was narrated with zero tool calls — no monster, no encounter, nothing
    ever actually created or persisted. Deliberately NOT gated to an active
    encounter, unlike _detect_missing_followup — that gate is exactly what
    let this slip through: the bug is precisely that no encounter/monster
    ever gets created in the first place, so there's nothing to gate on."""
    if called & _COMBAT_ROLL_TOOLS:
        return None
    # An OOC report is a rules/meta answer, not fiction — quoted rule text
    # is legitimately full of combat vocabulary ("24 fire damage", "attack
    # roll", "the spell hits at half damage") with no roll behind it, so
    # every arm of the mention regex false-fires on it. The mechanics prompt
    # requires OOC reports to START with the literal marker (see OOC_MARKER
    # and the OOC section in prompts.py), so a prefix check is the whole
    # convention.
    if notes.lstrip().startswith(OOC_MARKER):
        return None
    if not _COMBAT_ROLL_MENTION_RE.search(notes):
        return None
    return (
        "Your resolution report describes a combat roll (an attack, damage, or a "
        "saving throw), but no resolve_attack / resolve_saving_throw / cast_spell / "
        "resolve_pending_action / roll_dice call backs it up this turn. Never narrate "
        "a die roll or damage that didn't actually happen through a real tool call. "
        "If this is combat, register any new opponents with create_monster and call "
        "start_encounter if needed, then resolve the action for real before reporting it."
    )


_LOOT_TOOLS = {
    "update_character_currency", "add_item_to_character", "create_magic_item", "reveal_loot",
    # end_encounter now rolls and reveals post-combat loot automatically (see
    # backend/tools/loot_generator.py) — its own tool result is a valid backing
    # call for a narrated gain, same as calling reveal_loot by hand.
    "end_encounter",
}

# Gain-verbs broadened beyond "finds/receives" to cover mundane handoffs/pickups
# ("snatches the weapon", "recovers his shortsword") that the original verb list
# missed entirely — observed live: a companion picking up a dropped weapon, and
# a body search turning up a pouch/ring/shortsword, both slipped through with no
# add_item_to_character call because no word here matched.
_LOOT_GAIN_VERBS = (
    r"gains?|receives?|finds?|found|discovers?|discovered|recovers?|retrieves?|"
    r"grabs?|snatch(?:es)?|takes?|strips?|pockets?|picks?\s+up|hands?\s+over|"
    r"claims?|stows?|packs?\s+away|tucks?\s+away"
)
# Mundane gear nouns so a plain pickup/handoff of ordinary equipment (not just
# currency/treasure/a magic item) still trips the guardrail.
_MUNDANE_GEAR_NOUNS = (
    r"sword|shortsword|longsword|greatsword|dagger|blade|weapon|pouch|ring|coins?|purse|"
    r"bow|shortbow|longbow|crossbow|quiver|arrows?|bolts?|key|whetstone|rations?|meat|"
    r"scroll|potion|gem|amulet|cloak|wand|staff|rod|shield|armor|helm|helmet|boots|gloves|"
    r"cape|belt|bracers?|torch|lantern|rope|tool|kit|instrument|trinket|coin|treasure|"
    r"satchel|bag|sack|case|chest|box|crate|coffer|backpack|pack|contents?"
)

_LOOT_MENTION_RE = re.compile(
    r"\b\d+\s*(?:gp|sp|cp|pp|ep)\b"
    r"|\b(?:gold|silver|copper|platinum|electrum)\s+pieces?\b"
    rf"|\b(?:{_LOOT_GAIN_VERBS})\b[^.]{{0,40}}\b"
    rf"(?:gp|gold|coin|coins|treasure|gem|item|loot|{_MUNDANE_GEAR_NOUNS})\b"
    r"|\bloot\s*:"
    # Magic item shape ("+1 Longsword", "+2 studded leather armor") — a bonus
    # followed by either a capitalized name or a common equipment noun, not
    # just any word, so an unrelated "+5 to his attack roll" doesn't match.
    r"|\+\d+\s+(?:(?-i:[A-Z]\w*)|studded|leather|chain|plate|scale|splint|banded|hide|"
    r"padded|longsword|shortsword|greatsword|dagger|mace|axe|hammer|bow|armor|"
    r"shield|ring|wand|staff|rod|amulet|cloak|boots|gloves|potion|scroll)\b"
    # Plain item grant with no numeric prefix ("Tarvokk: +Crude Iron Key",
    # "+Small Whetstone") — the resolution-report convention (prompts.py's
    # "Sir Valiant: +3 gp" / "Mira Swiftfoot: +1 Ancient Sunburst Coin"
    # examples) doesn't actually require a leading digit for a non-currency
    # item, so a "+ItemName" grant with no number slipped past the
    # digit-anchored magic-item branch above entirely.
    r"|\+\s*(?-i:[A-Z])\w*(?:\s+(?-i:[A-Z])\w*){0,3}\b",
    re.IGNORECASE,
)


def _detect_missing_loot_followup(notes: str, called: set[str]) -> str | None:
    """Catches the mechanics model's own resolution report claiming a
    currency/item gain (see _MECHANICS_BASE's Loot section — the "Sir
    Valiant: +3 gp" line shape) without update_character_currency /
    add_item_to_character / create_magic_item actually having been called
    this turn. Unlike _detect_missing_followup this is NOT gated to active
    combat — the original incident this catches (a narrated gold find with
    the party's actual currency left unchanged) happened during ordinary
    exploration, not combat, so gating it the same way would leave the exact
    bug it's meant to catch unaddressed."""
    if called & _LOOT_TOOLS:
        return None
    if not _LOOT_MENTION_RE.search(notes):
        return None
    return (
        "Your resolution report mentions a currency or item gain, but no "
        "update_character_currency / add_item_to_character / create_magic_item "
        "call backs it up this turn. If something was actually found, call the "
        "real tool now with the correct character and amount/item before "
        "reporting it. If nothing was actually gained, rewrite the report "
        "without implying one."
    )


_KILL_VERBS = (
    r"kills?|killed|slays?|slew|slain|executes?|executed|finishes?\s+off|finished\s+off|"
    r"slits?\s+(?:his|her|their|its)\s+throat|slit\s+(?:his|her|their|its)\s+throat|"
    r"dies?|died|dead|dispatche[sd]|puts?\s+(?:him|her|them|it)\s+down|put\s+(?:him|her|them|it)\s+down"
)

_MONSTER_DEATH_MENTION_RE = re.compile(rf"\b(?:{_KILL_VERBS})\b", re.IGNORECASE)

_MONSTER_DEATH_TOOLS = {"update_monster_hp", "resolve_attack", "end_encounter"}


def _detect_missing_monster_death_followup(notes: str, called: set[str]) -> str | None:
    """Catches the mechanics model's own resolution report narrating a
    monster's death (a killing blow, or executing/finishing off an
    already-helpless one) with no update_monster_hp/resolve_attack/
    end_encounter call backing it this turn. Deliberately NOT gated to an
    active encounter — the incident this catches (three bound, captured
    goblins narrated as executed, "🎲 Tarvokk — Dagger: [4] piercing" flavor
    text and all, HP left untouched in the DB) happened well after the fight
    that captured them had already (functionally) ended, so gating on
    active_encounter would leave exactly this case uncaught, same reasoning
    as _detect_missing_loot_followup's own un-gated design."""
    if called & _MONSTER_DEATH_TOOLS:
        return None
    if not _MONSTER_DEATH_MENTION_RE.search(notes):
        return None
    return (
        "Your resolution report narrates a monster's death (a killing blow, or "
        "executing/finishing off an already-helpless one), but no update_monster_hp "
        "call backs it up this turn. Call update_monster_hp now to actually bring it "
        "to 0 HP before reporting the death — narrating a kill without it leaves the "
        "monster alive in the campaign's real state."
    )


_COMBAT_ENDED_MENTION_RE = re.compile(
    r"\b(?:combat|the fight|the battle|the encounter)\s+(?:has\s+)?"
    r"(?:ended|ends|is\s+over|concluded|finished)\b"
    # Capture/surrender/incapacitation endings — a fight can be functionally
    # over without anyone ever saying "the fight is over": the party binds,
    # captures, or otherwise neutralizes every hostile combatant, and the
    # narration moves straight into post-combat roleplay (interrogation,
    # looting, travel) without end_encounter ever being called. Observed
    # live, 2026-07-12: three captured/bound goblins left active_encounter
    # stuck True through an interrogation, a moral debate, and an execution,
    # silently mis-gating every ordinary roleplay/travel turn afterward as
    # "no tool calls during an active encounter" (see
    # _detect_missing_followup).
    r"|\b(?:bound|captured|subdued|restrained|surrender(?:s|ed)?|"
    r"no\s+longer\s+(?:a\s+threat|hostile|fighting)|"
    r"(?:all|every one of them?)\s+(?:are|is)\s+(?:unconscious|bound|captured))\b",
    re.IGNORECASE,
)


def _detect_missing_end_encounter_followup(
    notes: str, called: set[str], campaign: Campaign
) -> str | None:
    """Catches the mechanics model's own resolution report declaring combat
    over (e.g. "the combat has ended") with no end_encounter call backing it
    this turn. Without this, active_encounter.is_active stays True forever —
    the encounter/initiative UI keeps showing a fight that's conversationally
    over, and the automatic post-combat loot roll (end_encounter's job, see
    loot_generator.py) never fires. Symmetric to
    _detect_missing_loot_followup/_detect_missing_combat_roll_followup: a
    narrated claim vs. an actually-called tool, same "don't say it happened
    unless you made it happen" principle applied to encounter closure.
    Observed live, 2026-07-11: the model resolved a killing blow, narrated
    "the combat has ended," moved straight into post-combat roleplay (loot
    search, tying up prisoners) — and never called end_encounter at all."""
    if "end_encounter" in called:
        return None
    if not _COMBAT_ENDED_MENTION_RE.search(notes):
        return None
    if not campaign.active_encounter or not campaign.active_encounter.is_active:
        return None
    return (
        "Your resolution report describes combat as over, but no end_encounter call "
        "backs it up this turn — the encounter is still marked active, so the combat/"
        "initiative UI will keep showing it as ongoing and no post-combat loot roll "
        "has happened. If every enemy is actually dead, fled, or incapacitated, call "
        "end_encounter now (with xp_awarded if applicable) before reporting the fight "
        "as finished, then continue resolving whatever the player asked for next."
    )


_COMBAT_RESOLUTION_TOOLS = {"resolve_attack", "resolve_saving_throw"}


def _detect_missing_encounter_followup(state: DMState, campaign: Campaign) -> str | None:
    """Catches resolve_attack/resolve_saving_throw landing against a Monster that
    survived (current_hp > 0) with no active_encounter backing it — observed
    live: a multi-guard ambush resolved entirely through bare resolve_attack
    calls, with no start_encounter ever called, no initiative order, no round
    tracking. A single decisive blow against a target that dies or is otherwise
    fully resolved is fine standalone (the bright-line rule in the Combat
    prompt section covers that); this only fires when the target is still
    capable of acting back and nothing has formalized the fight."""
    calls = [
        c for m in _messages_since_last_turn_boundary(state["messages"])
        if isinstance(m, AIMessage) and m.tool_calls
        for c in m.tool_calls
        if c["name"] in _COMBAT_RESOLUTION_TOOLS
    ]
    if not calls:
        return None
    if campaign.active_encounter and campaign.active_encounter.is_active:
        return None
    names: set[str] = set()
    for c in calls:
        args = c.get("args", {})
        if "target_name" in args:
            names.add(args["target_name"])
        names.update(args.get("target_names", []))
    survivors = [m.name for m in campaign.monsters if m.name in names and m.current_hp > 0]
    if not survivors:
        return None
    return (
        f"You resolved an attack/save against {', '.join(survivors)} outside any "
        "active encounter, and they survived — this could become an ongoing fight. "
        "If this is truly a single decisive blow that's now fully resolved, proceed. "
        "Otherwise, formalize combat now: call create_monster for every hostile actor "
        "not already registered, then start_encounter, before resolving anything further."
    )


def _live_current_turn(campaign: Campaign) -> tuple[InitiativeEntry, bool] | None:
    """(current initiative entry, is_player_controlled) for whoever's turn it
    live-is right now, or None if there's no active encounter / no pending
    reaction to resolve first. Shared ground-truth lookup for both
    _detect_stalled_non_player_turn_followup (forces real resolution) and
    _next_turn_ground_truth_note (tells the narrator the fact directly,
    regardless of whether that guardrail's bounded retry already fired) —
    see both docstrings for why neither alone is sufficient."""
    enc = campaign.active_encounter
    if not enc or not enc.is_active or enc.pending_action:
        return None
    current = next((e for e in enc.initiative_order if e.is_current_turn), None)
    if not current:
        return None
    if current.combatant_type == CombatantType.CHARACTER:
        char = find_char(campaign, current.name)
        is_player = bool(char and char.is_player_controlled)
    else:
        is_player = False
    return current, is_player


def _detect_stalled_non_player_turn_followup(campaign: Campaign) -> str | None:
    """Catches the mechanics model stopping to prompt the (human) player when
    the combatant actually due to act next — per live initiative_order — is a
    monster, a DM-controlled companion, or an NPC. Nobody is there to act
    until the model resolves it itself; the Combat prompt section already
    instructs auto-continuing through every non-player turn in one response
    (see prompts.py's "turn of this conversation vs turn of initiative" rule),
    this is the deterministic backstop for when that instruction doesn't get
    followed — same shape as the loot/encounter guardrails above. Observed
    live: the model stopped and asked the human player to act for a DM
    companion (a different character than the human's own), leaving the
    actual player-controlled turn never reached. Exempt whenever there's a
    pending_action — that's a real reaction prompt legitimately awaiting the
    player's decision, not a stall."""
    hit = _live_current_turn(campaign)
    if not hit:
        return None
    current, is_player = hit
    if is_player:
        return None
    return (
        f"It's currently {current.name}'s turn ({current.combatant_type.value}, not "
        "player-controlled) — nobody is waiting to act until you resolve it yourself. "
        "Call advance_initiative and resolve their action now with a sensible tactical "
        "choice, then write your resolution report and stop — don't stop to ask the "
        "player for input on a turn that isn't theirs. Just this one combatant's turn; "
        "the game brings you back automatically for the next one if it's also "
        "non-player, per the Combat section's auto-continuation rule."
    )


def _next_turn_ground_truth_note(campaign: Campaign) -> str | None:
    """Deterministic "whose turn is it, really" fact, appended to the
    resolution report unconditionally (no retry budget — this never loops,
    just annotates) right before the narrator sees it. Exists because
    _detect_stalled_non_player_turn_followup's fix wasn't sufficient on its
    own: observed live (2026-07-09) that even with its own dedicated retry
    budget, a single long auto-continued response can still stall on a
    SECOND non-player turn after the guardrail's one retry already got spent
    correcting a first, unrelated mistake earlier in the same response —
    three different DM companions (Elara, Kaelen, Thrainna) each independently
    got asked directly across one fight, and the model's own free-text
    tracking of "whose turn" drifted every time rather than reading the live
    fact it already has better access to. This can't loop indefinitely
    (unlike the guardrail, it has no bounded-retry mechanism to force a real
    re-resolution), so it's a narration-correctness backstop, not a
    substitute for the guardrail actually resolving the turn — it guarantees
    the player is never told a wrong or premature "X, you are up" even in the
    worst case where a companion's turn is still technically unresolved when
    this response ends; the player's own next message starts a fresh
    stalled_turn_guardrail_count budget that can then force the real fix."""
    hit = _live_current_turn(campaign)
    if not hit:
        return None
    current, is_player = hit
    if is_player:
        return (
            f"\n\n[GROUND TRUTH — it is now {current.name}'s turn (player-controlled). "
            "End your reply prompting exactly this character, by this exact name, for "
            "their action. Do not address any other character.]"
        )
    return (
        f"\n\n[GROUND TRUTH — it is still {current.name}'s turn "
        f"({current.combatant_type.value}, NOT player-controlled). No player input is "
        "being waited on right now — do not end your reply asking the player what they "
        "do. Narrate the scene continuing/holding instead; the DM will resolve "
        f"{current.name}'s action on the next pass.]"
    )


# ── Stage 2 lore guardrails ─────────────────────────────────────────────────
# Same guardrail-chain shape as the combat/loot detectors above, but for the
# lore/retrieval path specifically: forced citations, abstention enforcement,
# and spoiler-tier non-leakage — the report's CRAG/Self-RAG discipline
# layered onto the existing mechanics loop rather than a new graph node. Given
# their own separate lore_guardrail_count budget (see DMState) rather than
# sharing correction_count, for the same starvation-avoidance reason
# tool_error_count/narrator_correction_count already document: a combat/loot
# correction spending the turn's one correction_count retry shouldn't zero
# out the budget a lore check needs later the same turn.

_LORE_TOOLS = {"search_lore", "lookup_entity", "search_rules"}


def _lore_tool_outputs_since_last_human(messages: list[BaseMessage]) -> list[str]:
    """Contents of this turn's search_lore/lookup_entity/search_rules
    ToolMessages — the source of truth for citation verification and
    spoiler-leak detection below."""
    return [
        _extract_text(m.content)
        for m in _messages_since_last_turn_boundary(messages)
        if isinstance(m, ToolMessage) and getattr(m, "name", None) in _LORE_TOOLS
    ]


_CHUNK_ID_CLAIM_RE = re.compile(r'chunk_id:\s*([a-f0-9]+)', re.IGNORECASE)


def _detect_uncited_or_invalid_lore_claim(notes: str, messages: list[BaseMessage]) -> str | None:
    """Extracts every 'chunk_id: X' token the mechanics model wrote into its
    own resolution notes (i.e. a claimed citation), and checks each against
    the set of chunk_ids ACTUALLY present in this turn's search_lore/
    lookup_entity/search_rules tool outputs. Fires only on a genuinely
    fabricated citation — an id that was never actually returned this turn —
    not on the mere presence/absence of a citation (that's
    _detect_abstention_violation's job, right below)."""
    cited = set(_CHUNK_ID_CLAIM_RE.findall(notes))
    if not cited:
        return None
    real_ids: set[str] = set()
    for output in _lore_tool_outputs_since_last_human(messages):
        real_ids.update(_CHUNK_ID_CLAIM_RE.findall(output))
    fake = cited - real_ids
    if not fake:
        return None
    return (
        f"Your resolution notes cite chunk_id(s) {', '.join(sorted(fake))} that were NOT "
        "actually returned by any search_lore/lookup_entity/search_rules call this turn — "
        "a fabricated citation. Only cite a chunk_id that genuinely appeared in a tool "
        "result this turn, or drop the citation and rephrase as your own improvisation."
    )


# Heuristic, same spirit as _LOOT_MENTION_RE above: a regex can't perfectly
# detect "this is a confident, ungrounded factual claim," but a reasonable
# set of phrasings that typically accompany one is enough to catch the real
# failure mode (a rules/lore fact stated as if verified, with nothing behind
# it) without needing an extra LLM judge call on every single turn.
_LORE_CLAIM_HINT_RE = re.compile(
    r"\b(?:according to the|the rules state|the book (?:says|states)|per the (?:phb|dmg|"
    r"player'?s handbook|dungeon master'?s guide|monster manual)|has an ac of \d|"
    r"\d+\s*hit points?\b|spell save dc (?:is|of) \d|is a(?:n)? cr \d)\b",
    re.IGNORECASE,
)

_ABSTENTION_RE = re.compile(
    r"\b(?:don'?t have (?:that|this) in|not (?:in|covered by) the (?:sources?|books?|text)|"
    r"sources? don'?t (?:cover|mention)|no (?:relevant )?(?:rules?|lore) (?:found|available)|"
    r"can'?t find (?:that|this) in|nothing in the (?:sources?|books?|text))\b",
    re.IGNORECASE,
)


def _detect_abstention_violation(notes: str, called_this_turn: set[str]) -> str | None:
    """Catches the mechanics model stating a specific rules/lore fact with NO
    backing search_rules/search_lore/lookup_entity call at all this turn. An
    explicit 'not in the source' abstention SATISFIES this check — that's the
    report's required behavior, not a violation — so this only fires on a
    confident, ungrounded claim, never on a genuine abstention."""
    if called_this_turn & _LORE_TOOLS:
        return None
    if _ABSTENTION_RE.search(notes):
        return None
    if not _LORE_CLAIM_HINT_RE.search(notes):
        return None
    return (
        "Your resolution notes state a specific rules/lore fact, but no "
        "search_rules/search_lore/lookup_entity call backs it up this turn. If this "
        "is genuinely grounded, call the right tool now before finalizing. If you "
        "don't actually have a source for it, say so plainly instead of stating it "
        "as verified fact."
    )


_DM_ONLY_TAG = "[DM-ONLY — do not reveal directly]"


def _detect_spoiler_leak(notes: str, messages: list[BaseMessage]) -> str | None:
    """Scans this turn's lookup_entity/search_lore ToolMessages for any
    '[DM-ONLY — do not reveal directly]'-labeled text, and checks whether
    notes states that same fact near-verbatim without going through the
    existing reveal_npc_knowledge/reveal_hidden_element pattern this turn
    (the sanctioned way to actually reveal something to the party). This is
    the deterministic enforcement point for spoiler tiering — the mechanics
    model necessarily SEES DM-only facts (it needs them to run the game
    consistently) but must never STATE them outright in narration."""
    dm_only_facts: list[str] = []
    for output in _lore_tool_outputs_since_last_human(messages):
        for line in output.splitlines():
            if _DM_ONLY_TAG in line:
                fact = line.split(_DM_ONLY_TAG, 1)[1].strip()
                if fact:
                    dm_only_facts.append(fact.lower())
    if not dm_only_facts:
        return None

    notes_lower = notes.lower()
    # Guard against trivially short fragments matching by coincidence.
    leaked = [f for f in dm_only_facts if len(f) > 8 and f in notes_lower]
    if not leaked:
        return None
    return (
        "Your resolution notes appear to state a DM-only fact almost verbatim "
        "(flagged '[DM-ONLY — do not reveal directly]' in a lookup_entity/search_lore "
        "result this turn) without going through reveal_npc_knowledge or "
        "reveal_hidden_element first. Never state a DM-only fact outright in narration "
        "unless the party has genuinely discovered/been told it — rephrase to withhold "
        "it, or call the proper reveal tool first if this is a legitimate reveal."
    )


def get_agent(
    campaign: Campaign,
    store: CampaignStore,
    rules_store: RulesStore,
    history_store: HistoryStore,
    lore_store: LoreStore,
    graph_store: RelationGraphStore,
):
    """Build the in-game DM agent: a mechanics node (tool-calling loop) that
    hands off to a narrator node (prose, no tools) once its tool calls are
    resolved. Only the narrator's output is ever appended to `messages` as
    the turn's DM turn — see mechanics_node below."""
    if _checkpointer is None:
        raise RuntimeError("Agent lifespan not started — call agent_lifespan() first.")

    tools = get_tools(campaign.id, store, rules_store, history_store, campaign.books_in_play, lore_store, graph_store)
    mechanics_modifier = _make_mechanics_modifier(get_mechanics_system_prompt(campaign), campaign.id, store)
    narrator_modifier = _make_narrator_modifier(get_narrator_system_prompt(campaign), campaign.id, store)
    mechanics_model = _get_mechanics_model().bind_tools(tools)
    narrator_model = _get_narrator_model()

    async def mechanics_node(state: DMState) -> Command[Literal["tools", "narrator", "mechanics"]]:
        if (recovery := _recover_orphaned_turn(state["messages"])) is not None:
            return recovery

        correction_count = state.get("correction_count", 0)

        # Bounded retry for a bad/hallucinated tool call (see DMState.tool_error_count) —
        # distinct from correction_count above, which covers a different failure class
        # (the model narrating without calling tools at all, not a tool call it DID make
        # coming back as an error). Without this, a model stuck emitting bad tool calls
        # has no backstop but recursion_limit (60), which surfaces to the player as a
        # generic, misleading "Lost connection to the model backend" error.
        tool_error_count = state.get("tool_error_count", 0)
        tool_error_count = tool_error_count + 1 if _last_tool_batch_had_error(state["messages"]) else 0
        if tool_error_count > _MAX_TOOL_ERROR_RETRIES:
            scratch = _messages_since_last_turn_boundary(state["messages"])
            removals = [RemoveMessage(id=m.id) for m in scratch if m.id]
            return Command(
                goto="narrator",
                update={
                    "messages": removals,
                    "mechanics_notes": (
                        "The DM had trouble resolving that action correctly after a "
                        "couple of tries — tell the player, in character, that you're "
                        "not sure how to handle that and ask them to try rephrasing "
                        "what they're attempting."
                    ),
                    "correction_note": "",
                    "tool_error_count": 0,
                },
            )

        response = await mechanics_model.ainvoke(await mechanics_modifier(state))
        if response.tool_calls:
            return Command(
                goto="tools",
                update={"messages": [response], "correction_note": "", "tool_error_count": tool_error_count},
            )

        notes = _extract_text(response.content)

        # Stale-pending auto-decline: a pending reaction that wasn't created by a
        # resolve_attack/cast_spell call THIS turn must be left over from an
        # earlier turn whose reaction window the player's message didn't address
        # (see PendingAction's docstring) — close it rather than let it survive
        # indefinitely. Re-running this on a guardrail retry (see below) is safe:
        # once cleared, enc.pending_action is None and this is a no-op.
        called_this_turn = _tools_called_since_last_turn_boundary(state["messages"])
        if "resolve_attack" not in called_this_turn and "cast_spell" not in called_this_turn:
            async with campaign_write_lock(campaign.id):
                live_campaign = await store.load(campaign.id)
                enc = live_campaign.active_encounter
                if enc and enc.pending_action:
                    decline_note = await resolve_pending_action_impl(live_campaign)
                    await store.save(live_campaign)
                    notes = (notes + "\n" if notes else "") + f"[Stale pending reaction auto-declined] {decline_note}"

        # ONE live snapshot for everything below — the guardrail chain, the
        # stalled-turn check, and the ground-truth note are all read-only
        # consumers of the same post-tool-execution state, and the
        # stale-pending block above is the last mutator this pass. Each
        # detector used to load its own copy: 3-5 full Campaign loads
        # (Postgres round-trip + pydantic parse) per combatant resolution,
        # multiplied again by stream_response's per-combatant loop.
        live_campaign = await store.load(campaign.id)

        # Guardrail chain: catch the mechanics model stopping mid-resolution
        # during an already-active encounter (rolling/narrating an outcome
        # without the tool call that actually applies it — see
        # _detect_missing_followup); narrating a combat roll (attack/damage/
        # save) with NO tool call at all, active encounter or not — see
        # _detect_missing_combat_roll_followup, which exists precisely because
        # the encounter-gated check above can't catch a "combat" that never
        # got a real encounter/monster created in the first place; claiming a
        # loot/currency gain with no backing tool call at all, combat or not
        # (see _detect_missing_loot_followup — end_encounter itself now counts
        # as a backing call, since it rolls and reveals post-combat loot
        # automatically, see backend/tools/loot_generator.py; there's no more
        # "silently skipped loot after a kill" guardrail needed here — that's
        # now handled deterministically by end_encounter itself, generic DMG
        # treasure and any adventure-specific item alike, rather than by
        # nagging the model to remember to check); or resolving an attack/save
        # against a surviving monster with no active encounter backing it (see
        # _detect_missing_encounter_followup); or declaring combat over in prose
        # with no end_encounter call backing it, leaving active_encounter stuck
        # True forever (see _detect_missing_end_encounter_followup). Capped at
        # one retry per PLAYER TURN via correction_count (reset in
        # stream_response), not per
        # no-tool-calls cycle — a model that stops short after nearly every
        # real tool call could otherwise re-trigger this every loop iteration
        # and never reach the recursion limit's stop condition.
        if live_campaign and correction_count < 1:
            issue = _detect_missing_followup(state, live_campaign)
            if not issue:
                issue = _detect_missing_combat_roll_followup(notes, called_this_turn)
            if not issue:
                issue = _detect_missing_loot_followup(notes, called_this_turn)
            if not issue:
                issue = _detect_missing_monster_death_followup(notes, called_this_turn)
            if not issue:
                issue = _detect_missing_encounter_followup(state, live_campaign)
            if not issue:
                issue = _detect_missing_end_encounter_followup(notes, called_this_turn, live_campaign)
            if issue:
                log.info("guardrail fired: %s", issue)
                return Command(
                    goto="mechanics",
                    update={
                        "correction_note": issue,
                        "correction_count": correction_count + 1,
                        "tool_error_count": 0,
                    },
                )

        # Stopping to prompt the human player when live initiative_order says
        # it's actually a monster's/DM companion's/NPC's turn — nobody is
        # there to act until the model resolves it itself (see
        # _detect_stalled_non_player_turn_followup). Own separate budget
        # (stalled_turn_guardrail_count) — see DMState's doc comment for the
        # starvation this was split out to avoid.
        stalled_turn_guardrail_count = state.get("stalled_turn_guardrail_count", 0)
        if live_campaign and stalled_turn_guardrail_count < 1:
            stalled_issue = _detect_stalled_non_player_turn_followup(live_campaign)
            if stalled_issue:
                log.info("guardrail fired: %s", stalled_issue)
                return Command(
                    goto="mechanics",
                    update={
                        "correction_note": stalled_issue,
                        "stalled_turn_guardrail_count": stalled_turn_guardrail_count + 1,
                        "tool_error_count": 0,
                    },
                )

        # Stage 2 lore guardrails — same shape as the chain above, own
        # separate budget (lore_guardrail_count) for the starvation reason
        # documented on DMState.lore_guardrail_count.
        lore_guardrail_count = state.get("lore_guardrail_count", 0)
        if lore_guardrail_count < 1:
            lore_issue = _detect_uncited_or_invalid_lore_claim(notes, state["messages"])
            if not lore_issue:
                lore_issue = _detect_abstention_violation(notes, called_this_turn)
            if not lore_issue:
                lore_issue = _detect_spoiler_leak(notes, state["messages"])
            if lore_issue:
                log.info("lore guardrail fired: %s", lore_issue)
                return Command(
                    goto="mechanics",
                    update={
                        "correction_note": lore_issue,
                        "lore_guardrail_count": lore_guardrail_count + 1,
                        "tool_error_count": 0,
                    },
                )

        # Session-start directives (build_session_kickoff_message's first-
        # session opening_hook grounding, or a later-session "previously
        # on..." recap) carry hard facts the narrator MUST use verbatim —
        # the mechanics model's own free-text `notes` is a paraphrase, not a
        # reliable relay of them. _narrative_messages deliberately filters
        # this directive out of the narrator's own message history (so
        # players never see the raw internal marker), so its content has no
        # other path to the narrator except through `notes` — meaning a
        # paraphrase that drops the mandatory specifics silently loses them
        # for good. Observed live (2026-07-04): a forceful "MANDATORY
        # OPENING" directive (Out of the Abyss's manacles/Velkynvelve
        # opening_hook) produced a mechanics note that didn't preserve it,
        # and the narrator improvised a generic Underdark scene instead —
        # same failure class as every other "don't trust the model to relay
        # something exactly" bug fixed elsewhere in this app. Fix: when this
        # turn's trigger was a session-start directive, force the ORIGINAL
        # directive text through as the resolution notes, ignoring whatever
        # the mechanics model wrote.
        trigger = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
        )
        if trigger and _extract_text(trigger.content).startswith(_INTERNAL_DIRECTIVE_MARKER):
            notes = _extract_text(trigger.content)

        # Loop-ending response: capture it as the resolution report for the
        # narrator, but never append it to `messages` — it must never look
        # like a DM turn to format_transcript/summarize_session. Also purge
        # this turn's tool-calling scratch-work (every AIMessage-with-
        # tool_calls and ToolMessage since the player's message) from the
        # persisted thread via RemoveMessage — it already did its job driving
        # this turn's resolution and is never displayed (format_transcript/
        # _narrative_messages already filter it out), but left in place it
        # keeps counting toward every future turn's context/KV cache. Combat
        # is what generates the most of these — a thread hit ~27GB resident
        # (up from a ~17GB baseline) after just 4 combat turns before this
        # fix. Only the player's HumanMessage and the narrator's own final
        # AIMessage remain as the permanent record of this turn.
        scratch = _messages_since_last_turn_boundary(state["messages"])

        # Same "don't trust the model to relay something exactly" principle
        # as the session-start directive handling below, applied to this
        # turn's dice — see _verified_rolls_note for the full story.
        notes += _verified_rolls_note(scratch)
        # Same principle, generalized beyond dice to every other mutating
        # tool call this turn — see _verified_state_changes_note.
        notes += _verified_state_changes_note(scratch)

        removals = [RemoveMessage(id=m.id) for m in scratch if m.id]

        # Deterministic "whose turn is it" fact, appended unconditionally —
        # see _next_turn_ground_truth_note's docstring for why this is a
        # necessary backstop even with _detect_stalled_non_player_turn_followup
        # already in the guardrail chain above.
        ground_truth = _next_turn_ground_truth_note(live_campaign) if live_campaign else None
        if ground_truth:
            notes += ground_truth

        return Command(
            goto="narrator",
            update={
                "messages": removals,
                "mechanics_notes": notes,
                "correction_note": "",
                "tool_error_count": 0,
            },
        )

    async def narrator_node(state: DMState) -> dict:
        response = await narrator_model.ainvoke(await narrator_modifier(state))
        return {"messages": [response], "mechanics_notes": ""}

    graph = StateGraph(DMState)
    graph.add_node("mechanics", mechanics_node, destinations=("tools", "narrator", "mechanics"))
    graph.add_node("tools", _make_tool_node(tools, campaign.id))
    graph.add_node("narrator", narrator_node)
    graph.add_edge(START, "mechanics")
    graph.add_edge("tools", "mechanics")
    graph.add_edge("narrator", END)

    return graph.compile(checkpointer=_checkpointer)


# ── session zero agent ────────────────────────────────────────────────────────
# Two-node mechanics/narrator split, mirroring get_agent()'s in-game pattern —
# added 2026-07-04 after a live Session 0 conversation caught the single
# combined create_react_agent (tool-calling + player-facing prose in one
# generation pass) narrating fake tool calls: visibly as literal `<call:...>`
# text leaking into a reply, and silently as a claimed-but-never-made
# update_character_draft (a confirmed character name that never reached the
# draft). The in-game agent already solved the analogous tension by never
# letting the tool-calling node's output reach the player directly; Session 0
# had no equivalent separation. See _detect_fake_tool_call for the other half
# of the fix — a structural guardrail is what actually catches this, not just
# a differently-shaped prompt.

_FAKE_TOOL_CALL_RE = re.compile(
    r"<call:|```json|\bI(?:'m| am) calling\b|\bI(?:'ll| will) call\b|"
    r"\bcalling `?\w+`?\s+behind the scenes\b",
    re.IGNORECASE,
)


def _detect_fake_tool_call(notes: str) -> str | None:
    """Catches the mechanics model narrating a tool call as text instead of
    actually invoking it — observed live: `<call:list_options(...)>` leaked
    verbatim into a reply. A terse internal resolution note has no legitimate
    reason to contain call-like pseudo-syntax or a code fence, so this is a
    tight, low-false-positive signal rather than an attempt to parse general
    prose for claims. Returns a correction note for the retry, or None."""
    if not _FAKE_TOOL_CALL_RE.search(notes):
        return None
    return (
        "Your last response wrote out what looks like a tool call as text "
        "instead of actually invoking one (e.g. `<call:...>` syntax, a code "
        "fence, or narrating 'I am calling X'). Make the real tool call now "
        "through the actual tool-calling mechanism — don't describe it, don't "
        "narrate it, just call it. If you don't actually need a tool for what "
        "you were about to say, drop the fake-call text entirely and just "
        "report normally."
    )


_BOLD_TERM_RE = re.compile(r"\*{2,3}([A-Za-z][A-Za-z' \-]{2,30}?)\*{2,3}")
_CANTRIP_OFFER_RE = re.compile(
    r"\bcantrips?\b[^.\n]{0,40}\b(?:pick|choose|select)\b"
    r"|\b(?:pick|choose|select)\b[^.\n]{0,40}\bcantrips?\b",
    re.IGNORECASE,
)
# Suppresses _CANTRIP_OFFER_RE for a correct "you don't get any cantrips"
# explanation — observed live to otherwise false-positive on phrasing like
# "you don't get cantrips, but you do get two level-1 spells to pick", where
# "cantrip" and "pick" land within the offer window despite this being the
# CORRECT thing to say, not an invented offer.
_NO_CANTRIPS_RE = re.compile(
    r"\b(?:don't|doesn't|do not|does not|no|none|zero|never|isn't|aren't)\b"
    r"[^.\n]{0,20}\bcantrips?\b",
    re.IGNORECASE,
)
_ALL_SPELLS_LOWER = {name.lower() for name in ALL_SPELLS}


def _detect_invented_spells(text: str, char_class: str) -> str | None:
    """Catches the mechanics/narrator handoff inventing spell names beyond
    the real curated menu (backend/data/spells.py's SPELL_MENUS) — observed
    live: a Human Ranger offered a "Choose your Cantrips" step with Druid
    cantrips (Rangers have zero) plus level-1 spells not on Ranger's real
    4-spell menu. Unlike _detect_fake_tool_call this needs no leaked
    call-syntax to fire — it cross-references named spells directly against
    ALL_SPELLS/SPELL_MENUS, since the narrator has no programmatic access to
    that data and can invent plausible-sounding but wrong names even in a
    perfectly formatted reply.

    Skips entirely for non-casters (no SPELL_MENUS entry at all) — without
    this a class like Fighter mentioning a spell name in ordinary backstory
    prose ("his mentor once showed him a trick with Prestidigitation") would
    false-positive, since there's no real menu to have invented content
    beyond in the first place."""
    menu = SPELL_MENUS.get(char_class)
    if not menu:
        return None

    # Pre-gate: only worth the fine-grained check if this text actually looks
    # like a spell/cantrip offer — avoids firing on a single incidental
    # bolded common word (Shield, Command, Guidance are all real spell names
    # but also ordinary English) in unrelated prose.
    bold_terms = _BOLD_TERM_RE.findall(text)
    looks_like_spell_offer = (
        re.search(r"\b(?:spell|cantrip)s?\b", text, re.IGNORECASE) and len(bold_terms) >= 2
    )

    real_names = {n.lower() for tier in menu.values() for n in tier}
    bad = []
    if looks_like_spell_offer:
        for term in bold_terms:
            low = term.strip().lower()
            if low in _ALL_SPELLS_LOWER and low not in real_names:
                bad.append(term.strip())

    no_cantrips_for_class = 0 not in menu
    cantrip_offered = bool(_CANTRIP_OFFER_RE.search(text)) and not _NO_CANTRIPS_RE.search(text)

    if not bad and not (cantrip_offered and no_cantrips_for_class):
        return None

    parts = []
    if bad:
        parts.append(f"named spell(s) not on {char_class}'s real menu: {', '.join(bad)}")
    if cantrip_offered and no_cantrips_for_class:
        parts.append(f"{char_class} has no cantrips at all, but cantrips were offered")

    return (
        f"Your last response has a problem: {'; '.join(parts)}. Call "
        f"list_options('spells {char_class}') now and use ONLY the exact "
        f"names it returns — quote the real list verbatim, don't paraphrase "
        f"or add anything from memory."
    )


def get_session_zero_agent(
    campaign: Campaign,
    player_slug: str,
    store: CampaignStore,
    rules_store: RulesStore,
    ds: DraftStore,
):
    """Build a two-node character-creation agent for one player's Session 0:
    a mechanics node (tool-calling loop, low temperature) hands off to a
    narrator node (prose only, no tools, higher temperature) once its tool
    calls are resolved for the turn — same split as get_agent(), same reason.
    Only the narrator's output is ever appended to `messages` as the turn's
    reply."""
    if _checkpointer is None:
        raise RuntimeError("Agent lifespan not started — call agent_lifespan() first.")

    tools = [
        *dice.make_tools(),
        *rules.make_tools(rules_store, campaign.books_in_play),
        *chargen.make_tools(campaign.id, player_slug, store, ds),
        *companion.make_tools(campaign.id, store),
    ]
    mechanics_modifier = _make_state_modifier(get_session_zero_mechanics_prompt(campaign))
    narrator_modifier = _make_narrator_modifier(get_session_zero_narrator_prompt(campaign), campaign.id, store)
    mechanics_model = _get_mechanics_model().bind_tools(tools)
    narrator_model = _get_narrator_model()

    async def chargen_mechanics_node(state: DMState) -> Command[Literal["tools", "narrator", "mechanics"]]:
        if (recovery := _recover_orphaned_turn(state["messages"])) is not None:
            return recovery

        correction_count = state.get("correction_count", 0)

        # See get_agent()'s mechanics_node for the full rationale — same bounded
        # retry for a bad/hallucinated tool call, distinct from correction_count.
        tool_error_count = state.get("tool_error_count", 0)
        tool_error_count = tool_error_count + 1 if _last_tool_batch_had_error(state["messages"]) else 0
        if tool_error_count > _MAX_TOOL_ERROR_RETRIES:
            scratch = _messages_since_last_turn_boundary(state["messages"])
            removals = [RemoveMessage(id=m.id) for m in scratch if m.id]
            return Command(
                goto="narrator",
                update={
                    "messages": removals,
                    "mechanics_notes": (
                        "Had trouble processing that step correctly after a couple of "
                        "tries — tell the player you're not sure how to handle that and "
                        "ask them to try rephrasing what they're choosing/describing."
                    ),
                    "correction_note": "",
                    "tool_error_count": 0,
                },
            )

        response = await mechanics_model.ainvoke(mechanics_modifier(state))
        if response.tool_calls:
            return Command(
                goto="tools",
                update={"messages": [response], "correction_note": "", "tool_error_count": tool_error_count},
            )

        notes = _extract_text(response.content)

        # Fix C: append any list_options tool output verbatim, regardless of
        # whether the model's own note quoted it — this makes the real data
        # structurally present for the narrator rather than depending on the
        # model having followed the "quote it verbatim" prompt instruction.
        scratch = _messages_since_last_turn_boundary(state["messages"])
        tool_outputs = [
            _extract_text(m.content) for m in scratch
            if isinstance(m, ToolMessage) and getattr(m, "name", None) == "list_options"
        ]
        if tool_outputs:
            notes += (
                "\n\n[VERBATIM TOOL OUTPUT — the only real menu; do not add to it]\n"
                + "\n\n".join(tool_outputs)
            )

        # Same reasoning as get_agent()'s mechanics_node — see
        # _verified_rolls_note; the wider tool set additionally captures
        # roll_ability_scores (see _CHARGEN_VERIFIED_ROLL_TOOLS).
        notes += _verified_rolls_note(scratch, _CHARGEN_VERIFIED_ROLL_TOOLS)

        if correction_count < 1:
            issue = _detect_fake_tool_call(notes)
            if not issue:
                draft = ds.get(campaign.id, player_slug)
                issue = _detect_invented_spells(notes, draft.get("char_class", ""))
            if issue:
                return Command(
                    goto="mechanics",
                    update={
                        "correction_note": issue,
                        "correction_count": correction_count + 1,
                        "tool_error_count": 0,
                    },
                )

        # Same scratch-purge reasoning as the in-game mechanics_node: this
        # turn's tool-call AIMessages/ToolMessages already did their job and
        # are never displayed (narrator gets `notes`, not raw messages), left
        # in place they'd only keep counting toward every future turn's
        # context/KV cache. See get_agent()'s mechanics_node for the original
        # ~27GB-resident incident this pattern fixed.
        removals = [RemoveMessage(id=m.id) for m in scratch if m.id]
        return Command(
            goto="narrator",
            update={
                "messages": removals,
                "mechanics_notes": notes,
                "correction_note": "",
                "tool_error_count": 0,
            },
        )

    async def chargen_narrator_node(state: DMState) -> Command[Literal["mechanics", "__end__"]]:
        # Fix D: this node validates its OWN output before it ever reaches the
        # player, using a separate retry budget from the mechanics-side one —
        # see DMState's narrator_correction_count comment for why sharing the
        # counter risks starvation. main.py's session_zero_stream buffers
        # narrator tokens per-invocation and only forwards the invocation that
        # actually reaches END, so a retry here never leaks a discarded first
        # draft to the player — the tradeoff is that Session 0 replies arrive
        # as one block instead of typing out live.
        narrator_correction_count = state.get("narrator_correction_count", 0)
        response = await narrator_model.ainvoke(await narrator_modifier(state))
        text = _extract_text(response.content)

        if narrator_correction_count < 1:
            draft = ds.get(campaign.id, player_slug)
            issue = _detect_invented_spells(text, draft.get("char_class", ""))
            if issue:
                return Command(
                    goto="mechanics",
                    update={
                        "correction_note": f"[Your last narration had a problem] {issue}",
                        "narrator_correction_count": narrator_correction_count + 1,
                    },
                )

        return Command(
            goto=END,
            update={"messages": [response], "mechanics_notes": "", "narrator_correction_count": 0},
        )

    graph = StateGraph(DMState)
    graph.add_node("mechanics", chargen_mechanics_node, destinations=("tools", "narrator", "mechanics"))
    graph.add_node("tools", _make_tool_node(tools, campaign.id))
    graph.add_node("narrator", chargen_narrator_node, destinations=("mechanics", END))
    graph.add_edge(START, "mechanics")
    graph.add_edge("tools", "mechanics")

    return graph.compile(checkpointer=_checkpointer)


# ── world-prep agent ───────────────────────────────────────────────────────────

def get_world_prep_agent(
    campaign: Campaign,
    store: CampaignStore,
    rules_store: RulesStore,
    books_in_play: list[str],
    lore_store: LoreStore,
):
    """One-shot, non-interactive agent for automatic world-prep. No
    checkpointer — this isn't a resumable conversation, just a single
    bounded tool-calling run whose prompt is passed directly as a HumanMessage."""
    tools = get_world_prep_tools(campaign.id, store, rules_store, books_in_play, lore_store)

    return create_react_agent(
        model=_get_model(),
        tools=_make_tool_node(tools, campaign.id),
    )


def get_npc_prep_agent(
    campaign: Campaign,
    store: CampaignStore,
    rules_store: RulesStore,
    books_in_play: list[str],
    lore_store: LoreStore,
):
    """One-shot, non-interactive agent for the opening-scene NPC/site-detail
    seeding pass — same shape as get_world_prep_agent, restricted instead to
    get_npc_prep_tools (create_npc + set_opening_location_detail + rules
    search, no party/combat/quest/movement/travel/NPC-runtime tools)."""
    tools = get_npc_prep_tools(campaign.id, store, rules_store, books_in_play, lore_store)

    return create_react_agent(
        model=_get_model(),
        tools=_make_tool_node(tools, campaign.id),
    )


# ── party-fill agent ───────────────────────────────────────────────────────────

def _fill_party_prompt(campaign: Campaign, requested_class: str | None = None) -> str:
    """Build the one-shot party-fill instruction.

    The "don't repeat an overrepresented class" constraint is computed in
    Python and stated as a direct fact — not left for the model to infer
    from get_campaign_summary's prose. Verified necessary: qwen2.5:14b,
    given only a general "complement, don't duplicate" instruction (plus an
    illustrative example mentioning "healer"), anchored on generating
    Cleric regardless of actual party composition — including once adding a
    4th Cleric to an already all-Cleric party, while its own summary said
    the party "lacks variety beyond Clerics." Stating the overrepresented
    classes explicitly removes the inference step that was failing.

    If the DM picked an explicit class from the UI (requested_class), skip
    that class-selection reasoning entirely — the class is a direct DM
    choice, not the model's to reconsider.

    No recommended-party-size gate: the adventure's recommended size is just
    a suggestion surfaced elsewhere in the UI, not a cap — every click of
    "Ask DM to add a member" adds exactly one companion, however large the
    party already is. (2026-07-04: previously told the model to skip
    creation once the party met the recommended size; removed per user
    request — the DM should be able to press the button as many times as
    they want.)
    """
    if campaign.party:
        roster = ", ".join(f"{c.race} {c.char_class}" for c in campaign.party)
        party_line = f"Current party ({len(campaign.party)}): {roster}."
    else:
        party_line = "Current party: empty."

    if requested_class:
        return f"""{party_line}

The DM has specifically requested a new companion of class: {requested_class}. \
Generate exactly ONE DM-controlled companion of that class by calling \
generate_companion_character — choose a race, background, and ability scores that \
make sense for a {requested_class} and fit well with the existing party above. Add \
them regardless of the adventure's recommended party size — the DM has already \
decided to add this member; do not skip the call. \
Reply with a short (2-3 sentence) plain-text summary of what you did and why."""

    counts: dict[str, int] = {}
    for c in campaign.party:
        counts[c.char_class] = counts.get(c.char_class, 0) + 1
    overrepresented = [cls for cls, n in counts.items() if n >= 2]
    avoid_line = (
        f"\nAlready overrepresented — do not add another one of these unless "
        f"every other reasonable class is also already covered: {', '.join(overrepresented)}."
        if overrepresented else ""
    )

    return f"""{party_line}{avoid_line}

Generate exactly ONE DM-controlled companion via generate_companion_character — pick \
a class, race, and background that fills an actual gap in the roster above, not a \
class that's already represented. Always add one — the recommended party size shown \
elsewhere in the UI is just a suggestion, not a cap; do not skip the call just \
because the party is already at or above that size. \
Reply with a short (2-3 sentence) plain-text summary of what you did and why."""


def get_party_fill_agent(campaign: Campaign, store: CampaignStore):
    """One-shot, non-interactive agent for DM-triggered party filling. No
    checkpointer, same pattern as get_world_prep_agent — a single bounded
    tool-calling run, not a resumable conversation."""
    tools = [
        *campaign_tools.make_tools(campaign.id, store),
        *companion.make_tools(campaign.id, store),
    ]
    return create_react_agent(
        model=_get_model(),
        tools=_make_tool_node(tools, campaign.id),
    )


async def run_fill_party(campaign: Campaign, store: CampaignStore, requested_class: str | None = None) -> str:
    """Run the party-fill agent once and return its final summary text.

    requested_class: an explicit class the DM picked from the UI dropdown,
    bypassing the model's own class judgment. None (the default "Random —
    DM decides" option) leaves the choice to the model as before.
    """
    agent = get_party_fill_agent(campaign, store)
    before_count = len(campaign.party)
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=_fill_party_prompt(campaign, requested_class))]},
        config={"recursion_limit": 20},
    )
    text = _extract_text(response["messages"][-1].content)

    # Observed: qwen2.5:14b sometimes appends a fenced ```json block after its
    # real prose summary — looks like a second, never-actually-executed tool
    # call it's narrating rather than invoking (the real call already ran;
    # this is just decorative leftover text). Strip it so the DM-facing
    # summary shown in the UI doesn't show raw tool-call JSON.
    fence = text.find("```")
    if fence != -1:
        text = text[:fence].rstrip()

    # Ground truth check (found 2026-07-04, live): this is a one-shot
    # create_react_agent with no mechanics/narrator split (see _get_model's
    # docstring — the same combined-role failure class that forced Session
    # 0's structural split can still happen here), so the model's final text
    # is not trustworthy evidence a companion was actually created — observed
    # live: a confident "I've added a new companion..." summary with no real
    # generate_companion_character call underneath, DraftStore-empty-style.
    # Re-read the campaign from the store instead of believing the narration.
    # No legitimate no-op case anymore (2026-07-04: the recommended-size gate
    # was removed from the prompt, every click is expected to add exactly one
    # companion), so a flat count-didn't-grow check is now sufficient.
    after = await store.load(campaign.id)
    if after and len(after.party) > before_count:
        return text
    return (
        "No companion was added — the model reported making a change but "
        "didn't actually call the tool to do it. Nothing was saved; try again."
    )


# ── stall detection ─────────────────────────────────────────────────────────
# The Ollama MLX runner can wedge indefinitely mid-request (observed live —
# see the incident where `ps`/`kill` on the host was the only way to recover
# a hung gemma4:26b-mlx runner; keep_alive=-1 residency pinning was tried and
# didn't fully prevent it, so this doesn't try to prevent the wedge, only
# surface it). Without this, a wedged turn looks identical to a slow one from
# the player's side: the SSE stream just sits silent forever with no error
# and no way to tell "still thinking" from "will never respond."

STALL_TIMEOUT = 120.0  # seconds with no new event before we tell the player

STALL_MESSAGE = (
    "The model backend hasn't responded in over two minutes — it may be "
    "wedged (a known Ollama/MLX issue, not a problem with your message). "
    "Still waiting in case it's just slow. If it doesn't recover, open a "
    "terminal on the host and run:\n\n"
    "    pkill -9 -f \"ollama runner\"\n\n"
    "then send your message again."
)


class _Stall:
    """Sentinel yielded by watch_for_stalls — no item arrived within
    STALL_TIMEOUT. Purely observational: the wrapped source keeps running
    untouched, this only reports that nothing has come through yet."""


STALL = _Stall()
_DONE = object()


class _TurnBoundary:
    """Sentinel yielded by stream_response between two separate combatant
    turns in the same player-message reply (see stream_response's own
    per-combatant astream_events loop). Tells the SSE layer (backend/main.py)
    to close out the current chat bubble and start a fresh one, same pattern
    as STALL."""


TURN_BOUNDARY = _TurnBoundary()


async def watch_for_stalls(source: AsyncIterator, stall_after: float = STALL_TIMEOUT) -> AsyncIterator:
    """Wrap an async iterator, yielding STALL if `stall_after` seconds pass
    with no new item — and again every `stall_after` seconds after that for
    as long as the stall continues, not just once. Draining happens in a
    background task so a stall report never cancels or otherwise disturbs
    the underlying call — by the time we know enough to call it "stuck"
    rather than "slow," canceling it ourselves would just trade one guess
    for another. Re-raises any exception from the source (e.g. a connection
    error once the wedged runner is killed) once draining reaches it.

    Confirmed live, 2026-07-12: a cold local-model load took ~15 minutes to
    produce its first token. The stall-suppression here used to only fire
    STALL once (an `already_stalled` guard skipped every timeout after the
    first), so the SSE connection went completely silent for the remaining
    ~13 minutes with no ping/stall/token of any kind — exactly the window
    a browser/OS/network idle-connection timeout will kill, which is what
    happened, surfacing to the player as a bare "Connection lost" with no
    indication anything had even been attempted. Repeating the STALL event
    on every timeout keeps the connection alive with real traffic for as
    long as a slow-but-not-actually-wedged call keeps running."""
    queue: asyncio.Queue = asyncio.Queue()

    async def _drain():
        try:
            async for item in source:
                await queue.put(item)
        except Exception as e:
            await queue.put(e)
        finally:
            await queue.put(_DONE)

    task = asyncio.create_task(_drain())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=stall_after)
            except asyncio.TimeoutError:
                yield STALL
                continue
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        if not task.done():
            task.cancel()


# ── streaming helper ───────────────────────────────────────────────────────────

_REASONING_LEAK_RE = re.compile(
    r"<\|?channel\|?>.*?<\|?channel\|?>|<think>.*?</think>",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning_leakage(text: str) -> str:
    """Best-effort removal of stray reasoning/channel-tag artifacts from
    narrator-facing text. Originally defense-in-depth alongside
    reasoning=False on every ChatOllama instance, back when chat ran on
    Ollama/gemma4:26b-mlx — that kwarg has no vLLM/ChatOpenAI equivalent
    (see backend/llm.py's vllm_chat()), and Step 0 testing (vllm-migration-
    plan.md) didn't reproduce the reasoning-channel leak this originally
    guarded against, but this stays as a cheap safety net in case a leak
    reaches `.content` anyway on the new backend. Not chunk-boundary-safe
    — a tag split across two stream chunks won't be caught here; not worth
    buffering the live game's token-by-token stream to fix that."""
    return _REASONING_LEAK_RE.sub("", text)


# Bounds stream_response's own per-combatant resumption loop below — real
# multi-round fights shouldn't hit this, but it's a hard stop against a
# combatant that genuinely never resolves (mechanics exhausts its own
# guardrail retries every time without ever calling advance_initiative)
# looping forever in real time. Each iteration gets a fresh recursion_limit,
# so this isn't bounded by that the way a single shared invocation would be.
_MAX_COMBATANT_TURNS = 10


async def stream_response(
    campaign: Campaign,
    store: CampaignStore,
    rules_store: RulesStore,
    history_store: HistoryStore,
    user_message: str,
    thread_id: str,
    lore_store: LoreStore,
    graph_store: RelationGraphStore,
) -> AsyncIterator[str]:
    """Yield plain-text tokens from the narrator node as they stream, plus a
    TURN_BOUNDARY sentinel (interleaved the same way watch_for_stalls
    interleaves STALL) between two separate combatant turns.

    One player message can auto-resolve several non-player combatants'
    turns in a row before control returns to a real player (see the Combat
    prompt section's auto-continuation rule) — each such turn is its own
    separate `astream_events` call on the same thread_id here, not one call
    covering all of them. Two reasons this is a loop of separate calls
    rather than one call with the graph looping internally (which is how
    this worked until 2026-07-11):

    1. recursion_limit is fixed per astream_events call, not resettable
       from inside the graph. One call covering a whole multi-combatant
       response meant every combatant shared the same 60-step budget — a
       single hard-to-resolve combatant (needing several guardrail-retry
       round trips) could exhaust most of it, leaving too little for later,
       perfectly resolvable combatants and ending the whole response in a
       hard GraphRecursionError instead of any narration at all. Each
       iteration below gets its own fresh 60.
    2. Each combatant's guardrail-retry budget (correction_count,
       tool_error_count, lore_guardrail_count, stalled_turn_guardrail_count
       — see DMState) needs to start fresh too, for the same
       starvation reason documented on stalled_turn_guardrail_count's field
       comment. Passing them as 0 in each iteration's input achieves that
       directly — LangGraph resumes every other channel (messages, live
       campaign state) from the checkpoint as normal.

    Only the FIRST iteration carries the player's actual message; later
    iterations resume the same thread with no new human message, relying
    entirely on the checkpointer's persisted `messages` plus live campaign
    state (mechanics_node's own ground-truth checks) to know whose turn it
    genuinely is. The mechanics node's tool-calling loop runs first each
    iteration and is never streamed — filtering by langgraph_node keeps its
    tool-call JSON and reasoning off the wire entirely."""
    agent = get_agent(campaign, store, rules_store, history_store, lore_store, graph_store)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    turn_input: dict = {
        "messages": [HumanMessage(content=user_message)],
        **_FRESH_GUARDRAIL_BUDGETS,
    }

    for _ in range(_MAX_COMBATANT_TURNS):
        async for event in agent.astream_events(turn_input, config=config, version="v2"):
            if event["event"] != "on_chat_model_stream":
                continue
            if event.get("metadata", {}).get("langgraph_node") != "narrator":
                continue
            chunk = event["data"].get("chunk")
            if chunk and chunk.content:
                yield strip_reasoning_leakage(chunk.content)

        live_campaign = await store.load(campaign.id)
        hit = _live_current_turn(live_campaign) if live_campaign else None
        if hit is None:
            return
        _, is_player = hit
        if is_player:
            return

        # Another non-player combatant's turn is owed before a real player
        # can act — resume the same thread for it, no new human message,
        # every guardrail budget fresh.
        yield TURN_BOUNDARY
        turn_input = dict(_FRESH_GUARDRAIL_BUDGETS)


# ── transcript retrieval ───────────────────────────────────────────────────────

async def get_thread_messages(thread_id: str) -> list[BaseMessage]:
    """Load all messages stored in the LangGraph checkpoint for this thread."""
    if _checkpointer is None:
        return []
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = await _checkpointer.aget_tuple(config)
    if not checkpoint_tuple:
        return []
    return checkpoint_tuple.checkpoint["channel_values"].get("messages", [])


_NEAR_LIMIT_MARGIN = 5  # warn this many messages before _MAX_MESSAGES trimming kicks in


async def get_context_status(thread_id: str) -> dict:
    """Raw message count for a thread vs the mechanics trim window
    (_MAX_MESSAGES). Once trimming kicks in, older context silently drops —
    surfaced to the player as a UI warning (see /campaigns/{id}/thread-info)
    rather than left for the model to notice and mention organically, which
    isn't reliable (see the fill-party Cleric-anchoring bug for a concrete
    example of that failure mode elsewhere in this app)."""
    count = len(await get_thread_messages(thread_id))
    return {
        "message_count": count,
        "max_messages": _MAX_MESSAGES,
        "near_limit": count >= _MAX_MESSAGES - _NEAR_LIMIT_MARGIN,
    }


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    return str(content)


def format_transcript(messages: list[BaseMessage]) -> list[dict]:
    """Return [{role, content}] keeping only human turns and DM narrative.

    Tool calls and tool responses are filtered out — they're implementation
    detail, not the story. AIMessages that only contain tool_calls (no text
    content) are also dropped. The mechanics model's resolution report never
    reaches this list in the first place (see mechanics_node in get_agent) —
    only the narrator's prose is ever a "dm" turn.
    """
    return [
        {"role": "player" if isinstance(m, HumanMessage) else "dm", "content": _extract_text(m.content).strip()}
        for m in _narrative_messages(messages)
    ]


# ── session summarisation ──────────────────────────────────────────────────────

_SUMMARY_PROMPT = """\
You are archiving a D&D campaign called "{campaign_name}".

Below is a session transcript. Write a detailed chronicle (5-10 paragraphs) covering:
- Where the party was and what they were trying to accomplish
- Key events, battles, and challenges faced
- Important NPCs encountered and what was learned from/about them
- Major decisions the party made and their consequences
- Plot revelations, mysteries uncovered, or hooks established for the future
- How the session ended

Focus on narrative significance. A random combat deserves one sentence; meeting
a key NPC and learning a secret deserves a full paragraph.

Actual current party state (ground truth — the transcript's own prose can
narrate things that were never actually applied to a character; do NOT claim
any item, weapon, or currency gain in your chronicle or key events unless
it's actually reflected here):
{party_state}

Relevant adventure text retrieved for context (use this ONLY to help judge
which transcript events are most significant to the module's actual
structure — do not invent plot content or connections beyond what the
transcript itself describes):
{book_context}

After the chronicle, list 5-10 KEY EVENTS as short bullet points capturing the
most plot-relevant moments.

Then, in one or two sentences, describe what part of the adventure module
this session's events leave the party at — a chapter/section name if you can
identify one, otherwise a plain description of the current story beat (where
they are, what they're about to do next). This will be used to re-ground the
next session in the right part of the book.

Finally, list any NEW relationships between named entities (NPCs, locations,
factions, items) that this session's events established or revealed — e.g. an
NPC joining a faction, an NPC being placed at a new location, an item
changing hands, two NPCs turning out to be allied or related. Only include a
relationship the transcript actually states or clearly implies — do not
invent connections. One line per relationship, in the form:
NEW_RELATION: <source entity name> | <short relation phrase> | <target entity name>
If there are none, output nothing after the marker.

Format your response EXACTLY as:
{chronicle_marker}
<narrative here>
{events_marker}
- event 1
- event 2
{progress_marker}
<one or two sentences on the party's current point in the adventure>
{relations_marker}
NEW_RELATION: <source> | <relation> | <target>

Transcript:
{transcript}"""


def _party_ground_truth(campaign: Campaign) -> str:
    """Real persisted party state — HP, inventory, currency — handed to the
    summarizer as ground truth. summarize_session has no tool access of its
    own (unlike the mechanics node), so the transcript's own prose is
    otherwise its only source of truth — exactly what let a fabricated
    "stole their weapons" claim slip into a chronicle with no backing loot
    tool call anywhere in the actual game state."""
    if not campaign.party:
        return "(no party members)"
    lines = []
    for c in campaign.party:
        inv = ", ".join(
            f"{i.name} x{i.quantity}" if i.quantity > 1 else i.name for i in c.inventory
        ) or "(none)"
        coins = ", ".join(
            f"{v} {k}" for k, v in (
                ("pp", c.currency.pp), ("gp", c.currency.gp), ("ep", c.currency.ep),
                ("sp", c.currency.sp), ("cp", c.currency.cp),
            ) if v
        ) or "none"
        lines.append(f"- {c.name}: {c.current_hp}/{c.max_hp} HP. Inventory: {inv}. Currency: {coins}.")
    return "\n".join(lines)


async def _book_context_for_summary(campaign: Campaign, rules_store: RulesStore) -> str:
    """Retrieve adventure text relevant to where the party currently stands,
    for the summarizer to use in judging which transcript events are actually
    plot-salient — without this, "salience" is purely the summarizer's own
    guess from prose, with no connection to what the module itself considers
    important. Query preference: the prior session's own adventure_progress
    note (closes the loop with build_session_kickoff_message, which uses this
    same field to re-ground the NEXT session) — falling back to the party's
    current location for a first session, since there's no prior progress
    note yet. Returns a graceful placeholder, never raises, if nothing usable
    is available (no query, index not built, no books in play)."""
    query = None
    if campaign.sessions and campaign.sessions[-1].adventure_progress:
        query = campaign.sessions[-1].adventure_progress
    elif campaign.current_location_id:
        loc = next((l for l in campaign.locations if l.id == campaign.current_location_id), None)
        if loc:
            query = f"{loc.name}. {loc.description}".strip()

    if not query or not await rules_store.is_ready() or not campaign.books_in_play:
        return "(none retrieved)"

    # search_adventure_only (plain dense search) rather than the hybrid
    # search() — see design.md's Evolution section, 2026-07-09: search()'s
    # reranker made its own separate ChatOllama call, so this one retrieval
    # step used to force an embed -> chat model swap on Ollama every time a
    # session ends (that was the exact trigger already root-caused for
    # world-prep's recurring whole-app freeze). The embed<->chat swap risk
    # itself is now moot (2026-07-13, vllm-migration-plan.md) — Ollama only
    # ever serves the embedder now, chat runs on a separate vLLM-metal
    # server, so there's no single-process model-swap to trigger regardless
    # of which retrieval path this uses. Kept dense-only anyway: it's also
    # just a better fit than search() was in the first place, independent of
    # the swap concern — matches search_adventure_only's own original
    # rationale: adventure-scoped dense search surfaces a location's own
    # named content instead of generic core-rulebook DM advice drowning it out.
    chunks = []
    for book in campaign.books_in_play:
        chunks += await rules_store.search_adventure_only(query, adventure=book, k=4)
    if not chunks:
        return "(none retrieved)"
    return "\n\n---\n\n".join(f"[{c.book} — {c.section}]\n{c.content}" for c in chunks)


_NEW_RELATION_RE = re.compile(
    r'^\s*NEW_RELATION:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$', re.MULTILINE,
)


async def summarize_session(
    thread_id: str,
    campaign: Campaign,
    rules_store: RulesStore,
) -> tuple[str, list[str], str, list[tuple[str, str, str]]]:
    """Generate a narrative chronicle, key-events list, adventure-progress
    note, and any newly-stated entity relationships for a session.

    Returns (summary_text, key_events_list, adventure_progress,
    relations — a list of (source_name, relation, target_name) tuples for
    Stage 1.5's incremental relation graph). All empty if the thread has no
    dialogue. This piggybacks on the one LLM call this function already
    makes — zero new round-trips, and only the just-ended session's
    transcript is scanned, the genuinely incremental property Stage 1.5
    needs (see backend/stores/graph_store.py).
    """
    messages = await get_thread_messages(thread_id)
    turns = format_transcript(messages)

    if not turns:
        return "Session contained no dialogue to summarize.", [], "", []

    transcript = "\n\n".join(
        f"{'PLAYER' if t['role'] == 'player' else 'DM'}: {t['content']}"
        for t in turns
    )

    # Random per-call markers prevent player-injected text from confusing the
    # parser — a player can't predict or inject "---CHRONICLE-<hex>---".
    token = secrets.token_hex(8)
    chronicle_marker = f"---CHRONICLE-{token}---"
    events_marker = f"---EVENTS-{token}---"
    progress_marker = f"---PROGRESS-{token}---"
    relations_marker = f"---RELATIONS-{token}---"

    # asyncio.to_thread — _book_context_for_summary is synchronous
    # (RulesStore.search_adventure_only, a blocking Ollama embed call under
    # the hood); called bare here it would freeze this process's single
    # event loop for every request, not just this session-end call — same
    # bug class as the 2026-06-30 audit's add_session finding and
    # world_prep.py's call sites; see design.md's Evolution section.
    #
    # asyncio.wait_for on top of that as a backstop. Note
    # _book_context_for_summary itself no longer calls the hybrid search()
    # (2026-07-09 — see its own docstring): it now uses
    # search_adventure_only, which only needs the embedding model, not a
    # second embed->chat model swap via search()'s own LLM reranker. That
    # was the actual root cause of a whole-app freeze here, not just a slow
    # call — same mechanism as world-prep's, see design.md.
    try:
        book_context = await asyncio.wait_for(
            _book_context_for_summary(campaign, rules_store), timeout=30.0
        )
    except asyncio.TimeoutError:
        book_context = "(book context unavailable — retrieval timed out)"

    prompt = _SUMMARY_PROMPT.format(
        campaign_name=campaign.name,
        transcript=transcript,
        chronicle_marker=chronicle_marker,
        events_marker=events_marker,
        progress_marker=progress_marker,
        relations_marker=relations_marker,
        party_state=_party_ground_truth(campaign),
        book_context=book_context,
    )

    llm = _get_model()
    # Bounded retry on an Ollama timeout — the book-context step above still
    # makes one embedding call, so a single embed->chat swap remains
    # possible right here even after the fix above. Same discipline as
    # world-prep's _ainvoke_with_retry: one retry, not a loop.
    #
    # asyncio.wait_for wrapping each attempt, NOT relying on the httpx
    # client_kwargs timeout alone — confirmed live, 2026-07-09: this exact
    # call still froze the whole app for 4+ minutes (well past the
    # configured 120s client timeout) on a retry immediately after this fix
    # first shipped. Leading theory: Ollama's response is a stream, and a
    # naive read-timeout resets on any partial byte received, so a
    # connection that trickles data without ever completing can outlast a
    # timeout that's only measuring silence, not total request time.
    # asyncio.wait_for is different in kind, not just a second copy of the
    # same idea — it's a real wall-clock deadline enforced by asyncio
    # itself, and because ainvoke() here runs on the native async httpx
    # client (not a to_thread-wrapped sync one), cancelling it actually
    # propagates into the request and aborts the connection, rather than
    # abandoning an unkillable OS thread the way the to_thread cases
    # earlier tonight could only leak.
    import httpx
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=120.0)
    except (httpx.TimeoutException, asyncio.TimeoutError):
        log.warning("session summary LLM call timed out, retrying once (campaign=%s)", campaign.id)
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=120.0)
    text = _extract_text(response.content)

    # Parse the structured response
    chronicle = text.strip()
    key_events: list[str] = []
    adventure_progress = ""
    relations: list[tuple[str, str, str]] = []

    if chronicle_marker in text and events_marker in text:
        parts = text.split(events_marker, 1)
        chronicle = parts[0].replace(chronicle_marker, "").strip()
        rest = parts[1]
        if progress_marker in rest:
            events_part, rest = rest.split(progress_marker, 1)
        else:
            events_part = rest
            rest = ""
        for line in events_part.splitlines():
            line = line.lstrip("-•* \t").strip()
            if line:
                key_events.append(line)

        if relations_marker in rest:
            progress_part, relations_part = rest.split(relations_marker, 1)
            adventure_progress = progress_part.strip()
            relations = [
                (m.group(1), m.group(2), m.group(3))
                for m in _NEW_RELATION_RE.finditer(relations_part)
            ]
        else:
            adventure_progress = rest.strip()

    return chronicle, key_events, adventure_progress, relations
