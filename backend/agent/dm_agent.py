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
`summarize_session()` generates a narrative chronicle + key events list from
that history, suitable for storing as a session record and indexing in
HistoryStore for future RAG retrieval.
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
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.types import Command
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from backend.config import settings
from backend.data.spells import ALL_SPELLS, SPELL_MENUS
from backend.models import Campaign
from backend.agent.prompts import get_mechanics_system_prompt, get_narrator_system_prompt
from backend.agent.session_zero_prompt import get_session_zero_mechanics_prompt, get_session_zero_narrator_prompt
from backend.stores.campaign_store import CampaignStore
from backend.stores.draft_store import DraftStore
from backend.stores.history_store import HistoryStore
from backend.stores.rules_store import RulesStore
from backend.tools import chargen, companion, dice, rules
from backend.tools import campaign as campaign_tools  # aliased — `campaign` is used throughout this file as a Campaign instance
from backend.tools.combat import build_encounter_context
from backend.tools.npc import build_traveling_npcs_context
from backend.tools.registry import get_tools, get_world_prep_tools
from backend.tools.resolution import resolve_pending_action_impl

log = logging.getLogger(__name__)

# ── module-level singletons (set by lifespan) ─────────────────────────────────

_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None

# Per-campaign locks to serialize parallel tool calls. LangGraph's ToolNode
# runs multi-tool AIMessages via asyncio.gather; each tool does load→mutate→save
# on the campaign. Without serialization the last writer silently wins.
_tool_locks: dict[str, asyncio.Lock] = {}


def _get_tool_lock(campaign_id: str) -> asyncio.Lock:
    if campaign_id not in _tool_locks:
        _tool_locks[campaign_id] = asyncio.Lock()
    return _tool_locks[campaign_id]


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

def _get_model() -> ChatOllama:
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
    called. settings.mechanics_model (gemma4:26b-mlx) has been extensively
    live-tested elsewhere in this app for reliable, genuine tool-calling
    discipline (including self-correcting after guardrail rejections across
    multi-turn combat) — standardizing on one validated model closes this
    failure class rather than working around it per call site. Note this
    model choice didn't fully close the gap for Session 0 either, hence the
    2026-07-04 structural split — see above."""
    return ChatOllama(
        model=settings.mechanics_model,
        base_url=settings.ollama_base_url,
        temperature=0.7,
        reasoning=False,
    )


def _get_mechanics_model() -> ChatOllama:
    """Low-temperature tool-calling model for the in-game agent.

    reasoning=False (2026-07-04, applies to all three ChatOllama instances
    in this file): gemma4:26b-mlx always wraps output in a
    <|channel>thought...<channel|> block, empty or not, whenever thinking
    isn't explicitly disabled. langchain_ollama's `reasoning` param
    defaults to None, which per its own docs leaves any such tags embedded
    directly in `.content` instead of split into `additional_kwargs` — a
    near-exact match for a previously-investigated bug (garbled
    <channel|>thought fragments leaking into a Session 0 reply, "fixed" at
    the time by purging unbounded scratch, which likely just made the leak
    rarer rather than addressing its cause). None of these three roles ever
    read reasoning_content, so False (skip reasoning entirely) rather than
    True (perform it, but capture it separately) is the right call — no
    product value in paying the latency for reasoning nothing uses.

    keep_alive is left at Ollama's default (idle-timeout eviction) rather
    than forced residency — tried keep_alive=-1 while chasing a live hang
    during combat testing (a request landing mid-idle-eviction seemed to
    leave the MLX runner stuck reporting "Stopping..." indefinitely), and it
    did get further before a second, different stall hit even with the
    model pinned resident — so forced residency wasn't a real fix for that
    class of hang, just extra always-on memory pressure for an unconfirmed
    benefit. Reverted."""
    return ChatOllama(
        model=settings.mechanics_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        reasoning=False,
    )


def _get_narrator_model() -> ChatOllama:
    """Higher-temperature prose model for the in-game agent. No tool access.

    Same underlying model as the mechanics node (settings.mechanics_model),
    just a different temperature/prompt — benchmarked against a smaller
    dedicated narrator model (gemma4:12b-mlx) and found to be both faster
    at raw generation and to incur no residency/swap cost either way, so a
    second model wasn't worth the extra ~7.6GB resident and the added
    config surface."""
    return ChatOllama(
        model=settings.mechanics_model,
        base_url=settings.ollama_base_url,
        temperature=0.8,
        reasoning=False,
    )


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


def _make_tool_node(tools: list[BaseTool], campaign_id: str) -> ToolNode:
    """Wrap tools in a ToolNode that serializes execution per campaign.

    LangGraph gathers parallel tool calls with asyncio.gather; the campaign
    store uses delete-all/reinsert so the last writer wins. The lock ensures
    only one tool's load→mutate→save cycle runs at a time per campaign.
    """
    lock = _get_tool_lock(campaign_id)

    async def _serialize(request, execute):
        async with lock:
            return await execute(request)

    return ToolNode(tools, awrap_tool_call=_serialize, handle_tool_errors=_handle_any_tool_error)


_STATE_CHANGE_TOOLS = {
    "update_character_hp", "update_monster_hp", "add_condition", "remove_condition",
    "advance_initiative", "end_encounter",
}


def _messages_since_last_human(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages appended since the current player turn started — walks
    backward until (excluding) the most recent HumanMessage. This is the
    mechanics loop's scratch-work for the turn in progress: tool-call
    AIMessages and their ToolMessage results."""
    since: list[BaseMessage] = []
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        since.append(m)
    since.reverse()
    return since


def _tools_called_since_last_human(messages: list[BaseMessage]) -> set[str]:
    """Tool names called since the current player turn started. Used by
    _detect_missing_followup to see what the mechanics model has already
    done this turn without needing extra state to track it."""
    called: set[str] = set()
    for m in _messages_since_last_human(messages):
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


def _turn_was_explicitly_ended(messages: list[BaseMessage]) -> bool:
    """True if this turn's scratch already advanced (or asked to advance) the
    initiative pointer — either a direct advance_initiative call, or a
    resolve_attack/resolve_saving_throw call with end_turn=True. Distinct from
    _tools_called_since_last_human because this needs each call's ARGS, not
    just its name. Doesn't specially recognize a resolve_pending_action call
    honoring a stored attacker_wanted_end_turn — that intent lives in
    persisted PendingAction state, not in this turn's own call args; accepted
    as a minor blind spot since that path is rare and resolve_pending_action's
    own docstring already tells the model when a turn still needs finishing."""
    for m in _messages_since_last_human(messages):
        if not (isinstance(m, AIMessage) and m.tool_calls):
            continue
        for c in m.tool_calls:
            if c["name"] == "advance_initiative":
                return True
            if c["name"] in ("resolve_attack", "resolve_saving_throw") and c.get("args", {}).get("end_turn"):
                return True
    return False


_RESOLUTION_TOOLS = {"resolve_attack", "resolve_saving_throw", "cast_spell", "resolve_pending_action"}


async def _detect_missing_followup(
    state: DMState, campaign_id: str, store: CampaignStore
) -> str | None:
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
    tool)."""
    campaign = await store.load(campaign_id)
    if not campaign or not campaign.active_encounter or not campaign.active_encounter.is_active:
        return None
    called = _tools_called_since_last_human(state["messages"])
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


_LOOT_TOOLS = {"update_character_currency", "add_item_to_character", "create_magic_item", "reveal_loot"}

# Gain-verbs broadened beyond "finds/receives" to cover mundane handoffs/pickups
# ("snatches the weapon", "recovers his shortsword") that the original verb list
# missed entirely — observed live: a companion picking up a dropped weapon, and
# a body search turning up a pouch/ring/shortsword, both slipped through with no
# add_item_to_character call because no word here matched.
_LOOT_GAIN_VERBS = (
    r"gains?|receives?|finds?|found|discovers?|discovered|recovers?|retrieves?|"
    r"grabs?|snatch(?:es)?|takes?|strips?|pockets?|picks?\s+up|hands?\s+over"
)
# Mundane gear nouns so a plain pickup/handoff of ordinary equipment (not just
# currency/treasure/a magic item) still trips the guardrail.
_MUNDANE_GEAR_NOUNS = r"sword|shortsword|longsword|greatsword|dagger|blade|weapon|pouch|ring|coins?|purse"

_LOOT_MENTION_RE = re.compile(
    r"\b\d+\s*(?:gp|sp|cp|pp|ep)\b"
    r"|\b(?:gold|silver|copper|platinum|electrum)\s+pieces?\b"
    rf"|\b(?:{_LOOT_GAIN_VERBS})\b[^.]{{0,40}}\b"
    rf"(?:gp|gold|coin|coins|treasure|gem|item|{_MUNDANE_GEAR_NOUNS})\b"
    # Magic item shape ("+1 Longsword", "+2 studded leather armor") — a bonus
    # followed by either a capitalized name or a common equipment noun, not
    # just any word, so an unrelated "+5 to his attack roll" doesn't match.
    r"|\+\d+\s+(?:(?-i:[A-Z]\w*)|studded|leather|chain|plate|scale|splint|banded|hide|"
    r"padded|longsword|shortsword|greatsword|dagger|mace|axe|hammer|bow|armor|"
    r"shield|ring|wand|staff|rod|amulet|cloak|boots|gloves|potion|scroll)\b",
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


_COMBAT_RESOLUTION_TOOLS = {"resolve_attack", "resolve_saving_throw"}


async def _detect_missing_encounter_followup(
    state: DMState, campaign_id: str, store: CampaignStore
) -> str | None:
    """Catches resolve_attack/resolve_saving_throw landing against a Monster that
    survived (current_hp > 0) with no active_encounter backing it — observed
    live: a multi-guard ambush resolved entirely through bare resolve_attack
    calls, with no start_encounter ever called, no initiative order, no round
    tracking. A single decisive blow against a target that dies or is otherwise
    fully resolved is fine standalone (the bright-line rule in the Combat
    prompt section covers that); this only fires when the target is still
    capable of acting back and nothing has formalized the fight."""
    calls = [
        c for m in _messages_since_last_human(state["messages"])
        if isinstance(m, AIMessage) and m.tool_calls
        for c in m.tool_calls
        if c["name"] in _COMBAT_RESOLUTION_TOOLS
    ]
    if not calls:
        return None
    campaign = await store.load(campaign_id)
    if not campaign or (campaign.active_encounter and campaign.active_encounter.is_active):
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


def get_agent(
    campaign: Campaign,
    store: CampaignStore,
    rules_store: RulesStore,
    history_store: HistoryStore,
):
    """Build the in-game DM agent: a mechanics node (tool-calling loop) that
    hands off to a narrator node (prose, no tools) once its tool calls are
    resolved. Only the narrator's output is ever appended to `messages` as
    the turn's DM turn — see mechanics_node below."""
    if _checkpointer is None:
        raise RuntimeError("Agent lifespan not started — call agent_lifespan() first.")

    tools = get_tools(campaign.id, store, rules_store, history_store, campaign.books_in_play)
    mechanics_modifier = _make_mechanics_modifier(get_mechanics_system_prompt(campaign), campaign.id, store)
    narrator_modifier = _make_narrator_modifier(get_narrator_system_prompt(campaign), campaign.id, store)
    mechanics_model = _get_mechanics_model().bind_tools(tools)
    narrator_model = _get_narrator_model()

    async def mechanics_node(state: DMState) -> Command[Literal["tools", "narrator", "mechanics"]]:
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
            scratch = _messages_since_last_human(state["messages"])
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
        called_this_turn = _tools_called_since_last_human(state["messages"])
        if "resolve_attack" not in called_this_turn and "cast_spell" not in called_this_turn:
            async with _get_tool_lock(campaign.id):
                live_campaign = await store.load(campaign.id)
                enc = live_campaign.active_encounter
                if enc and enc.pending_action:
                    decline_note = await resolve_pending_action_impl(live_campaign)
                    await store.save(live_campaign)
                    notes = (notes + "\n" if notes else "") + f"[Stale pending reaction auto-declined] {decline_note}"

        # Guardrail: catch the mechanics model stopping mid-resolution during
        # combat (rolling/narrating an outcome without the tool call that
        # actually applies it — see _detect_missing_followup), or claiming a
        # loot/currency gain with no backing tool call at all, combat or not
        # (see _detect_missing_loot_followup). Capped at one retry per PLAYER
        # TURN via correction_count (reset in stream_response), not per
        # no-tool-calls cycle — a model that stops short after nearly every
        # real tool call could otherwise re-trigger this every loop iteration
        # and never reach the recursion limit's stop condition.
        if correction_count < 1:
            issue = await _detect_missing_followup(state, campaign.id, store)
            if not issue:
                issue = _detect_missing_loot_followup(notes, called_this_turn)
            if not issue:
                issue = await _detect_missing_encounter_followup(state, campaign.id, store)
            if issue:
                log.debug("guardrail fired: %s", issue)
                return Command(
                    goto="mechanics",
                    update={
                        "correction_note": issue,
                        "correction_count": correction_count + 1,
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
        scratch = _messages_since_last_human(state["messages"])
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
        correction_count = state.get("correction_count", 0)

        # See get_agent()'s mechanics_node for the full rationale — same bounded
        # retry for a bad/hallucinated tool call, distinct from correction_count.
        tool_error_count = state.get("tool_error_count", 0)
        tool_error_count = tool_error_count + 1 if _last_tool_batch_had_error(state["messages"]) else 0
        if tool_error_count > _MAX_TOOL_ERROR_RETRIES:
            scratch = _messages_since_last_human(state["messages"])
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
        scratch = _messages_since_last_human(state["messages"])
        tool_outputs = [
            _extract_text(m.content) for m in scratch
            if isinstance(m, ToolMessage) and getattr(m, "name", None) == "list_options"
        ]
        if tool_outputs:
            notes += (
                "\n\n[VERBATIM TOOL OUTPUT — the only real menu; do not add to it]\n"
                + "\n\n".join(tool_outputs)
            )

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
):
    """One-shot, non-interactive agent for automatic world-prep. No
    checkpointer — this isn't a resumable conversation, just a single
    bounded tool-calling run whose prompt is passed directly as a HumanMessage."""
    tools = get_world_prep_tools(campaign.id, store, rules_store, books_in_play)

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


async def watch_for_stalls(source: AsyncIterator, stall_after: float = STALL_TIMEOUT) -> AsyncIterator:
    """Wrap an async iterator, yielding STALL if `stall_after` seconds pass
    with no new item. Draining happens in a background task so a stall
    report never cancels or otherwise disturbs the underlying call — by the
    time we know enough to call it "stuck" rather than "slow," canceling it
    ourselves would just trade one guess for another. Re-raises any
    exception from the source (e.g. a connection error once the wedged
    runner is killed) once draining reaches it."""
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
    already_stalled = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=stall_after)
            except asyncio.TimeoutError:
                if not already_stalled:
                    already_stalled = True
                    yield STALL
                continue
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            already_stalled = False
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
    narrator-facing text. Defense-in-depth alongside reasoning=False on
    every ChatOllama instance (see _get_mechanics_model's docstring) for
    the case where a leak reaches `.content` anyway. Not chunk-boundary-safe
    — a tag split across two stream chunks won't be caught here; not worth
    buffering the live game's token-by-token stream to fix that."""
    return _REASONING_LEAK_RE.sub("", text)


async def stream_response(
    campaign: Campaign,
    store: CampaignStore,
    rules_store: RulesStore,
    history_store: HistoryStore,
    user_message: str,
    thread_id: str,
) -> AsyncIterator[str]:
    """Yield plain-text tokens from the narrator node as they stream. The
    mechanics node's tool-calling loop runs first and is never streamed —
    filtering by langgraph_node keeps its tool-call JSON and reasoning off
    the wire entirely."""
    agent = get_agent(campaign, store, rules_store, history_store)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

    async for event in agent.astream_events(
        {"messages": [HumanMessage(content=user_message)], "correction_count": 0, "tool_error_count": 0},
        config=config,
        version="v2",
    ):
        if event["event"] != "on_chat_model_stream":
            continue
        if event.get("metadata", {}).get("langgraph_node") != "narrator":
            continue
        chunk = event["data"].get("chunk")
        if chunk and chunk.content:
            yield strip_reasoning_leakage(chunk.content)


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

After the chronicle, list 5-10 KEY EVENTS as short bullet points capturing the
most plot-relevant moments.

Format your response EXACTLY as:
{chronicle_marker}
<narrative here>
{events_marker}
- event 1
- event 2

Transcript:
{transcript}"""


async def summarize_session(
    thread_id: str,
    campaign_name: str,
) -> tuple[str, list[str]]:
    """Generate a narrative chronicle and key-events list for a session.

    Returns (summary_text, key_events_list). Both are empty strings / lists
    if the thread has no dialogue.
    """
    messages = await get_thread_messages(thread_id)
    turns = format_transcript(messages)

    if not turns:
        return "Session contained no dialogue to summarize.", []

    transcript = "\n\n".join(
        f"{'PLAYER' if t['role'] == 'player' else 'DM'}: {t['content']}"
        for t in turns
    )

    # Random per-call markers prevent player-injected text from confusing the
    # parser — a player can't predict or inject "---CHRONICLE-<hex>---".
    token = secrets.token_hex(8)
    chronicle_marker = f"---CHRONICLE-{token}---"
    events_marker = f"---EVENTS-{token}---"

    prompt = _SUMMARY_PROMPT.format(
        campaign_name=campaign_name,
        transcript=transcript,
        chronicle_marker=chronicle_marker,
        events_marker=events_marker,
    )

    llm = _get_model()
    response = await llm.ainvoke(prompt)
    text = _extract_text(response.content)

    # Parse the structured response
    chronicle = text.strip()
    key_events: list[str] = []

    if chronicle_marker in text and events_marker in text:
        parts = text.split(events_marker, 1)
        chronicle = parts[0].replace(chronicle_marker, "").strip()
        for line in parts[1].splitlines():
            line = line.lstrip("-•* \t").strip()
            if line:
                key_events.append(line)

    return chronicle, key_events
