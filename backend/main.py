"""
FastAPI application — DM web interface.

Routes
------
GET  /                                → campaign selector (index.html)
POST /campaigns                       → create campaign, redirect to /campaigns/{id}
GET  /campaigns/{id}                  → game interface (game.html)
DELETE /campaigns/{id}                → delete campaign, redirect to /
GET  /campaigns/{id}/stream           → SSE: stream DM response (query: thread_id, message)
GET  /campaigns/{id}/thread-info      → JSON: message count vs the mechanics trim window (query: thread_id)
POST /campaigns/{id}/message          → queue a player message (HTMX form)
POST /campaigns/{id}/session/begin    → queue a server-built session-opening message (first-session intro or recap)
POST /campaigns/{id}/books            → add adventure slug to campaign mid-run
POST /campaigns/{id}/session/end      → summarise session, save chronicle, return new thread_id
GET  /campaigns/{id}/sessions         → session list page (sessions.html)
GET  /campaigns/{id}/sessions/{sid}   → session detail / transcript page
POST /campaigns/{id}/safety-flag      → player X-card: flag a topic for the DM agent to avoid
POST /campaigns/{id}/safety-flag/clear → DM-only: clear active safety flags
GET  /campaigns/{id}/rolls            → JSON list of recent dice rolls
GET  /campaigns/{id}/party/{char_id}  → JSON: full character sheet for the right-side detail panel
GET  /campaigns/{id}/messages         → JSON: prior turns for a thread, used to hydrate the chat panel on reload
POST /campaigns/{id}/session-zero/fill-party → DM-triggered: generate a companion if the party is short
POST /campaigns/{id}/rest/long        → whole-party long rest (deterministic, no LLM) — see apply_long_rest
POST /campaigns/{id}/rest/short       → whole-party short rest (deterministic, no LLM) — see apply_short_rest
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from sqlalchemy.ext.asyncio import create_async_engine

import re

from langchain_core.messages import HumanMessage

from backend.agent.dm_agent import (
    STALL,
    STALL_MESSAGE,
    agent_lifespan,
    format_transcript,
    get_context_status,
    get_session_zero_agent,
    get_thread_messages,
    run_fill_party,
    stream_response,
    strip_reasoning_leakage,
    summarize_session,
    watch_for_stalls,
)
from backend.agent.prompts import build_session_kickoff_message
from backend.agent.world_prep import run_world_prep
from backend.config import settings
from backend.data.fivee_options import CLASSES
from backend.models import Campaign, Session
from backend.rag.reranker import LLMJudgeReranker
from backend.tools._helpers import apply_long_rest, apply_short_rest, find_char, find_location, find_npc, read_adventure_meta
from backend.stores.campaign_store import CampaignStore
from backend.stores.draft_store import draft_store
from backend.stores.graph_store import PostgresRelationGraphStore
from backend.stores.history_store import HistoryStore
from backend.stores.lore_store import LoreStore
from backend.stores.rules_store import RulesStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)

_engine = create_async_engine(settings.database_url)

# ── lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with agent_lifespan():
        yield

app = FastAPI(title="D&D DM", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── shared singletons ─────────────────────────────────────────────────────────

def _store() -> CampaignStore:
    return CampaignStore(_engine)

# One reranker instance shared by both stores (constructor injection).
# LLMJudgeReranker (Ollama-based), not CrossEncoderReranker — this is a
# low-throughput single-user app, so the extra Ollama round-trip's latency
# is negligible next to the mechanics/narrator calls already made every
# turn, and it keeps the container light: no torch/sentence-transformers,
# which real testing showed OOM-killing under Docker Desktop's default
# memory allocation anyway.
_reranker = LLMJudgeReranker()
_rules_store = RulesStore(settings.chroma_persist_dir, reranker=_reranker)
_rules_store.load()  # opens the existing ChromaDB collection — without this, is_ready() is
                      # permanently False and every search_rules call fails with "not ready",
                      # silently undermining every "ground it via search_rules" instruction.
_history_store = HistoryStore(settings.chroma_persist_dir, settings.ollama_base_url, reranker=_reranker)

def _rules() -> RulesStore:
    return _rules_store

def _history() -> HistoryStore:
    return _history_store

def _lore() -> LoreStore:
    return LoreStore(_engine)

def _graph() -> PostgresRelationGraphStore:
    return PostgresRelationGraphStore(_engine)

def _scan_adventures() -> list[dict]:
    adv_dir = Path("docs/source/adventures")
    if not adv_dir.exists():
        return []
    result = []
    for d in sorted(adv_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        result.append({"slug": d.name, **read_adventure_meta(d.name)})
    return result


# ── in-memory message queues keyed by campaign_id ────────────────────────────

_queues: dict[str, asyncio.Queue] = {}

def _queue(campaign_id: str) -> asyncio.Queue:
    if campaign_id not in _queues:
        _queues[campaign_id] = asyncio.Queue()
    return _queues[campaign_id]


# ── background world-prep tasks ───────────────────────────────────────────────
# Holds references to fire-and-forget asyncio.Tasks so they aren't garbage
# collected mid-run (nothing else holds a reference to a bare create_task()).

_background_tasks: set[asyncio.Task] = set()

def _spawn_world_prep(campaign_id: str) -> None:
    task = asyncio.create_task(run_world_prep(campaign_id, _store(), _rules(), _lore(), _graph()))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    campaigns  = await _store().list_all()
    adventures = _scan_adventures()
    return templates.TemplateResponse(
        request, "index.html", {"campaigns": campaigns, "adventures": adventures}
    )


@app.post("/campaigns")
async def create_campaign(
    name:       str       = Form(...),
    setting:    str       = Form(""),
    adventures: list[str] = Form(default=[]),
):
    campaign = Campaign(name=name, setting=setting, books_in_play=adventures)
    await _store().create(campaign)
    if campaign.books_in_play:
        _spawn_world_prep(campaign.id)
    return RedirectResponse(f"/campaigns/{campaign.id}", status_code=303)


@app.get("/campaigns/{campaign_id}", response_class=HTMLResponse)
async def game(request: Request, campaign_id: str):
    campaign   = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    adventures = _scan_adventures()
    available  = [a for a in adventures if a["slug"] not in campaign.books_in_play]
    # Generate a thread_id for this page load; the client persists it in
    # sessionStorage so refreshes within the same browser tab reuse it.
    thread_id  = f"{campaign_id}:{uuid.uuid4().hex}"
    return templates.TemplateResponse(
        request, "game.html", {
            "campaign": campaign,
            "available_adventures": available,
            "thread_id": thread_id,
        }
    )


@app.post("/campaigns/{campaign_id}/books")
async def add_adventure(campaign_id: str, adventure: str = Form(...)):
    store    = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if adventure not in campaign.books_in_play:
        campaign.books_in_play.append(adventure)
        await store.save(campaign)
        _spawn_world_prep(campaign_id)
    return RedirectResponse(f"/campaigns/{campaign_id}", status_code=303)


@app.delete("/campaigns/{campaign_id}")
async def delete_campaign(campaign_id: str):
    await _store().delete(campaign_id)
    return RedirectResponse("/", status_code=303)


@app.post("/campaigns/{campaign_id}/message")
async def post_message(campaign_id: str, message: str = Form(...), thread_id: str = Form(...)):
    await _queue(campaign_id).put((thread_id, message.strip()))
    return HTMLResponse("", status_code=204)


@app.post("/campaigns/{campaign_id}/session/begin")
async def session_begin(campaign_id: str, thread_id: str = Form(...)):
    """"Begin/Continue the Adventure" button — builds a session-opening
    message server-side (first-session intro or recap of the last chronicle,
    see build_session_kickoff_message) and enqueues it exactly like a normal
    player message, so it flows through the same mechanics -> narrator
    pipeline and streams back over the existing /stream SSE endpoint."""
    campaign = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    kickoff = build_session_kickoff_message(campaign)
    await _queue(campaign_id).put((thread_id, kickoff))
    return HTMLResponse("", status_code=204)


@app.get("/campaigns/{campaign_id}/stream")
async def stream(request: Request, campaign_id: str, thread_id: str):
    """SSE endpoint. Reads thread_id from query param so the client controls
    which LangGraph thread (and therefore which conversation history) to use."""
    store   = _store()
    rules   = _rules()
    history = _history()
    q       = _queue(campaign_id)

    async def event_generator():
        campaign = await store.load(campaign_id)
        if not campaign:
            yield {"event": "error", "data": "Campaign not found"}
            return

        while True:
            if await request.is_disconnected():
                break

            try:
                queued_tid, message = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": ""}
                continue

            # Discard messages queued by a different tab or a previous session.
            if queued_tid != thread_id:
                continue

            campaign = await store.load(campaign_id)

            try:
                async for token in watch_for_stalls(
                    stream_response(campaign, store, rules, history, message, thread_id, _lore(), _graph())
                ):
                    if token is STALL:
                        yield {"event": "stall", "data": STALL_MESSAGE}
                        continue
                    yield {"event": "token", "data": token}
            except Exception:
                log.exception("Streaming turn failed for campaign %s, thread %s", campaign_id, thread_id)
                yield {"event": "error", "data": "Lost connection to the model backend."}
                continue

            yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@app.get("/campaigns/{campaign_id}/thread-info")
async def thread_info(thread_id: str):
    """Raw message count for a thread vs the mechanics trim window, so the
    frontend can warn the player before older context silently drops (see
    get_context_status in dm_agent.py)."""
    return JSONResponse(await get_context_status(thread_id))


@app.get("/campaigns/{campaign_id}/messages")
async def get_messages(campaign_id: str, thread_id: str):
    """Prior turns for a thread — used to hydrate the chat panel on page
    load/reload. The client always knows the right thread_id (sessionStorage,
    see game.html) but the server never re-rendered its history into the DOM
    on a fresh page load, so a reload looked like the conversation vanished
    even though it was intact in the LangGraph checkpoint the whole time."""
    messages = await get_thread_messages(thread_id)
    return JSONResponse(format_transcript(messages))


# ── session management ────────────────────────────────────────────────────────

@app.post("/campaigns/{campaign_id}/session/end")
async def end_session(campaign_id: str, thread_id: str = Form(...)):
    """Summarise the current session, save a Session record, index the chronicle
    in ChromaDB, and return a fresh thread_id for the next session."""
    store    = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Fix 5: if this thread was already summarised (e.g. retry after Ollama failure),
    # return the existing record rather than creating a duplicate.
    existing = next((s for s in campaign.sessions if s.thread_id == thread_id), None)
    if existing:
        new_thread_id = f"{campaign_id}:{uuid.uuid4().hex}"
        return JSONResponse({
            "session_id": existing.id,
            "session_number": existing.session_number,
            "new_thread_id": new_thread_id,
            "key_events": existing.key_events,
            "summary_preview": existing.summary[:400],
        })

    summary, key_events, adventure_progress, relations = await summarize_session(thread_id, campaign, _rules())

    session_number = len(campaign.sessions) + 1
    session = Session(
        session_number=session_number,
        real_date=date.today(),
        summary=summary,
        key_events=key_events,
        adventure_progress=adventure_progress,
        thread_id=thread_id,
    )
    campaign.sessions.append(session)
    campaign.session_count = session_number
    await store.save(campaign)

    # Stage 1.5: best-effort incremental relation-graph update from this
    # session's newly-stated relationships — resolves each name against
    # live NPCs/Locations/party characters, silently skipping any name that
    # doesn't resolve to a known entity (the summarizer has no tool access,
    # so it can occasionally paraphrase a name slightly).
    if relations:
        graph_store = _graph()
        for source_name, relation, target_name in relations:
            src = find_npc(campaign, source_name) or find_location(campaign, source_name) or find_char(campaign, source_name)
            dst = find_npc(campaign, target_name) or find_location(campaign, target_name) or find_char(campaign, target_name)
            if src and dst:
                src_type = "npc" if src in campaign.npcs else ("location" if src in campaign.locations else "character")
                dst_type = "npc" if dst in campaign.npcs else ("location" if dst in campaign.locations else "character")
                try:
                    await graph_store.add_edge(
                        campaign_id, src_type, src.id, src.name, dst_type, dst.id, dst.name, relation,
                    )
                except Exception:
                    log.exception("Failed to record session-end relation %s -> %s for campaign %s", source_name, target_name, campaign_id)

    # Fix 4: OllamaEmbeddings.embed_documents is a blocking httpx call — run it
    # off the event loop. Fix 5: if Ollama is down, log and continue; the chronicle
    # is already in Postgres and the route must not fail silently on retry.
    try:
        await asyncio.to_thread(
            lambda: _history().add_session(campaign_id, session.id, session_number, summary, key_events)
        )
    except Exception:
        log.exception("ChromaDB indexing failed for session %s (campaign %s); chronicle is safe in Postgres", session.id, campaign_id)

    new_thread_id = f"{campaign_id}:{uuid.uuid4().hex}"

    return JSONResponse({
        "session_id": session.id,
        "session_number": session_number,
        "new_thread_id": new_thread_id,
        "key_events": key_events,
        "summary_preview": summary[:400],
    })


@app.get("/campaigns/{campaign_id}/sessions", response_class=HTMLResponse)
async def sessions_list(request: Request, campaign_id: str):
    campaign = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    sessions = sorted(campaign.sessions, key=lambda s: s.session_number)
    return templates.TemplateResponse(
        request, "sessions.html", {"campaign": campaign, "sessions": sessions, "selected": None}
    )


@app.get("/campaigns/{campaign_id}/sessions/{session_id}", response_class=HTMLResponse)
async def session_detail(request: Request, campaign_id: str, session_id: str):
    campaign = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    session = next((s for s in campaign.sessions if s.id == session_id), None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    transcript = []
    if session.thread_id:
        messages   = await get_thread_messages(session.thread_id)
        transcript = format_transcript(messages)

    sessions = sorted(campaign.sessions, key=lambda s: s.session_number)
    return templates.TemplateResponse(
        request, "sessions.html", {
            "campaign": campaign,
            "sessions": sessions,
            "selected": session,
            "transcript": transcript,
        }
    )


# ── session zero ─────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-") or "player"


@app.get("/campaigns/{campaign_id}/session-zero", response_class=HTMLResponse)
async def session_zero_index(request: Request, campaign_id: str):
    campaign = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    recommended = [
        {"slug": slug, **read_adventure_meta(slug)}
        for slug in campaign.books_in_play
        if read_adventure_meta(slug).get("recommended_players")
    ]
    return templates.TemplateResponse(
        request, "session_zero_index.html",
        {"campaign": campaign, "recommended": recommended, "class_names": list(CLASSES.keys())}
    )


@app.get("/campaigns/{campaign_id}/party/{character_id}")
async def get_party_member(campaign_id: str, character_id: str):
    """Full character sheet for the right-side detail panel on the game
    screen — JSON, rendered client-side (same convention as the Session 0
    draft endpoint)."""
    campaign = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    char = next((c for c in campaign.party if c.id == character_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    return JSONResponse(char.model_dump(mode="json"))


@app.delete("/campaigns/{campaign_id}/party/{character_id}")
async def remove_party_member(campaign_id: str, character_id: str):
    """DM-only: remove a character from the party — a misgenerated companion,
    a duplicate, or a player who's dropped out. Works for both PCs and
    DM companions; redirects back to the Session 0 lobby either way."""
    store = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    campaign.party = [c for c in campaign.party if c.id != character_id]
    await store.save(campaign)
    return RedirectResponse(f"/campaigns/{campaign_id}/session-zero", status_code=303)


@app.post("/campaigns/{campaign_id}/session-zero/fill-party")
async def session_zero_fill_party(campaign_id: str, char_class: str = Form("")):
    """DM-triggered one-shot: ask the DM agent to generate a companion if
    the party is short of the adventure's recommended size, or of the
    explicitly requested class if the DM picked one from the dropdown
    (bypassing the model's own class judgment). Synchronous — typically
    takes a couple of minutes (1-2 tool calls on qwen2.5:14b), unlike
    world-prep's multi-book background pass; the frontend shows a loading
    state rather than treating this like an instant action."""
    store = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    summary = await run_fill_party(campaign, store, requested_class=char_class or None)
    return JSONResponse({"summary": summary})


@app.post("/campaigns/{campaign_id}/rest/long")
async def long_rest(campaign_id: str):
    """Whole-party long rest, triggered directly from the UI button — pure
    deterministic arithmetic (see apply_long_rest), no LLM call, so it's
    instant and can't hallucinate. Also the only place that ever sets
    last_long_rest_day, which the sidebar's rest-status line depends on."""
    store = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    summary = apply_long_rest(campaign)
    await store.save(campaign)
    return JSONResponse({"summary": summary})


@app.post("/campaigns/{campaign_id}/rest/short")
async def short_rest(campaign_id: str):
    """Whole-party short rest, triggered directly from the UI button — same
    deterministic, no-LLM reasoning as long_rest (see apply_short_rest)."""
    store = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    summary = apply_short_rest(campaign)
    await store.save(campaign)
    return JSONResponse({"summary": summary})


@app.post("/campaigns/{campaign_id}/session-zero")
async def session_zero_start(campaign_id: str, player_name: str = Form(...)):
    slug = _slugify(player_name)
    return RedirectResponse(
        f"/campaigns/{campaign_id}/session-zero/{slug}?player={player_name}",
        status_code=303,
    )


@app.get("/campaigns/{campaign_id}/session-zero/{player_slug}", response_class=HTMLResponse)
async def session_zero_chat(
    request: Request,
    campaign_id: str,
    player_slug: str,
    player: str = "",
):
    campaign = await _store().load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
    thread_id = f"{campaign_id}:chargen:{player_slug}:{uuid.uuid4().hex}"
    player_name = player or player_slug.replace("-", " ").title()
    return templates.TemplateResponse(
        request, "session_zero.html", {
            "campaign": campaign,
            "player_slug": player_slug,
            "player_name": player_name,
            "thread_id": thread_id,
        }
    )


@app.get("/campaigns/{campaign_id}/session-zero/{player_slug}/draft")
async def get_draft(campaign_id: str, player_slug: str):
    return JSONResponse(draft_store.get(campaign_id, player_slug))


@app.post("/campaigns/{campaign_id}/session-zero/{player_slug}/message")
async def session_zero_message(
    campaign_id: str, player_slug: str, message: str = Form(...), thread_id: str = Form(...)
):
    q_key = f"{campaign_id}:chargen:{player_slug}"
    await _queue(q_key).put((thread_id, message.strip()))
    return HTMLResponse("", status_code=204)


@app.get("/campaigns/{campaign_id}/session-zero/{player_slug}/stream")
async def session_zero_stream(
    request: Request,
    campaign_id: str,
    player_slug: str,
    thread_id: str,
):
    store   = _store()
    rules   = _rules()
    q_key   = f"{campaign_id}:chargen:{player_slug}"
    q       = _queue(q_key)

    async def event_generator():
        campaign = await store.load(campaign_id)
        if not campaign:
            yield {"event": "error", "data": "Campaign not found"}
            return

        agent = get_session_zero_agent(campaign, player_slug, store, rules, draft_store)
        # LangGraph's default recursion limit (25) is too low for a Session 0 turn
        # that walks through several tool calls in a row (e.g. list_options then
        # update_character_draft then get_draft_summary) — the main game's
        # stream_response already raises this to 60 for the same reason; Session 0
        # was missing the same override, confirmed by a live GraphRecursionError
        # during the spell-selection step, which added more back-and-forth per turn.
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}

        while True:
            if await request.is_disconnected():
                break
            try:
                queued_tid, message = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": ""}
                continue

            # Discard messages queued by a different tab or a previous chargen session.
            if queued_tid != thread_id:
                continue

            # Reload campaign so agent sees the latest party state.
            campaign = await store.load(campaign_id)
            agent = get_session_zero_agent(campaign, player_slug, store, rules, draft_store)

            # Session 0's narrator node can now retry itself (chargen_narrator_node
            # in dm_agent.py, guarded by _detect_invented_spells) before a reply
            # ever reaches the player — a discarded first draft must never have
            # streamed live, since there's no way to "unsend" tokens already in the
            # browser. So narrator tokens are buffered per-invocation (keyed by
            # run_id — a retry gets a new run_id, discarding whatever was
            # accumulated under the old one) and only the invocation that actually
            # survives to the end of the turn is flushed, as a single block. This
            # trades away the live-typing effect for Session 0 turns in exchange
            # for the player never seeing invented content, even briefly.
            narrator_buffer = ""
            narrator_run_id = None

            try:
                async for event in watch_for_stalls(agent.astream_events(
                    # correction_count/tool_error_count were never seeded here before —
                    # a pre-existing adjacent gap meaning each budget silently stayed
                    # exhausted for the rest of a thread after its first use, rather
                    # than resetting per player turn like the main game's stream_response.
                    {
                        "messages": [HumanMessage(content=message)],
                        "correction_count": 0,
                        "tool_error_count": 0,
                    },
                    config=config,
                    version="v2",
                )):
                    if event is STALL:
                        yield {"event": "stall", "data": STALL_MESSAGE}
                        continue
                    # finalize_character clears the draft on success — the preview
                    # panel would otherwise go blank right when the player expects
                    # confirmation. Surface a dedicated event so the frontend can
                    # show a success state instead of re-fetching an empty draft.
                    if event["event"] == "on_tool_end" and event.get("name") == "finalize_character":
                        output = event["data"].get("output")
                        content = getattr(output, "content", str(output))
                        if content.startswith("✓"):
                            yield {"event": "finalized", "data": content}
                        continue
                    if event["event"] != "on_chat_model_stream":
                        continue
                    # Session 0 is now a two-node mechanics/narrator graph (like the
                    # main game) — only the narrator's tokens are player-facing; the
                    # mechanics node's internal resolution note must never stream to
                    # the client, same reason stream_response filters by node.
                    if event.get("metadata", {}).get("langgraph_node") != "narrator":
                        continue
                    run_id = event.get("run_id")
                    if run_id != narrator_run_id:
                        narrator_run_id = run_id
                        narrator_buffer = ""
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content:
                        narrator_buffer += chunk.content
            except Exception:
                log.exception("Session 0 streaming turn failed for campaign %s, player %s", campaign_id, player_slug)
                yield {"event": "error", "data": "Lost connection to the model backend."}
                continue

            if narrator_buffer:
                yield {"event": "token", "data": strip_reasoning_leakage(narrator_buffer)}

            yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@app.post("/campaigns/{campaign_id}/safety-flag")
async def post_safety_flag(campaign_id: str, note: str = Form("")):
    """Player-facing X-card: add a topic the DM agent must steer away from, and
    log it permanently to campaign notes for the human DM."""
    store    = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    flag_text = note.strip() or "unspecified — steer away from the current scene"
    campaign.safety_flags.append(flag_text)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    log_line  = f"[Safety flag {timestamp}] {flag_text}"
    campaign.notes = f"{campaign.notes}\n{log_line}".strip() if campaign.notes else log_line

    await store.save(campaign)
    return JSONResponse({"active_flags": campaign.safety_flags})


@app.post("/campaigns/{campaign_id}/safety-flag/clear")
async def clear_safety_flags(campaign_id: str):
    """DM-only: clear active flags once handled. The note log is left intact."""
    store    = _store()
    campaign = await store.load(campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    campaign.safety_flags = []
    await store.save(campaign)
    return JSONResponse({"active_flags": []})


@app.get("/campaigns/{campaign_id}/rolls")
async def get_rolls(campaign_id: str, limit: int = 20):
    rolls = await _store().get_rolls(campaign_id, limit=limit)
    return JSONResponse([r.model_dump() for r in rolls])
