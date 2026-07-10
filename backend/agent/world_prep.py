"""Automatic world-prep — a one-shot background pass that reads a campaign's
selected adventure text and pre-populates region-scale locations/connections
before play begins. Fired from backend/main.py as an asyncio background task;
never blocks the HTTP response that triggers it.

A second phase, added 2026-07-05, seeds the OPENING SCENE specifically: real
NPC records for every named character tied to it, and site-scale detail
(description/points_of_interest/hidden_elements) for the location where play
actually begins — see _gather_opening_scene_context and the wiring at the
bottom of run_world_prep. Confirmed live that a generic region-scale seed
pass alone leaves the DM to invent the opening's cast from nothing.
"""

import asyncio
import json
import logging
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from backend.agent.world_prep_prompt import get_npc_prep_prompt, get_opening_location_prompt, get_world_prep_prompt
from backend.models import WorldPrepStatus
from backend.stores.campaign_store import CampaignStore
from backend.stores.graph_store import RelationGraphStore
from backend.stores.lore_store import LoreStore
from backend.stores.rules_store import RulesStore
from backend.tools._helpers import find_location, find_npc, read_adventure_meta

log = logging.getLogger(__name__)

_SEED_CONTEXT_MAX_CHARS = 20_000

# Out of the Abyss's full Chapter 1 is 65,615 chars; Curse of Strahd:
# Reloaded's Death House arc (Act I - Arc A) is 105,411 chars — both need to
# fit inside gemma4:26b-mlx's documented 32k-token (~128k-char) headroom
# (design.md), with margin left for the prompt's own instructions, the
# opening_hook/literal-match text _gather_opening_scene_context adds on top
# of this section, and this ReAct agent's own tool-call/response overhead.
# 110k covers both known chapters with a little room to spare — a 50k cap
# was tried first and silently cut off the numbered site/terrain breakdown
# (e.g. "## 13. Northern Watch Post") that sits near the end of Out of the
# Abyss's chapter. If a future adventure's opening section is bigger than
# this, raising the cap further eats directly into that remaining margin —
# check total prompt size against the model's real ceiling before doing so.
_OPENING_SECTION_CHARS = 110_000
_MAX_LITERAL_MATCHES = 15


def _gather_seed_context(book: str, max_chars: int = _SEED_CONTEXT_MAX_CHARS) -> str:
    """Deterministic, zero-Ollama-call seed context for the region-scale
    world-prep agent — replaces an earlier embedding-search version
    (search_adventure_only over "regional overview and map" / "travel
    distances..." / "major settlements..."). Root cause, confirmed live
    2026-07-08: alternating nomic-embed-text calls with gemma4:26b-mlx calls
    within one background task forces Ollama to swap models on every single
    world-prep run, and this codebase already has one documented prior
    incident of that exact transition leaving the MLX runner stuck
    ("Stopping..." indefinitely — see _get_mechanics_model()'s docstring in
    dm_agent.py). Removing the embedding call from world-prep removes the
    trigger itself, not just the odds of hitting it — every remaining Ollama
    call in this file's whole pipeline is gemma4:26b-mlx, no swap needed.

    Two deterministic sources, both a plain local file read:
      1. The adventure's own introduction — everything before its curated
         opening_section_marker (or the first max_chars, if that marker
         isn't curated for this book yet). Confirmed against Lost Mine of
         Phandelver: its Introduction/Background/Overview/Adventure Hook
         section is exactly the ~19K chars before "PART 1" — reliably where
         "regional overview"/"major settlements" content already lives,
         without needing to search for it.
      2. LocationExtractor's own pre-computed "_connections" (extract_
         entities.py's _entities.json) — already-grounded travel routes
         between named locations, extracted once at ingest time. This is
         *better* grounded than a fresh semantic search would have been
         (real source citations went into producing it), and it's the exact
         "travel distances and days between locations" content the old
         embedding query was trying to find live.
    """
    meta = read_adventure_meta(book)
    marker = meta.get("opening_section_marker", "")

    intro_text = ""
    for path in Path(f"docs/source/adventures/{book}").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        end = text.find(marker) if marker else -1
        intro_text += (text[:end] if end != -1 else text[:max_chars])[:max_chars]

    connections_text = ""
    entities_path = Path(f"docs/source/adventures/{book}/_entities.json")
    if entities_path.exists():
        try:
            registry = json.loads(entities_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = {}
        conns = registry.get("_connections", [])
        if conns:
            connections_text = "\n".join(
                f"{c.get('from', '')} -> {c.get('to', '')}: {c.get('via', '')}" for c in conns
            )

    parts = []
    if intro_text:
        parts.append(f"[Adventure introduction/overview]\n{intro_text}")
    if connections_text:
        parts.append(f"[Known location connections/travel routes]\n{connections_text}")
    return "\n\n===\n\n".join(parts)


def _read_opening_section(
    book: str, marker: str, end_marker: str = "", max_chars: int = _OPENING_SECTION_CHARS
) -> str:
    """Deterministic, non-RAG extraction: find `marker` (e.g. "## Chapter 1:
    Prisoners of the Drow") in the adventure's source markdown and return
    everything from there up to `end_marker` (or end-of-file, or max_chars —
    whichever comes first). This is the only reliable way to get a COMPLETE,
    uncut opening scene — confirmed directly that a name-proximity search
    (semantic or literal) misses most of a roster whose entries don't repeat
    the location's own name next to each one.

    end_marker is curated per-adventure (`opening_section_end_marker` in
    _meta.json) rather than derived automatically, because adventure source
    files vary in header structure: Out of the Abyss's OCR'd markdown is flat
    (every heading — real chapter boundary or in-chapter NPC entry alike —
    sits at the same `##` level), while a hand-authored file like Curse of
    Strahd: Reloaded has real nesting (`#` for Act/Arc titles, `##`/`###` for
    scenes and rooms below them). No single heading-level or regex rule
    covers both. Falling back to Out of the Abyss's original hardcoded
    `^## Chapter \\d+:` boundary when end_marker isn't curated keeps existing
    adventures working unchanged."""
    for path in Path(f"docs/source/adventures/{book}").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        idx = text.find(marker)
        if idx == -1:
            continue
        rest = text[idx:]
        after = rest[len(marker):]
        if end_marker:
            end_idx = after.find(end_marker)
            end = len(marker) + end_idx if end_idx != -1 else len(rest)
        else:
            m = re.search(r"^## Chapter \d+:", after, re.MULTILINE)
            end = len(marker) + m.start() if m else len(rest)
        return rest[:end][:max_chars]
    return ""


def _gather_opening_scene_context(
    rules_store: RulesStore,
    book: str,
    opening_location: str,
    opening_section_marker: str,
    opening_hook: str,
    opening_section_end_marker: str = "",
) -> str:
    """Combine three sources, in priority order: (1) the curated opening_hook
    text; (2) a deterministic bulk read of the whole opening chapter/section
    — the primary source, see _read_opening_section; (3) a capped exhaustive
    literal sweep for the location's name elsewhere in the book, catching any
    later foreshadowing/reappearance. Returns "" if opening_location/
    opening_section_marker aren't curated for this adventure yet, or nothing
    comes back — the caller treats that as "skip this phase," not an error,
    so every adventure without this curation degrades gracefully to today's
    region-only behavior."""
    if not opening_location or not opening_section_marker:
        return ""
    parts = []
    if opening_hook:
        parts.append(f"[Curated opening hook]\n{opening_hook}")
    section_text = _read_opening_section(book, opening_section_marker, opening_section_end_marker)
    if section_text:
        parts.append(f"[Opening chapter/section text]\n{section_text}")
    literal_chunks = rules_store.search_adventure_literal(
        opening_location, books_in_play=[book], limit=_MAX_LITERAL_MATCHES
    )
    if literal_chunks:
        parts.append(
            "[Other mentions of the opening location elsewhere in the book]\n"
            + "\n\n---\n\n".join(c.content for c in literal_chunks)
        )
    return "\n\n===\n\n".join(parts)


async def _seed_relation_graph_for_book(
    campaign, book: str, lore_store: LoreStore, graph_store: RelationGraphStore,
) -> None:
    """Canon seed for Stage 1.5's incremental relation graph — all derived
    from data already generated by extract_entities.py/this same world-prep
    pass, zero extra LLM calls:
      - Location<->Location, from scripts/extract_entities.py's _entities.json
        "_connections" (the same data LocationExtractor.generate_connections()
        produced offline).
      - NPC->Location, from each live NPC's own .location field (already
        loaded, no extra query).
      - Item->Location / NPC->Item, from the Lore Registry's item profiles'
        found_at/owned_by fields (no live Item object needs to exist yet —
        world-prep doesn't create Items, only NPCs/Locations)."""
    entities_path = Path(f"docs/source/adventures/{book}/_entities.json")
    if entities_path.exists():
        try:
            registry = json.loads(entities_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = {}
        for conn in registry.get("_connections", []):
            src = find_location(campaign, conn.get("from", ""))
            dst = find_location(campaign, conn.get("to", ""))
            if src and dst:
                await graph_store.add_edge(
                    campaign.id, "location", src.id, src.name,
                    "location", dst.id, dst.name, "connected to",
                    description=conn.get("via", ""),
                )

    for npc in campaign.npcs:
        if npc.location:
            loc = find_location(campaign, npc.location)
            if loc:
                await graph_store.add_edge(
                    campaign.id, "npc", npc.id, npc.name,
                    "location", loc.id, loc.name, "located in",
                )

    for item_entity in await lore_store.all_for_book(book, "item"):
        profile = item_entity.rolled_up_profile
        found_at = profile.get("found_at", "")
        owned_by = profile.get("owned_by", "")
        if found_at:
            loc = find_location(campaign, found_at)
            if loc:
                await graph_store.add_edge(
                    campaign.id, "item", item_entity.id, item_entity.canonical_name,
                    "location", loc.id, loc.name, "found at",
                    source_chunk_ids=item_entity.source_chunk_ids,
                )
        if owned_by:
            npc = find_npc(campaign, owned_by)
            if npc:
                await graph_store.add_edge(
                    campaign.id, "npc", npc.id, npc.name,
                    "item", item_entity.id, item_entity.canonical_name, "owns",
                    source_chunk_ids=item_entity.source_chunk_ids,
                )


async def _ainvoke_with_retry(agent, payload: dict, config: dict, *, label: str):
    """Retry a one-shot world-prep agent invocation once on an Ollama
    timeout. Confirmed live, 2026-07-08: even with a proper client-level
    httpx timeout configured (see dm_agent.py's _get_model()), the
    underlying MLX engine occasionally hangs on a single request — the
    timeout now fires cleanly (httpx.TimeoutException) instead of freezing
    the whole app, but the request itself still fails. Two live tests in a
    row both failed on the first attempt and would very likely have
    succeeded on a second try, so a bounded retry meaningfully improves the
    success rate rather than just the failure mode. One retry, not a loop —
    same discipline as Stage 2's CRAG grading (backend/rag/grading.py) and
    the live agent's correction_count/tool_error_count caps."""
    import httpx

    try:
        return await agent.ainvoke(payload, config=config)
    except httpx.TimeoutException:
        log.warning("world-prep agent step timed out, retrying once: %s", label)
        return await agent.ainvoke(payload, config=config)


async def run_world_prep(
    campaign_id: str, store: CampaignStore, rules_store: RulesStore,
    lore_store: LoreStore, graph_store: RelationGraphStore,
) -> None:
    """Process each of the campaign's books_in_play one at a time, seeding
    region-scale locations/connections via a bounded, non-interactive agent
    run. Failures are caught and recorded on the campaign rather than
    propagating — this runs as an orphaned asyncio.Task with nothing awaiting
    it, so an uncaught exception here would otherwise just vanish silently.
    """
    # Local imports: avoids a cycle with dm_agent's own imports.
    from backend.agent.dm_agent import get_npc_prep_agent, get_world_prep_agent

    campaign = await store.load(campaign_id)
    if campaign is None or not campaign.books_in_play:
        return

    campaign.world_prep_status = WorldPrepStatus.IN_PROGRESS
    await store.save(campaign)

    try:
        first_book = campaign.books_in_play[0]
        for book in campaign.books_in_play:
            # Deterministic, zero-Ollama-call seed context — see
            # _gather_seed_context's own docstring for why this replaced an
            # earlier embedding-search version (2026-07-08 root cause: the
            # embed-then-chat model swap it forced was the actual trigger
            # for a recurring whole-app freeze, not just a slow call).
            seed_context = _gather_seed_context(book)

            # Re-load: an earlier book's create_location/connect_locations
            # calls have already saved, so the agent sees prior progress.
            campaign = await store.load(campaign_id)
            agent = get_world_prep_agent(campaign, store, rules_store, books_in_play=[book], lore_store=lore_store)
            prompt = get_world_prep_prompt(campaign, book, seed_context)
            await _ainvoke_with_retry(
                agent, {"messages": [HumanMessage(content=prompt)]},
                {"recursion_limit": 60}, label=f"world-prep ({book})",
            )

            # Opening-scene NPC/site-detail seeding — "the opening scene" is
            # inherently a single-book concept (first_book only), and
            # current_location_id already being set means either this phase
            # already ran, or live play has moved the party on — either way,
            # re-running would silently teleport an in-progress party back to
            # session 1's starting room. POST /campaigns/{id}/books re-fires
            # run_world_prep over ALL books_in_play whenever a book is added
            # mid-campaign, so this guard is load-bearing, not defensive
            # boilerplate.
            campaign = await store.load(campaign_id)
            if book == first_book and not campaign.current_location_id:
                meta = read_adventure_meta(book)
                opening_location = meta.get("opening_location", "")
                # asyncio.to_thread + asyncio.wait_for — same reasoning as
                # the _SEED_QUERIES loop above (this function makes a
                # blocking Chroma/Ollama call via search_adventure_literal).
                # On timeout, npc_context stays "" — the function's own
                # contract already treats an empty result as "skip this
                # phase, not an error" (see its docstring), so this reuses
                # an existing graceful-degradation path rather than a new one.
                try:
                    npc_context = await asyncio.wait_for(
                        asyncio.to_thread(
                            _gather_opening_scene_context,
                            rules_store, book, opening_location,
                            meta.get("opening_section_marker", ""), meta.get("opening_hook", ""),
                            meta.get("opening_section_end_marker", ""),
                        ),
                        timeout=30.0,
                    )
                except asyncio.TimeoutError:
                    log.warning("opening-scene context gathering timed out after 30s, skipping (book=%r)", book)
                    npc_context = ""
                if npc_context:
                    existing_npc_names = {n.name for n in campaign.npcs}

                    # Two SEPARATE agent runs, not one combined sequence.
                    # Verified live: asking one agent to create a whole
                    # roster (10+ names, each needing real personality_traits/
                    # motivations grounding) AND THEN call
                    # set_opening_location_detail let the location call get
                    # starved — the agent ran out of budget partway through
                    # the roster and never reached it. Splitting means the
                    # location call always gets its own dedicated budget,
                    # independent of roster size. Both are one-shot
                    # background-task calls with no user waiting on them, so
                    # the extra round-trip costs nothing but a little time.
                    npc_agent = get_npc_prep_agent(campaign, store, rules_store, books_in_play=[book], lore_store=lore_store)
                    npc_prompt = get_npc_prep_prompt(campaign, book, opening_location, npc_context)
                    await _ainvoke_with_retry(
                        npc_agent, {"messages": [HumanMessage(content=npc_prompt)]},
                        {"recursion_limit": 120}, label=f"npc-prep ({book})",
                    )

                    campaign = await store.load(campaign_id)
                    new_npc_names = [n.name for n in campaign.npcs if n.name not in existing_npc_names]
                    location_agent = get_npc_prep_agent(campaign, store, rules_store, books_in_play=[book], lore_store=lore_store)
                    location_prompt = get_opening_location_prompt(
                        campaign, book, opening_location, npc_context, new_npc_names
                    )
                    await _ainvoke_with_retry(
                        location_agent, {"messages": [HumanMessage(content=location_prompt)]},
                        {"recursion_limit": 60}, label=f"opening-location ({book})",
                    )

            campaign = await store.load(campaign_id)
            await _seed_relation_graph_for_book(campaign, book, lore_store, graph_store)

        campaign = await store.load(campaign_id)
        campaign.world_prep_status = WorldPrepStatus.COMPLETE
        await store.save(campaign)
    except Exception as e:
        log.exception("World-prep failed for campaign %s", campaign_id)
        campaign = await store.load(campaign_id)
        if campaign is not None:
            campaign.world_prep_status = WorldPrepStatus.FAILED
            # type(e).__name__ prefix, not bare str(e) — confirmed live,
            # 2026-07-08: httpx's timeout exceptions (ReadTimeout etc.) carry
            # no message text, so str(e) alone silently produces "" and the
            # UI's failure banner shows nothing useful. The class name alone
            # (e.g. "ReadTimeout") is still informative even when the
            # instance has no message.
            campaign.world_prep_error = f"{type(e).__name__}: {e}"[:2000]
            await store.save(campaign)
