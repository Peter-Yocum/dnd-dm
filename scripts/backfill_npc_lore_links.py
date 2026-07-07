#!/usr/bin/env python3
"""
backfill_npc_lore_links.py — link existing NPCs/Locations/Items (created
before the Lore Registry existed, e.g. by the old world-prep pipeline) to
their canon Lore Registry counterpart, so lookup_entity/search_lore and the
relation graph can find them by provenance instead of re-deriving everything.

For every campaign, every NPC/Location, and every Item (flattened from
Character.inventory, NPC.inventory, Container.contents) with lore_entity_id
still None, fuzzy-matches its name against that campaign's books_in_play in
the Lore Registry (core books always included, same scope as
RulesStore.search()). On a match, sets lore_entity_id and MERGES IN (union,
never overwrites) aliases/source_chunk_ids, and sets spoiler_tier only if
the live record still has the default "public" (never downgrades an
already-customized tier). Never touches play-diverged fields (attitude,
notes, is_alive, quantity, attuned_to, etc.).

Idempotent — only touches records with lore_entity_id IS NULL, safe to
re-run (0 backfilled, all skipped on a second pass).

Usage:
    python backfill_npc_lore_links.py                          # all campaigns
    python backfill_npc_lore_links.py --dry-run                # report only, no writes
    python backfill_npc_lore_links.py --campaign-id <uuid>      # one campaign only

Run inside the app container via:  make backfill-lore-links campaign_id=<uuid> [dry_run=1]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.rag.entity_resolution import find_candidate_matches
from backend.stores.campaign_store import CampaignStore
from backend.stores.lore_store import LoreEntity, LoreStore

# Higher bar than the interactive create_npc/create_location dedup warning
# (0.80) — this backfill sets lore_entity_id with no human in the loop by
# default (only --dry-run offers review before committing), so a false
# positive link is worse here than an interactive warning a model can
# dismiss with force=True.
BACKFILL_MATCH_THRESHOLD = 0.85


async def _match(lore_store: LoreStore, book_slugs: list[str], name: str, entity_type: str) -> LoreEntity | None:
    exact = await lore_store.find_by_name_or_alias(book_slugs, name, entity_type=entity_type)
    if exact:
        return exact
    candidates = await lore_store.find_candidates(book_slugs, entity_type)
    matches = find_candidate_matches(name, candidates, threshold=BACKFILL_MATCH_THRESHOLD)
    if not matches:
        return None
    return await lore_store.find_by_name_or_alias(book_slugs, matches[0], entity_type=entity_type)


def _apply(live_obj, entity: LoreEntity, kind: str, label: str, dry_run: bool) -> bool:
    print(f"    [{kind}] {label} -> '{entity.canonical_name}' (from '{entity.book_slug}')")
    if dry_run:
        return True
    live_obj.lore_entity_id = entity.id
    live_obj.aliases = list(dict.fromkeys(live_obj.aliases + entity.aliases))
    live_obj.source_chunk_ids = list(dict.fromkeys(live_obj.source_chunk_ids + entity.source_chunk_ids))
    if live_obj.spoiler_tier == "public" and entity.spoiler_tier != "public":
        live_obj.spoiler_tier = entity.spoiler_tier
    return True


async def _backfill_campaign(lore_store: LoreStore, campaign, dry_run: bool) -> tuple[int, int]:
    book_slugs = list(campaign.books_in_play)
    linked = 0
    skipped = 0

    for npc in campaign.npcs:
        if npc.lore_entity_id:
            skipped += 1
            continue
        entity = await _match(lore_store, book_slugs, npc.name, "npc")
        if entity:
            linked += _apply(npc, entity, "npc", npc.name, dry_run)
        else:
            skipped += 1

    for loc in campaign.locations:
        if loc.lore_entity_id:
            skipped += 1
            continue
        entity = await _match(lore_store, book_slugs, loc.name, "location")
        if entity:
            linked += _apply(loc, entity, "location", loc.name, dry_run)
        else:
            skipped += 1

    item_holders = (
        [(item, f"{c.name}'s inventory") for c in campaign.party for item in c.inventory]
        + [(item, f"{n.name}'s inventory") for n in campaign.npcs for item in n.inventory]
        + [(item, f"container '{c.name}'") for c in campaign.containers for item in c.contents]
    )
    for item, holder in item_holders:
        if item.lore_entity_id:
            skipped += 1
            continue
        entity = await _match(lore_store, book_slugs, item.name, "item")
        if entity:
            linked += _apply(item, entity, "item", f"{item.name} ({holder})", dry_run)
        else:
            skipped += 1

    return linked, skipped


async def main(dry_run: bool, campaign_id: str | None) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)
    lore_store = LoreStore(engine)

    total_linked = 0
    total_skipped = 0

    summaries = [s for s in await store.list_all() if not campaign_id or s.id == campaign_id]
    for summary in summaries:
        campaign = await store.load(summary.id)
        if not campaign:
            continue
        print(f"── {campaign.name} ({campaign.id}) ──")
        linked, skipped = await _backfill_campaign(lore_store, campaign, dry_run)
        total_linked += linked
        total_skipped += skipped
        if linked and not dry_run:
            await store.save(campaign)

    verb = "Would link" if dry_run else "Linked"
    print(f"\n{verb} {total_linked} record(s); skipped {total_skipped} (already linked or no canon match).")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    parser.add_argument("--campaign-id", default=None, help="Only backfill this one campaign.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run, args.campaign_id))
