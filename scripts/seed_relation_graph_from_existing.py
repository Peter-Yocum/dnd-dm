#!/usr/bin/env python3
"""
seed_relation_graph_from_existing.py — seed a campaign's incremental relation
graph (entity_relations) from data that already exists in Postgres: each
NPC's own .location field, each Location's .connections, and each item's
holder (Character.inventory / NPC.inventory / Container.contents implies
"owns"/"carries"). Pure Python, ZERO LLM calls — for any campaign whose
NPCs/Locations/Items predate Stage 1.5, so lookup_entity/get_related_entities
have something to traverse without waiting for the next session-end or a new
world-prep run.

Idempotent — entity_relations' UniqueConstraint on
(campaign_id, source_id, target_id, relation) makes re-adding the same edge
a harmless no-op, so this is safe to re-run any time.

Usage:
    python seed_relation_graph_from_existing.py --campaign-id <uuid>
    python seed_relation_graph_from_existing.py --campaign-id <uuid> --dry-run

Run inside the app container via:  make seed-relation-graph campaign_id=<uuid>
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.stores.campaign_store import CampaignStore
from backend.stores.graph_store import PostgresRelationGraphStore
from backend.tools._helpers import find_location


async def _seed(campaign, graph_store: PostgresRelationGraphStore, dry_run: bool) -> int:
    count = 0

    for npc in campaign.npcs:
        if not npc.location:
            continue
        loc = find_location(campaign, npc.location)
        if not loc:
            continue
        print(f"  [npc] {npc.name} -> located in -> {loc.name}")
        count += 1
        if not dry_run:
            await graph_store.add_edge(
                campaign.id, "npc", npc.id, npc.name, "location", loc.id, loc.name, "located in",
            )

    for loc in campaign.locations:
        for conn in loc.connections:
            print(f"  [location] {loc.name} -> connected to -> {conn.to_location_name}")
            count += 1
            if not dry_run:
                await graph_store.add_edge(
                    campaign.id, "location", loc.id, loc.name,
                    "location", conn.to_location_id, conn.to_location_name, "connected to",
                    description=conn.notes,
                )

    for char in campaign.party:
        for item in char.inventory:
            print(f"  [character] {char.name} -> owns -> {item.name}")
            count += 1
            if not dry_run:
                await graph_store.add_edge(
                    campaign.id, "character", char.id, char.name, "item", item.id, item.name, "owns",
                )

    for npc in campaign.npcs:
        for item in npc.inventory:
            print(f"  [npc] {npc.name} -> owns -> {item.name}")
            count += 1
            if not dry_run:
                await graph_store.add_edge(
                    campaign.id, "npc", npc.id, npc.name, "item", item.id, item.name, "owns",
                )

    for container in campaign.containers:
        for item in container.contents:
            print(f"  [container] {container.name} -> contains -> {item.name}")
            count += 1
            if not dry_run:
                await graph_store.add_edge(
                    campaign.id, "container", container.id, container.name, "item", item.id, item.name, "contains",
                )

    return count


async def main(campaign_id: str, dry_run: bool) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)
    graph_store = PostgresRelationGraphStore(engine)

    campaign = await store.load(campaign_id)
    if not campaign:
        print(f"No campaign found with id {campaign_id}")
        await engine.dispose()
        return

    print(f"── {campaign.name} ({campaign.id}) ──")
    count = await _seed(campaign, graph_store, dry_run)

    verb = "Would seed" if dry_run else "Seeded"
    print(f"\n{verb} {count} edge(s).")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True, help="Campaign to seed.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be added without writing.")
    args = parser.parse_args()
    asyncio.run(main(args.campaign_id, args.dry_run))
