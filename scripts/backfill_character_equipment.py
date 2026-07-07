#!/usr/bin/env python3
"""
backfill_character_equipment.py — one-off fix for characters created before
starting equipment/gold/attacks existed in chargen (see derive_level1_stats
in backend/tools/_helpers.py).

For every character in every campaign that's untouched since creation (empty
attacks, empty inventory, 0 gold), derives a class-appropriate starting kit
and writes attacks/inventory/currency/ac. Characters with ANY existing gear
or gold are skipped — that means either a manually-fixed character or one
already played with (bought/sold something), and must not be clobbered.
Safe to run more than once; the "untouched" check makes it idempotent.

Usage:
    python backfill_character_equipment.py             # apply the backfill
    python backfill_character_equipment.py --dry-run    # report only, no writes

Run inside the app container via:  docker compose exec app python backfill_character_equipment.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as `python scripts/backfill_character_equipment.py` from
# anywhere — Python sets sys.path[0] to this script's own directory, not the
# repo root, so `backend` wouldn't otherwise be importable after this script
# moved out of the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import derive_level1_stats


def _untouched(char) -> bool:
    return not char.attacks and not char.inventory and char.currency.to_gp() == 0


async def main(dry_run: bool) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)

    total_backfilled = 0
    total_skipped = 0

    for summary in await store.list_all():
        campaign = await store.load(summary.id)
        if not campaign or not campaign.party:
            continue

        changed = False
        for char in campaign.party:
            if not _untouched(char):
                total_skipped += 1
                continue

            derived = derive_level1_stats(char.ability_scores, char.char_class, char.skill_proficiencies)
            print(
                f"  [{campaign.name}] {char.name} ({char.char_class}): "
                f"AC {char.ac} -> {derived['ac']}, "
                f"{len(derived['attacks'])} attack(s), {len(derived['inventory'])} item(s), "
                f"{derived['currency'].gp} gp"
            )
            if not dry_run:
                char.attacks = derived["attacks"]
                char.inventory = derived["inventory"]
                char.currency = derived["currency"]
                char.ac = derived["ac"]
                changed = True
            total_backfilled += 1

        if changed:
            await store.save(campaign)

    verb = "Would backfill" if dry_run else "Backfilled"
    print(f"\n{verb} {total_backfilled} character(s); skipped {total_skipped} (already had gear/gold).")
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
