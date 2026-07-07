#!/usr/bin/env python3
"""
backfill_character_saving_throws.py — one-off fix for characters created
before saving_throw_proficiencies was ever populated by chargen (see
derive_saving_throw_proficiencies in backend/tools/_helpers.py).

Character.saving_throw_proficiencies has existed on the model since it was
added for resolve_saving_throw's proficiency-bonus lookup (_save_bonus in
resolution.py), but nothing in chargen.py/companion.py ever set it — every
character in every campaign has been rolling saving throws with no
proficiency bonus applied, regardless of class. This is a straight lookup
from the class's two listed saves (fivee_options.CLASSES[...]["saving_throws"]),
not a per-character choice like skills/spells, so there's no ambiguity to
guess around.

Untouched check is an empty saving_throw_proficiencies set — any character
with something already set (e.g. a manual fix) is left alone. Safe to run
more than once; idempotent.

Usage:
    python backfill_character_saving_throws.py             # apply the backfill
    python backfill_character_saving_throws.py --dry-run    # report only, no writes

Run inside the app container via:  docker compose exec app python backfill_character_saving_throws.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as `python scripts/backfill_character_saving_throws.py` from
# anywhere — Python sets sys.path[0] to this script's own directory, not the
# repo root, so `backend` wouldn't otherwise be importable after this script
# moved out of the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import derive_saving_throw_proficiencies


def _untouched(char) -> bool:
    return not char.saving_throw_proficiencies


async def main(dry_run: bool) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)

    total_backfilled = 0
    total_skipped_touched = 0
    total_skipped_unknown_class = 0

    for summary in await store.list_all():
        campaign = await store.load(summary.id)
        if not campaign or not campaign.party:
            continue

        changed = False
        for char in campaign.party:
            if not _untouched(char):
                total_skipped_touched += 1
                continue

            profs = derive_saving_throw_proficiencies(char.char_class)
            if not profs:
                print(f"  [{campaign.name}] {char.name} ({char.char_class}): unrecognized class, skipped")
                total_skipped_unknown_class += 1
                continue

            print(f"  [{campaign.name}] {char.name} ({char.char_class}): saving_throw_proficiencies={sorted(profs)}")
            if not dry_run:
                char.saving_throw_proficiencies = profs
                changed = True
            total_backfilled += 1

        if changed:
            await store.save(campaign)

    verb = "Would backfill" if dry_run else "Backfilled"
    print(
        f"\n{verb} {total_backfilled} character(s); "
        f"skipped {total_skipped_touched} already-touched, {total_skipped_unknown_class} unrecognized class."
    )
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
