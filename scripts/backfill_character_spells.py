#!/usr/bin/env python3
"""
backfill_character_spells.py — one-off fix for characters created before
interactive spell selection existed (see backend/data/spells.py,
build_spells_known/derive_spellcasting_stats in backend/tools/_helpers.py).

For every spellcasting-class character in every campaign that's untouched
since creation (empty spells_known AND spellcasting_ability is None), computes
spellcasting_ability/spell_save_dc/spell_attack_bonus and auto-picks the first
N names from each tier of that class's SPELL_MENUS, where N is the class's
SPELL_REQUIREMENTS count. This is only a defensible default because
SPELL_MENUS is authored with each class's PHB-recommended starter spells
listed first — "first N in menu order" is "the PHB's own suggested starter
loadout," not an arbitrary pick (see spells.py's module docstring). These are
freely re-editable later — no re-preparation/swap tool exists yet, but direct
data edits work.

Non-caster classes (not in SPELL_REQUIREMENTS) are skipped before the
untouched check, with an explicit print line, so they're visibly accounted
for in --dry-run output rather than silently absent. Characters with ANY
existing spells or a set spellcasting_ability are skipped — that means either
a manually-fixed character or one already played with, and must not be
clobbered. Safe to run more than once; the "untouched" check makes it
idempotent.

Usage:
    python backfill_character_spells.py             # apply the backfill
    python backfill_character_spells.py --dry-run    # report only, no writes

Run inside the app container via:  docker compose exec app python backfill_character_spells.py
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Allow running as `python scripts/backfill_character_spells.py` from
# anywhere — Python sets sys.path[0] to this script's own directory, not the
# repo root, so `backend` wouldn't otherwise be importable after this script
# moved out of the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.data.spells import SPELL_MENUS, SPELL_REQUIREMENTS
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import build_spells_known, derive_spellcasting_stats


def _untouched(char) -> bool:
    return not char.spells_known and char.spellcasting_ability is None


async def main(dry_run: bool) -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)

    total_backfilled = 0
    total_skipped_noncaster = 0
    total_skipped_touched = 0

    for summary in await store.list_all():
        campaign = await store.load(summary.id)
        if not campaign or not campaign.party:
            continue

        changed = False
        for char in campaign.party:
            if char.char_class not in SPELL_REQUIREMENTS:
                print(f"  [{campaign.name}] {char.name} ({char.char_class}): not a spellcasting class, skipped")
                total_skipped_noncaster += 1
                continue

            if not _untouched(char):
                total_skipped_touched += 1
                continue

            menu = SPELL_MENUS[char.char_class]
            chosen = [name for tier in sorted(menu) for name in menu[tier][:SPELL_REQUIREMENTS[char.char_class].get(tier, 0)]]
            spell_objs, spells_prepared, err = build_spells_known(char.char_class, chosen)
            if err:
                print(f"  [{campaign.name}] {char.name} ({char.char_class}): backfill selection itself failed ({err}) — skipped, needs manual fix")
                continue

            spell_stats = derive_spellcasting_stats(char.ability_scores, char.char_class, char.proficiency_bonus)
            print(
                f"  [{campaign.name}] {char.name} ({char.char_class}): "
                f"spellcasting_ability={spell_stats['spellcasting_ability']}, "
                f"spell_save_dc={spell_stats['spell_save_dc']}, spell_attack_bonus={spell_stats['spell_attack_bonus']}, "
                f"spells={', '.join(chosen)}"
            )
            if not dry_run:
                char.spellcasting_ability = spell_stats["spellcasting_ability"]
                char.spell_save_dc = spell_stats["spell_save_dc"]
                char.spell_attack_bonus = spell_stats["spell_attack_bonus"]
                char.spells_known = spell_objs
                char.spells_prepared = spells_prepared
                changed = True
            total_backfilled += 1

        if changed:
            await store.save(campaign)

    verb = "Would backfill" if dry_run else "Backfilled"
    print(
        f"\n{verb} {total_backfilled} character(s); "
        f"skipped {total_skipped_noncaster} non-caster(s), {total_skipped_touched} already-touched."
    )
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing.")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
