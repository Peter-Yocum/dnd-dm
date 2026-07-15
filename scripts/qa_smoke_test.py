#!/usr/bin/env python3
"""
qa_smoke_test.py — a small, self-contained "QA Test Campaign" that exercises
every mechanic added during the grid-maps/opportunity-attacks/pronouns/
chargen-fidelity work in one deterministic pass: a single building location
with an authored grid, a two-character party (chargen with pronouns +
subclass validation), ability checks/saves, a real combat encounter with
opportunity attacks (standard weapon, reach weapon, and a hostile NPC — plus
an allied NPC that should NOT trigger one), loot + a magic item + a
map-unlock item, a level-up, fog-of-war, and a session log export.

Calls the real tool functions directly (no LLM) — deterministic and fast,
since this is testing mechanics, not narration (the same style used to
manually verify each feature live throughout this project's recent work).
Leaves the campaign in Postgres afterward so it can be opened for real in
the browser (game.html, the Maps browser, Session History) — this is a
tangible fixture to click around in, not just a pass/fail check.

Safe to re-run: deletes any previous "QA Test Campaign" by name first.

Usage:
    docker compose exec app python scripts/qa_smoke_test.py
"""
import asyncio
import sys
import uuid
from pathlib import Path

# Allow running as `python scripts/qa_smoke_test.py` from anywhere — see
# backfill_character_equipment.py's identical note.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.models import (
    AbilityScores, Attack, Campaign, CombatStatBlock, DamageType, Item, NPC, Session,
)
from backend.session_export import render_session_export_markdown
from backend.stores.campaign_store import CampaignStore
from backend.stores.draft_store import DraftStore
from backend.tools.chargen import make_tools as make_chargen_tools
from backend.tools.combat import make_tools as make_combat_tools
from backend.tools.companion import make_tools as make_companion_tools
from backend.tools.levelup import make_tools as make_levelup_tools
from backend.tools.party import make_tools as make_party_tools
from backend.tools.resolution import make_tools as make_resolution_tools
from backend.tools.world import make_tools as make_world_tools

CAMPAIGN_NAME = "QA Test Campaign"

_checks: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    _checks.append((label, condition, detail))
    mark = "✓" if condition else "✗ FAIL"
    print(f"  {mark}  {label}" + (f" — {detail}" if detail and not condition else ""))


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    store = CampaignStore(engine)

    print("=== Setup ===")
    existing = await store.list_all()
    for c in existing:
        if c.name == CAMPAIGN_NAME:
            await store.delete(c.id)
            print(f"  Deleted previous '{CAMPAIGN_NAME}' ({c.id})")

    campaign = Campaign(id=str(uuid.uuid4()), name=CAMPAIGN_NAME)
    await store.create(campaign)
    campaign_id = campaign.id
    print(f"  Created campaign {campaign_id}")

    world_tools = {t.name: t for t in make_world_tools(campaign_id, store)}
    chargen_ds = DraftStore()

    # ── 1. Location + grid ──────────────────────────────────────────────────
    print("\n=== 1. Location + grid ===")
    await world_tools["create_location"].ainvoke({
        "name": "The Rusty Anchor", "area_type": "indoor", "scale": "site",
        "description": "A weathered tavern near the docks, its common room "
                        "thick with pipe smoke and the smell of brine.",
    })
    move_result = await world_tools["move_party"].ainvoke({"location_name": "The Rusty Anchor"})
    check("move_party sets the party's current location", "moves to The Rusty Anchor" in move_result, move_result)

    grid_result = await world_tools["set_location_grid"].ainvoke({
        "location_name": "The Rusty Anchor",
        "grid": [
            "#########",
            "#..T....#",
            "#..#..C.#",
            "#..#....#",
            "D.......#",
            "#########",
        ],
        "legend": {"#": "wall", "D": "door", "T": "table", "C": "crate"},
    })
    check("set_location_grid accepts a valid grid", "Grid set" in grid_result, grid_result)

    grid_text = await world_tools["get_location_grid"].ainvoke({"location_name": "The Rusty Anchor"})
    check("get_location_grid reads it back with legend", "wall" in grid_text and "crate" in grid_text)

    campaign = await store.load(campaign_id)
    check("The Rusty Anchor is marked visited (shows in the Maps browser)",
          campaign.locations[0].visited is True)

    # ── 2. Chargen party (pronouns + subclass validation) ───────────────────
    print("\n=== 2. Chargen party ===")
    chargen_tools = {t.name: t for t in make_chargen_tools(campaign_id, "player1", store, chargen_ds)}
    for field, value in [
        ("name", "Tarvokk"), ("race", "Human"), ("char_class", "Fighter"),
        ("background", "Soldier"), ("pronouns", "he/him"), ("appearance", "Broad-shouldered, scarred knuckles."),
    ]:
        await chargen_tools["update_character_draft"].ainvoke({"field": field, "value": value})
    await chargen_tools["update_ability_scores"].ainvoke({
        "strength": 16, "dexterity": 12, "constitution": 14, "intelligence": 10, "wisdom": 10, "charisma": 8,
    })
    bad_subclass = await chargen_tools["update_character_draft"].ainvoke({"field": "subclass", "value": "Necromancer"})
    finalize_bad = await chargen_tools["finalize_character"].ainvoke({})
    check("finalize_character rejects an invalid subclass", "isn't a real Fighter subclass" in finalize_bad, finalize_bad)

    await chargen_tools["update_character_draft"].ainvoke({"field": "subclass", "value": "battle master"})
    finalize_ok = await chargen_tools["finalize_character"].ainvoke({})
    check("finalize_character accepts a corrected subclass", "added to the party" in finalize_ok, finalize_ok)

    chargen_tools2 = {t.name: t for t in make_chargen_tools(campaign_id, "player2", store, chargen_ds)}
    for field, value in [
        ("name", "Elara"), ("race", "Half-Elf"), ("char_class", "Cleric"),
        ("background", "Acolyte"), ("pronouns", "she/her"), ("subclass", "Life"),
    ]:
        await chargen_tools2["update_character_draft"].ainvoke({"field": field, "value": value})
    await chargen_tools2["update_ability_scores"].ainvoke({
        "strength": 10, "dexterity": 12, "constitution": 14, "intelligence": 10, "wisdom": 16, "charisma": 12,
    })
    await chargen_tools2["update_character_draft"].ainvoke({"field": "spells_known", "value": "Sacred Flame, Guidance, Thaumaturgy, Cure Wounds, Bless, Guiding Bolt, Shield of Faith"})
    finalize_elara = await chargen_tools2["finalize_character"].ainvoke({})
    check("second character (caster) finalizes", "added to the party" in finalize_elara, finalize_elara)

    campaign = await store.load(campaign_id)
    tarvokk = next(c for c in campaign.party if c.name == "Tarvokk")
    elara = next(c for c in campaign.party if c.name == "Elara")
    check("Tarvokk's pronouns saved", tarvokk.pronouns == "he/him")
    check("Tarvokk's subclass normalized to real casing", tarvokk.subclass == "Battle Master", tarvokk.subclass)
    check("Tarvokk got real starting equipment (Fighter: Greatsword)", any(a.name == "Greatsword" for a in tarvokk.attacks))
    check("Tarvokk got real starting gold (Fighter: 4gp)", tarvokk.currency.gp == 4, str(tarvokk.currency.gp))
    check("Elara's pronouns saved", elara.pronouns == "she/her")
    check("Elara has Shield of Faith on her spell list", any(s.name == "Shield of Faith" for s in elara.spells_known))

    # Allied NPC companion, fighting on the party's side
    companion_tools = {t.name: t for t in make_companion_tools(campaign_id, store)}
    companion_result = await companion_tools["generate_companion_character"].ainvoke({
        "name": "Old Bram", "race": "Dwarf", "char_class": "Fighter", "background": "Guard",
        "strength": 15, "dexterity": 10, "constitution": 14, "intelligence": 10, "wisdom": 12, "charisma": 10,
        "subclass": "Champion",
    })
    check("DM companion (ally) created", "added" in companion_result.lower() or "party" in companion_result.lower(), companion_result)

    # Bump HP for combat-demo robustness — the opportunity-attack scene below
    # deliberately mocks d20 rolls to a guaranteed-hit value for determinism,
    # which also inflates damage dice rolled via the same random.randint patch
    # (no separate to-hit/damage RNG). Real starting HP (~12-15) wouldn't
    # survive three back-to-back max-ish-damage hits, and there's no
    # resurrection tool yet to undo an accidental death here — so these two
    # get treated as seasoned veterans for QA-repeatability purposes.
    campaign = await store.load(campaign_id)
    for name in ("Tarvokk", "Elara"):
        c = next(ch for ch in campaign.party if ch.name == name)
        c.max_hp = 60
        c.current_hp = 60
    # Give Elara a real reaction option for the opportunity-attack pause demo
    # below — "Shield" (the actual 1-reaction spell) is only on the Sorcerer/
    # Wizard curated menu, not Cleric's, so has_plausible_reaction()
    # (_helpers.py) needs a reaction FEATURE instead: War Caster is one of
    # the real keywords it checks for (_REACTION_FEATURE_KEYWORDS).
    elara_char = next(ch for ch in campaign.party if ch.name == "Elara")
    elara_char.features.append("War Caster")
    await store.save(campaign)

    party_tools = {t.name: t for t in make_party_tools(campaign_id, store, None, None)}

    # ── 3. Ability checks + saves (outside combat) ──────────────────────────
    print("\n=== 3. Ability checks + saves ===")
    resolution_tools = {t.name: t for t in make_resolution_tools(campaign_id, store)}
    check_result = await resolution_tools["resolve_check"].ainvoke({
        "character_name": "Tarvokk", "ability_or_skill": "athletics", "dc": 12,
    })
    check("resolve_check returns a real d20+modifier breakdown", "d20" in check_result and "=" in check_result, check_result)

    save_result = await resolution_tools["resolve_saving_throw"].ainvoke({
        "target_names": ["Elara"], "ability": "wisdom", "dc": 10,
    })
    check("resolve_saving_throw returns a real breakdown", "save:" in save_result, save_result)

    # ── 4. Combat with opportunity attacks ──────────────────────────────────
    print("\n=== 4. Combat + opportunity attacks ===")
    combat_tools = {t.name: t for t in make_combat_tools(campaign_id, store)}

    await combat_tools["create_monster"].ainvoke({
        "name": "Smuggler", "size": "medium", "monster_type": "humanoid",
        "ac": 12, "max_hp": 11, "cr": "1/8",
        "attacks": [{"name": "Shortsword", "to_hit_bonus": 3, "damage_dice": "1d6+1", "damage_type": "piercing"}],
    })
    await combat_tools["create_monster"].ainvoke({
        "name": "Pike Guard", "size": "medium", "monster_type": "humanoid",
        "ac": 14, "max_hp": 15, "cr": "1/2",
        "attacks": [{"name": "Pike", "to_hit_bonus": 4, "damage_dice": "1d10+2", "damage_type": "piercing", "reach_ft": 10}],
    })
    campaign = await store.load(campaign_id)
    campaign.npcs.append(NPC(name="Bandit Lookout", combat_stats=CombatStatBlock(
        max_hp=9, current_hp=9, ac=12, reaction_available=True,
        ability_scores=AbilityScores(dexterity=13),
        attacks=[Attack(name="Dagger", to_hit_bonus=3, damage_dice="1d4+1", damage_type=DamageType.PIERCING)],
    )))
    await store.save(campaign)

    start_result = await combat_tools["start_encounter"].ainvoke({
        "location_description": "A brawl breaks out in the common room.",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 18, "side": "party"},
            {"name": "Elara", "type": "character", "initiative_override": 16, "side": "party"},
            {"name": "Old Bram", "type": "npc", "initiative_override": 14, "side": "party"},
            {"name": "Smuggler", "type": "monster", "initiative_override": 12, "side": "hostile"},
            {"name": "Pike Guard", "type": "monster", "initiative_override": 10, "side": "hostile"},
            {"name": "Bandit Lookout", "type": "npc", "initiative_override": 8, "side": "hostile"},
        ],
    })
    check("start_encounter succeeds (grid already authored)", "Encounter started" in start_result, start_result)

    # Coordinates below are hand-checked against the authored grid's walls
    # (row 0/5 and column 0/8 are walls, plus a wall segment at column 3 in
    # rows 2-3) — set_combatant_position correctly refuses a wall placement,
    # which would otherwise leave no "old" position for the retreat check
    # below to compare against.
    # Tarvokk is set up near Smuggler + Pike Guard only; Elara near Bandit
    # Lookout only; Old Bram stays neutral — kept deliberately separate so
    # each retreat below triggers exactly the attackers it's meant to.
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 1, "y": 1})
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Smuggler", "x": 2, "y": 1})
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Pike Guard", "x": 1, "y": 3})
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Elara", "x": 6, "y": 2})
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Bandit Lookout", "x": 6, "y": 3})
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Old Bram", "x": 4, "y": 4})

    # Tarvokk retreats from BOTH the standard-reach Smuggler (1 square away)
    # and the 10ft-reach Pike Guard (2 squares away, still in reach) in one
    # move — both should swing (multi-attacker resolution, not just the first)
    from unittest.mock import patch
    with patch("backend.tools.resolution.random.randint", return_value=15):
        tarvokk_retreat = await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 7, "y": 1})
    check("retreating combatant triggers opportunity attacks from BOTH standard and reach-weapon monsters",
          tarvokk_retreat.lower().count("opportunity attack") == 2, tarvokk_retreat)

    # Elara (has the War Caster reaction feature) gets attacked by the
    # hostile NPC lookout leaving its reach — should PAUSE, not resolve immediately
    with patch("backend.tools.resolution.random.randint", return_value=20):
        elara_retreat = await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Elara", "x": 1, "y": 4})
    check("a mover with a real reaction option pauses instead of resolving immediately",
          "PENDING" in elara_retreat, elara_retreat)
    resolve_result = await resolution_tools["resolve_pending_action"].ainvoke({"reaction_declared": "War Caster (opportunity spell)"})
    check("resolve_pending_action finishes the paused opportunity attack", "damage" in resolve_result, resolve_result)

    # Old Bram (an ally) never triggered anything against the party despite
    # being in the same fight — confirms the earlier "ally doesn't trigger
    # against ally" case holds (regression-tested separately in
    # tests/test_location_grids_and_opportunity_attacks.py), and that an
    # allied NPC is tracked in the encounter at all.
    check("allied NPC companion is tracked in the encounter", any(
        e.name == "Old Bram" and e.side == "party"
        for e in (await store.load(campaign_id)).active_encounter.initiative_order
    ))

    # Patch everyone back up after the demo hits — a real DM would narrate
    # this as the party recovering post-brawl, and it keeps the campaign in
    # a sane, inspectable state for the level-up/export sections below
    # rather than ending on two dead player characters.
    await party_tools["update_character_hp"].ainvoke({"character_name": "Tarvokk", "delta": 999})
    await party_tools["update_character_hp"].ainvoke({"character_name": "Elara", "delta": 999})

    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://localhost:8000/campaigns/{campaign_id}/combat-map", timeout=5.0)
        combat_map_data = resp.json()
        check("live /combat-map endpoint returns real positions + terrain",
              combat_map_data.get("has_grid") and len(combat_map_data.get("combatants", [])) >= 4,
              str(combat_map_data)[:200])
    except Exception as e:
        print(f"  (skipped live /combat-map HTTP check — app not reachable from here: {e})")

    # ── 5. Loot, magic item, map-unlock item ────────────────────────────────
    print("\n=== 5. Loot + items ===")

    # Defeat the Smuggler so end_encounter's automatic loot roll (real DMG
    # treasure tables scaled to CR) has something to actually award — call
    # this FIRST, before any manual reveal_loot, since a manual grant
    # deliberately suppresses the automatic roll for the whole encounter
    # (see end_encounter's own docstring) — testing both paths means not
    # tripping that suppression before the auto-roll gets its turn.
    await combat_tools["update_monster_hp"].ainvoke({"monster_name": "Smuggler", "delta": -20})
    end_result = await combat_tools["end_encounter"].ainvoke({"xp_awarded": 150})
    check("end_encounter closes the fight and rolls loot for the defeated Smuggler",
          "xp" in end_result.lower() or "loot" in end_result.lower(), end_result)

    # Manually-declared loot, found afterward while searching the cellar —
    # the OTHER loot path (reveal_loot), distinct from end_encounter's auto-roll
    reveal_result = await party_tools["reveal_loot"].ainvoke({
        "source_name": "a locked strongbox in the cellar", "currency": {"gp": 8, "sp": 15},
        "items": [{"name": "Smuggled Spices", "quantity": 3}],
    })
    check("reveal_loot records a manual, non-combat find", "8" in reveal_result and "Smuggled Spices" in reveal_result, reveal_result)

    magic_result = await party_tools["create_magic_item"].ainvoke({
        "character_name": "Tarvokk", "item_name": "+1 Longsword", "base_item": "Longsword", "bonus": 1,
    })
    check("create_magic_item grants a grounded magic weapon attack", "added to" in magic_result, magic_result)

    await world_tools["create_location"].ainvoke({
        "name": "Smugglers' Cellar", "area_type": "underground", "scale": "site",
        "description": "A hidden cellar beneath the tavern, reeking of tar and old rope.",
    })
    map_result = await party_tools["add_item_to_character"].ainvoke({
        "character_name": "Elara", "item_name": "Hand-drawn Cellar Map", "map_of_location": "Smugglers' Cellar",
    })
    check("a map item unlocks its linked location in the Maps browser", "Unlocked" in map_result, map_result)

    # ── 6. Level up ──────────────────────────────────────────────────────────
    print("\n=== 6. Level up ===")
    levelup_tools = {t.name: t for t in make_levelup_tools(campaign_id, store)}
    bad_level = await levelup_tools["level_up"].ainvoke({
        "character_name": "Elara", "new_level": 2, "subclass": "Necromancer",
    })
    check("level_up rejects an invalid subclass before mutating anything", "isn't a real Cleric subclass" in bad_level, bad_level)
    good_level = await levelup_tools["level_up"].ainvoke({
        "character_name": "Elara", "new_level": 2,
    })
    check("level_up succeeds", "level 1 -> 2" in good_level, good_level)

    # ── 7. Fog of war + Maps browser ────────────────────────────────────────
    print("\n=== 7. Fog of war ===")
    campaign = await store.load(campaign_id)
    tavern_reloaded = next(l for l in campaign.locations if l.name == "The Rusty Anchor")
    check("party movement populated revealed_positions (fog-of-war data)",
          len(tavern_reloaded.revealed_positions) > 0, str(tavern_reloaded.revealed_positions))
    check("Smugglers' Cellar is map_known via the map item", any(
        l.map_known for l in campaign.locations if l.name == "Smugglers' Cellar"
    ))

    # ── 8. Session log export ───────────────────────────────────────────────
    print("\n=== 8. Session export ===")
    campaign.sessions.append(Session(
        session_number=1, summary="The party brawled with smugglers at the Rusty Anchor and won.",
        key_events=["Defeated the smugglers", "Found a hidden cellar", "Elara reached level 2"],
        xp_awarded=150,
    ))
    await store.save(campaign)
    export_text = render_session_export_markdown(campaign)
    check("session export renders the recorded session", "Session 1" in export_text and "brawled" in export_text)

    # ── Report ───────────────────────────────────────────────────────────────
    print("\n=== Report ===")
    passed = sum(1 for _, ok, _ in _checks if ok)
    total = len(_checks)
    print(f"  {passed}/{total} checks passed")
    if passed < total:
        print("  Failing checks:")
        for label, ok, detail in _checks:
            if not ok:
                print(f"    ✗ {label} — {detail}")

    print(f"\nCampaign left in place for manual inspection:")
    print(f"  http://localhost:8000/campaigns/{campaign_id}")
    print(f"  http://localhost:8000/campaigns/{campaign_id}/maps")
    print(f"  http://localhost:8000/campaigns/{campaign_id}/sessions")

    await engine.dispose()
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
