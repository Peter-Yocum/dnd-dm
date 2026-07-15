"""BDD-style coverage for real grid maps + coordinate positions + opportunity
attacks (design.md's mapping-cluster idea #1, and the "opportunity attacks"
item in the "Deferred from combat resolution refactor" list) — built to
replace a rejected zone-transition heuristic with a real distance check.
See backend/tools/world.py (set_location_grid/get_location_grid),
backend/tools/combat.py (start_encounter's grid requirement,
set_combatant_position's x/y), backend/tools/_helpers.py
(check_opportunity_attacks/apply_map_reveal_if_needed), and
backend/tools/resolution.py (resolve_opportunity_attack).
"""
import pytest

from backend.models import (
    Attack, AbilityScores, Character, CombatStatBlock, DamageType, Location, Monster, NPC,
)
from backend.tools.combat import make_tools as make_combat_tools
from backend.tools.party import make_tools as make_party_tools
from backend.tools.resolution import make_tools as make_resolution_tools
from backend.tools.world import make_authoring_tools, make_movement_tools

pytestmark = pytest.mark.asyncio


def _grid_location(**overrides) -> Location:
    defaults = dict(
        name="Forest Clearing", scale="region",
        grid=[".....", ".T...", "....."], legend={"T": "tree"},
    )
    defaults.update(overrides)
    return Location(**defaults)


# ── set_location_grid validation ─────────────────────────────────────────────

async def test_set_location_grid_rejects_a_non_rectangular_grid(store, campaign):
    campaign.locations = [Location(name="Goblin Den", scale="site")]
    await store.save(campaign)
    tools = {t.name: t for t in make_authoring_tools(campaign.id, store)}

    result = await tools["set_location_grid"].ainvoke({
        "location_name": "Goblin Den", "grid": ["##", "."], "legend": {"#": "wall"},
    })
    assert "same length" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.locations[0].grid == []


async def test_set_location_grid_rejects_an_unlegended_symbol(store, campaign):
    campaign.locations = [Location(name="Goblin Den", scale="site")]
    await store.save(campaign)
    tools = {t.name: t for t in make_authoring_tools(campaign.id, store)}

    result = await tools["set_location_grid"].ainvoke({
        "location_name": "Goblin Den", "grid": ["#.T"], "legend": {"#": "wall"},
    })
    assert "no legend entry" in result
    assert "T" in result


async def test_set_location_grid_accepts_a_valid_grid_and_get_location_grid_reads_it_back(store, campaign):
    campaign.locations = [Location(name="Goblin Den", scale="site")]
    await store.save(campaign)
    tools = {t.name: t for t in make_authoring_tools(campaign.id, store)}

    result = await tools["set_location_grid"].ainvoke({
        "location_name": "Goblin Den", "grid": ["#..", ".T.", "..#"], "legend": {"#": "wall", "T": "tree"},
    })
    assert "3x3" in result

    grid_text = await tools["get_location_grid"].ainvoke({"location_name": "Goblin Den"})
    assert "wall" in grid_text
    assert "tree" in grid_text


# ── start_encounter's grid requirement ───────────────────────────────────────

async def test_start_encounter_refuses_without_a_grid(store, campaign):
    loc = Location(name="Forest Clearing", scale="region")  # no grid
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15)]
    campaign.monsters = [Monster(name="Goblin 1", current_hp=7, max_hp=7, ac=13)]
    await store.save(campaign)

    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    result = await tools["start_encounter"].ainvoke({
        "location_description": "a clearing",
        "combatants": [{"name": "Tarvokk", "type": "character"}, {"name": "Goblin 1", "type": "monster"}],
    })
    assert "no grid map yet" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.active_encounter is None


async def test_start_encounter_succeeds_once_a_grid_is_authored(store, campaign):
    loc = _grid_location()
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15)]
    campaign.monsters = [Monster(name="Goblin 1", current_hp=7, max_hp=7, ac=13)]
    await store.save(campaign)

    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    result = await tools["start_encounter"].ainvoke({
        "location_description": "a clearing",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Goblin 1", "type": "monster", "initiative_override": 10},
        ],
    })
    assert "Encounter started" in result


# ── set_combatant_position with real coordinates ─────────────────────────────

async def _started_encounter(store, campaign):
    loc = _grid_location()
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.monsters = [Monster(
        name="Goblin 1", current_hp=7, max_hp=7, ac=13, reaction_available=True,
        attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING)],
        ability_scores=AbilityScores(dexterity=14),
    )]
    await store.save(campaign)
    combat_tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    await combat_tools["start_encounter"].ainvoke({
        "location_description": "a clearing",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Goblin 1", "type": "monster", "initiative_override": 10},
        ],
    })
    return combat_tools


async def test_set_combatant_position_populates_coordinates(store, campaign):
    tools = await _started_encounter(store, campaign)
    result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    assert "(2, 1)" in result
    reloaded = await store.load(campaign.id)
    pos = next(p for p in reloaded.active_encounter.combatant_positions if p.name == "Tarvokk")
    assert pos.coordinates == (2, 1)


async def test_set_combatant_position_refuses_out_of_bounds(store, campaign):
    tools = await _started_encounter(store, campaign)
    result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 99, "y": 0})
    assert "out of bounds" in result


async def test_set_combatant_position_refuses_a_wall_cell(store, campaign):
    tools = await _started_encounter(store, campaign)
    # Re-author the grid to add a wall — reload first: the tool calls above
    # each did their own store.load/save round-trip, so the outer `campaign`
    # fixture object is stale relative to what's actually persisted now.
    reloaded = await store.load(campaign.id)
    reloaded.locations[0].grid = ["#....", ".T...", "....."]
    reloaded.locations[0].legend = {"T": "tree", "#": "wall"}
    await store.save(reloaded)
    result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    assert "wall" in result


async def test_set_combatant_position_refuses_coordinates_with_no_grid(store, campaign):
    loc = Location(name="Open Field", scale="region")  # no grid
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    enc_tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    from backend.models import Encounter
    campaign.active_encounter = Encounter(is_active=True, round=1)
    await store.save(campaign)
    result = await enc_tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    assert "no grid" in result


# ── opportunity attacks: real distance check ─────────────────────────────────

async def test_a_move_staying_within_reach_does_not_trigger_an_opportunity_attack(store, campaign, force_hit):
    tools = await _started_encounter(store, campaign)
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 1", "x": 2, "y": 2})
    # Move to a cell still within 1 square of the goblin
    result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 3, "y": 2})
    assert "opportunity attack" not in result.lower()
    reloaded = await store.load(campaign.id)
    assert reloaded.active_encounter.pending_action is None


async def test_leaving_reach_triggers_a_real_opportunity_attack(store, campaign):
    from unittest.mock import patch
    tools = await _started_encounter(store, campaign)
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 1", "x": 2, "y": 2})
    with patch("backend.tools.resolution.random.randint", return_value=20):
        result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    assert "opportunity attack" in result.lower()
    assert "HIT" in result
    reloaded = await store.load(campaign.id)
    tarvokk = next(c for c in reloaded.party if c.name == "Tarvokk")
    # Tarvokk has no reaction spell/feature, so this resolves immediately —
    # damage applied, not paused.
    assert tarvokk.current_hp < 20
    assert reloaded.active_encounter.pending_action is None


async def test_opportunity_attack_pauses_if_the_mover_has_a_real_reaction_option(store, campaign):
    from unittest.mock import patch
    from backend.models import Spell, SpellSlotLevel
    loc = _grid_location()
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(
        name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True,
        spells_known=[Spell(name="Shield", level=1, casting_time="1 reaction", effect_dice="", is_healing=False)],
        spells_prepared=["Shield"], spell_slots={1: SpellSlotLevel(max=2, used=0)},
    )]
    campaign.monsters = [Monster(
        name="Goblin 1", current_hp=7, max_hp=7, ac=13, reaction_available=True,
        attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING)],
        ability_scores=AbilityScores(dexterity=14),
    )]
    await store.save(campaign)
    combat_tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    resolution_tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    await combat_tools["start_encounter"].ainvoke({
        "location_description": "x",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Goblin 1", "type": "monster", "initiative_override": 10},
        ],
    })
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 1", "x": 2, "y": 2})

    with patch("backend.tools.resolution.random.randint", return_value=20):
        result = await combat_tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    assert "PENDING" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.active_encounter.pending_action.trigger_type == "movement_away"
    tarvokk = next(c for c in reloaded.party if c.name == "Tarvokk")
    assert tarvokk.current_hp == 20  # paused, not yet applied

    resolve_result = await resolution_tools["resolve_pending_action"].ainvoke({"reaction_declared": "Shield"})
    assert "damage" in resolve_result and "Shield" in resolve_result
    reloaded2 = await store.load(campaign.id)
    tarvokk2 = next(c for c in reloaded2.party if c.name == "Tarvokk")
    assert tarvokk2.current_hp < 20


# ── map-item acquisition unlocks a location ───────────────────────────────────

async def test_add_item_to_character_with_map_of_location_unlocks_it(store, campaign):
    loc = Location(name="The Sewers", scale="site")
    campaign.locations = [loc]
    campaign.party = [Character(name="Kaelen")]
    await store.save(campaign)

    tools = {t.name: t for t in make_party_tools(campaign.id, store, None, None)}
    result = await tools["add_item_to_character"].ainvoke({
        "character_name": "Kaelen", "item_name": "Map of the Sewers", "map_of_location": "The Sewers",
    })
    assert "Unlocked" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.locations[0].map_known is True


async def test_add_item_to_character_without_map_of_location_does_not_unlock_anything(store, campaign):
    loc = Location(name="The Sewers", scale="site")
    campaign.locations = [loc]
    campaign.party = [Character(name="Kaelen")]
    await store.save(campaign)

    tools = {t.name: t for t in make_party_tools(campaign.id, store, None, None)}
    await tools["add_item_to_character"].ainvoke({"character_name": "Kaelen", "item_name": "Rope (50ft)"})
    reloaded = await store.load(campaign.id)
    assert reloaded.locations[0].map_known is False


# ── automatic visited-tracking ─────────────────────────────────────────────

async def test_move_party_marks_the_destination_visited(store, campaign):
    loc = Location(name="Phandalin", scale="region")
    campaign.locations = [loc]
    await store.save(campaign)

    tools = {t.name: t for t in make_movement_tools(campaign.id, store)}
    await tools["move_party"].ainvoke({"location_name": "Phandalin"})
    reloaded = await store.load(campaign.id)
    assert reloaded.locations[0].visited is True


# ── reach weapons ────────────────────────────────────────────────────────────

async def test_a_reach_weapon_wielder_threatens_2_squares_not_just_1(store, campaign):
    from unittest.mock import patch
    loc = _grid_location(grid=["......", "......", "......", "......", "......"])
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.monsters = [Monster(
        name="Pike Goblin", current_hp=7, max_hp=7, ac=13, reaction_available=True,
        attacks=[Attack(name="Pike", to_hit_bonus=4, damage_dice="1d10+2", damage_type=DamageType.PIERCING, reach_ft=10)],
        ability_scores=AbilityScores(dexterity=14),
    )]
    await store.save(campaign)
    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    await tools["start_encounter"].ainvoke({
        "location_description": "x",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Pike Goblin", "type": "monster", "initiative_override": 10},
        ],
    })
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Pike Goblin", "x": 0, "y": 0})
    # Start 2 squares away — within a 10 ft (2-square) reach weapon's threat range
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 0})

    # Move to 4 squares away — NOW leaves even a reach weapon's threat range
    with patch("backend.tools.resolution.random.randint", return_value=15):
        result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 4, "y": 0})
    assert "opportunity attack" in result.lower()


async def test_a_standard_weapon_cannot_react_from_2_squares_away(store, campaign):
    loc = _grid_location(grid=["......", "......", "......", "......", "......"])
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.monsters = [Monster(
        name="Goblin 1", current_hp=7, max_hp=7, ac=13, reaction_available=True,
        attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING, reach_ft=5)],
        ability_scores=AbilityScores(dexterity=14),
    )]
    await store.save(campaign)
    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    await tools["start_encounter"].ainvoke({
        "location_description": "x",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Goblin 1", "type": "monster", "initiative_override": 10},
        ],
    })
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 1", "x": 0, "y": 0})
    # Already 2 squares away — beyond a standard 5 ft (1-square) reach, so
    # there's nothing to "leave" here; no opportunity attack is possible
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 0})
    result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 4, "y": 0})
    assert "opportunity attack" not in result.lower()


# ── NPC allegiance ────────────────────────────────────────────────────────────

async def test_an_allied_npc_does_not_trigger_an_opportunity_attack_on_the_party(store, campaign):
    loc = _grid_location()
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.npcs = [NPC(name="Guard Captain", combat_stats=CombatStatBlock(
        max_hp=15, current_hp=15, ac=16, reaction_available=True,
        ability_scores=AbilityScores(dexterity=12),
        attacks=[Attack(name="Longsword", to_hit_bonus=4, damage_dice="1d8+2", damage_type=DamageType.SLASHING)],
    ))]
    await store.save(campaign)
    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    await tools["start_encounter"].ainvoke({
        "location_description": "x",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15, "side": "party"},
            {"name": "Guard Captain", "type": "npc", "initiative_override": 12, "side": "party"},
        ],
    })
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Guard Captain", "x": 2, "y": 2})

    result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    assert "opportunity attack" not in result.lower()


async def test_a_hostile_npc_triggers_a_real_opportunity_attack(store, campaign):
    from unittest.mock import patch
    loc = _grid_location()
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.npcs = [NPC(name="Bandit", combat_stats=CombatStatBlock(
        max_hp=15, current_hp=15, ac=13, reaction_available=True,
        ability_scores=AbilityScores(dexterity=12),
        attacks=[Attack(name="Shortsword", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.PIERCING)],
    ))]
    await store.save(campaign)
    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    await tools["start_encounter"].ainvoke({
        "location_description": "x",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Bandit", "type": "npc", "initiative_override": 12, "side": "hostile"},
        ],
    })
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Bandit", "x": 2, "y": 2})

    with patch("backend.tools.resolution.random.randint", return_value=20):
        result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    assert "opportunity attack" in result.lower()
    reloaded = await store.load(campaign.id)
    tarvokk = next(c for c in reloaded.party if c.name == "Tarvokk")
    assert tarvokk.current_hp < 20


# ── multiple qualifying opportunity attackers ────────────────────────────────

async def test_multiple_qualifying_attackers_all_actually_resolve(store, campaign):
    from unittest.mock import patch
    loc = _grid_location()
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=30, max_hp=30, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.monsters = [
        Monster(name="Goblin 1", current_hp=7, max_hp=7, ac=13, reaction_available=True,
                attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING)],
                ability_scores=AbilityScores(dexterity=14)),
        Monster(name="Goblin 2", current_hp=7, max_hp=7, ac=13, reaction_available=True,
                attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING)],
                ability_scores=AbilityScores(dexterity=14)),
    ]
    await store.save(campaign)
    tools = {t.name: t for t in make_combat_tools(campaign.id, store)}
    await tools["start_encounter"].ainvoke({
        "location_description": "x",
        "combatants": [
            {"name": "Tarvokk", "type": "character", "initiative_override": 15},
            {"name": "Goblin 1", "type": "monster", "initiative_override": 12},
            {"name": "Goblin 2", "type": "monster", "initiative_override": 10},
        ],
    })
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 1", "x": 2, "y": 2})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 2", "x": 3, "y": 1})

    with patch("backend.tools.resolution.random.randint", return_value=15):
        result = await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    # Both goblins' names appear as real resolved attacks, not just the first
    assert "Goblin 1" in result
    assert "Goblin 2" in result
    assert result.count("opportunity attack") == 2
