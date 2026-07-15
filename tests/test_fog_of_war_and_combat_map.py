"""BDD-style coverage for fog-of-war (design.md's mapping-cluster idea #3)
and the live combat-map panel (idea #2) — both deferred from the original
grid-maps/opportunity-attacks work, built once real (x,y) positions and a
colored-grid renderer already existed. See backend/map_render.py
(render_grid_fogged), backend/tools/combat.py (set_combatant_position's
revealed_positions hook), and backend/main.py (GET /campaigns/{id}/combat-map,
GET /campaigns/{id}/maps's conditional fog).
"""
import pytest

from backend.map_render import render_grid, render_grid_fogged
from backend.models import Attack, AbilityScores, Character, DamageType, Location, Monster


# ── render_grid_fogged (pure function, no DB) ────────────────────────────────

def test_a_cell_within_radius_of_a_revealed_position_keeps_its_real_symbol():
    grid = [".....", ".....", ".....", ".....", "....."]
    rows = render_grid_fogged(grid, {}, revealed_positions=[(2, 2)], radius=1)
    assert rows[2][2] == {"symbol": ".", "kind": "floor"}
    assert rows[1][2] == {"symbol": ".", "kind": "floor"}  # 1 square away — in view


def test_a_cell_beyond_radius_becomes_fog():
    grid = [".....", ".....", ".....", ".....", "....."]
    rows = render_grid_fogged(grid, {}, revealed_positions=[(2, 2)], radius=1)
    assert rows[0][0] == {"symbol": "?", "kind": "fog"}  # 2 squares away — beyond radius 1


def test_multiple_revealed_positions_each_light_up_their_own_area():
    grid = ["......", "......", "......", "......", "......", "......"]
    rows = render_grid_fogged(grid, {}, revealed_positions=[(0, 0), (5, 5)], radius=0)
    assert rows[0][0]["kind"] == "floor"
    assert rows[5][5]["kind"] == "floor"
    assert rows[2][2]["kind"] == "fog"  # nowhere near either revealed position


def test_empty_revealed_positions_fogs_the_entire_grid():
    # Caller's responsibility (per this function's own docstring) to only
    # call it when revealed_positions is non-empty — verifying the raw
    # behavior here so that contract stays honest.
    grid = ["...", "...", "..."]
    rows = render_grid_fogged(grid, {}, revealed_positions=[], radius=5)
    assert all(cell["kind"] == "fog" for row in rows for cell in row)


def test_render_grid_fogged_preserves_real_terrain_kind_when_in_view():
    grid = ["#T."]
    legend = {"#": "wall", "T": "tree"}
    rows = render_grid_fogged(grid, legend, revealed_positions=[(0, 0)], radius=5)
    assert rows[0][0] == {"symbol": "#", "kind": "wall"}
    assert rows[0][1] == {"symbol": "T", "kind": "vegetation"}


# ── set_combatant_position's revealed_positions hook ─────────────────────────

async def _started_encounter_with_grid(store, campaign):
    from backend.tools.combat import make_tools as make_combat_tools
    loc = Location(name="Goblin Den", scale="site", grid=[".....", ".....", "....."], legend={})
    campaign.locations = [loc]
    campaign.current_location_id = loc.id
    campaign.party = [Character(name="Tarvokk", current_hp=20, max_hp=20, ac=15, is_player_controlled=True, reaction_available=True)]
    campaign.monsters = [Monster(
        name="Goblin 1", current_hp=7, max_hp=7, ac=13, reaction_available=True,
        attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING)],
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
    return tools


@pytest.mark.asyncio
async def test_a_party_side_movers_position_is_added_to_revealed_positions(store, campaign):
    tools = await _started_encounter_with_grid(store, campaign)
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    reloaded = await store.load(campaign.id)
    assert (2, 1) in reloaded.locations[0].revealed_positions


@pytest.mark.asyncio
async def test_moving_to_the_same_cell_twice_does_not_duplicate_the_entry(store, campaign):
    tools = await _started_encounter_with_grid(store, campaign)
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 0, "y": 0})
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Tarvokk", "x": 2, "y": 1})
    reloaded = await store.load(campaign.id)
    assert reloaded.locations[0].revealed_positions.count((2, 1)) == 1


@pytest.mark.asyncio
async def test_a_hostile_movers_position_is_not_added_to_revealed_positions(store, campaign):
    tools = await _started_encounter_with_grid(store, campaign)
    await tools["set_combatant_position"].ainvoke({"combatant_name": "Goblin 1", "x": 2, "y": 1})
    reloaded = await store.load(campaign.id)
    assert (2, 1) not in reloaded.locations[0].revealed_positions


# ── /campaigns/{id}/maps route falls back to unfogged for a location with
#    no revealed_positions recorded (verified against the pure rendering
#    logic the route calls, matching main.py's own conditional exactly) ────

def test_a_location_with_no_revealed_positions_renders_fully_not_fogged():
    grid = [".....", ".....", "....."]
    # Mirrors main.py's own logic: render_grid (not _fogged) when
    # revealed_positions is empty — a peaceful, never-fought-in location
    # should never look solid-black just because nobody's tracked positions there.
    revealed_positions = []
    rendered = render_grid_fogged(grid, {}, revealed_positions) if revealed_positions else render_grid(grid, {})
    assert all(cell["kind"] == "floor" for row in rendered for cell in row)
