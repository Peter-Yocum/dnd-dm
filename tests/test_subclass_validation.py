"""BDD-style coverage for subclass name validation (design.md's chargen.py
follow-up, "a small, independent win that could ship first" ahead of full
subclass mechanics modeling) — validate_subclass in backend/tools/_helpers.py,
wired into finalize_character (chargen.py), generate_companion_character
(companion.py), and level_up (levelup.py).
"""
import pytest

from backend.stores.draft_store import DraftStore
from backend.tools.chargen import make_tools as make_chargen_tools
from backend.tools.companion import make_tools as make_companion_tools
from backend.tools.levelup import make_tools as make_levelup_tools

from tests.conftest import make_character

pytestmark = pytest.mark.asyncio


async def _finalize_a_fighter(store, campaign, subclass_value):
    ds = DraftStore()
    tools = {t.name: t for t in make_chargen_tools(campaign.id, "player1", store, ds)}
    for field, value in [
        ("name", "Tarvokk"), ("race", "Human"), ("char_class", "Fighter"),
        ("background", "Soldier"),
    ]:
        await tools["update_character_draft"].ainvoke({"field": field, "value": value})
    if subclass_value is not None:
        await tools["update_character_draft"].ainvoke({"field": "subclass", "value": subclass_value})
    await tools["update_ability_scores"].ainvoke({
        "strength": 14, "dexterity": 14, "constitution": 14,
        "intelligence": 10, "wisdom": 10, "charisma": 10,
    })
    return await tools["finalize_character"].ainvoke({})


async def test_finalize_character_rejects_an_invalid_subclass(store, campaign):
    result = await _finalize_a_fighter(store, campaign, "Necromancer")
    assert "isn't a real Fighter subclass" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.party == []  # rejected — nothing was added


async def test_finalize_character_normalizes_subclass_casing(store, campaign):
    result = await _finalize_a_fighter(store, campaign, "battle master")
    assert "added to the party" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.party[0].subclass == "Battle Master"


async def test_finalize_character_allows_no_subclass(store, campaign):
    result = await _finalize_a_fighter(store, campaign, None)
    assert "added to the party" in result


async def test_generate_companion_character_rejects_an_invalid_subclass(store, campaign):
    tools = {t.name: t for t in make_companion_tools(campaign.id, store)}
    result = await tools["generate_companion_character"].ainvoke({
        "name": "Bram", "race": "Dwarf", "char_class": "Fighter", "background": "Soldier",
        "strength": 16, "dexterity": 12, "constitution": 14,
        "intelligence": 10, "wisdom": 10, "charisma": 10,
        "subclass": "Necromancer",
    })
    assert "isn't a real Fighter subclass" in result
    reloaded = await store.load(campaign.id)
    assert reloaded.party == []


async def test_level_up_rejects_an_invalid_subclass_without_mutating_the_character(store, campaign):
    char = make_character("Tarvokk", char_class="Fighter", level=2)
    campaign.party = [char]
    await store.save(campaign)

    tools = {t.name: t for t in make_levelup_tools(campaign.id, store)}
    result = await tools["level_up"].ainvoke({
        "character_name": "Tarvokk", "new_level": 3, "subclass": "Necromancer",
    })
    assert "isn't a real Fighter subclass" in result

    # Then nothing was mutated — level, HP, subclass all untouched
    reloaded = await store.load(campaign.id)
    reloaded_char = reloaded.party[0]
    assert reloaded_char.level == 2
    assert reloaded_char.subclass is None


async def test_level_up_accepts_and_normalizes_a_valid_subclass(store, campaign):
    char = make_character("Tarvokk", char_class="Fighter", level=2)
    campaign.party = [char]
    await store.save(campaign)

    tools = {t.name: t for t in make_levelup_tools(campaign.id, store)}
    result = await tools["level_up"].ainvoke({
        "character_name": "Tarvokk", "new_level": 3, "subclass": "champion",
    })
    assert "level 2 -> 3" in result

    reloaded = await store.load(campaign.id)
    assert reloaded.party[0].subclass == "Champion"
