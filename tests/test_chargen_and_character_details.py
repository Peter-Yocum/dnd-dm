"""BDD-style coverage for two 2026-07-13 fixes:

1. STARTING_KITS data fidelity (design.md's chargen.py follow-up) — every
   class's kit must reference real WEAPONS/ARMOR entries, and the two
   originally-reported bugs (Ranger/Rogue missing a dagger, flat/wrong gold)
   must not regress.
2. Character pronouns end-to-end (Session 0 draft -> finalized Character)
   and update_character_detail, the in-game cosmetic-field editor for an
   already-finalized party member.
"""
import pytest

from backend.data.equipment import ARMOR, STARTING_KITS, WEAPONS
from backend.models import Character
from backend.stores.draft_store import DraftStore
from backend.tools.chargen import make_tools as make_chargen_tools
from backend.tools.party import make_tools as make_party_tools


@pytest.mark.parametrize("char_class", sorted(STARTING_KITS))
def test_every_starting_kit_weapon_and_armor_is_real_data(char_class):
    # Given each class's starting kit
    kit = STARTING_KITS[char_class]

    # Then its weapon/armor must be entries this app actually knows the
    # stats for, not a typo/omission that'd silently produce a weaponless
    # or unarmored character
    assert kit["weapon"] in WEAPONS, f"{char_class}: '{kit['weapon']}' not in WEAPONS"
    if kit["armor"]:
        assert kit["armor"] in ARMOR, f"{char_class}: '{kit['armor']}' not in ARMOR"


def test_ranger_and_rogue_kits_include_a_dagger():
    # Given the two classes originally reported missing a dagger
    ranger_gear = " ".join(STARTING_KITS["Ranger"]["gear"])
    rogue_gear = " ".join(STARTING_KITS["Rogue"]["gear"])

    # Then Rogue's real PHB Option A kit includes 2 daggers (Ranger's real
    # kit does NOT — that was a mistaken addition caught during the full
    # fidelity pass, see design.md)
    assert "Dagger" in rogue_gear
    assert "Dagger" not in ranger_gear


@pytest.mark.parametrize("char_class,expected_gold", [
    ("Ranger", 7), ("Rogue", 8), ("Cleric", 7), ("Fighter", 4), ("Barbarian", 15),
])
def test_starting_gold_matches_the_real_phb_option_a_row(char_class, expected_gold):
    assert STARTING_KITS[char_class]["gold"] == expected_gold


@pytest.mark.asyncio
async def test_pronouns_flow_from_draft_to_finalized_character(store, campaign):
    # Given a Session 0 draft with every required field set, including pronouns
    ds = DraftStore()
    tools = {t.name: t for t in make_chargen_tools(campaign.id, "player1", store, ds)}
    update_draft = tools["update_character_draft"]
    finalize = tools["finalize_character"]

    for field, value in [
        ("name", "Kaelen"), ("race", "Elf"), ("char_class", "Fighter"),
        ("background", "Soldier"), ("pronouns", "they/them"),
    ]:
        await update_draft.ainvoke({"field": field, "value": value})
    await tools["update_ability_scores"].ainvoke({
        "strength": 14, "dexterity": 14, "constitution": 14,
        "intelligence": 10, "wisdom": 10, "charisma": 10,
    })

    # When the character is finalized
    result = await finalize.ainvoke({})
    assert "added to the party" in result

    # Then pronouns landed on the real, persisted Character — not lost
    # between the draft and the finalized record
    reloaded = await store.load(campaign.id)
    kaelen = next(c for c in reloaded.party if c.name == "Kaelen")
    assert kaelen.pronouns == "they/them"


@pytest.mark.asyncio
async def test_update_character_detail_edits_a_finalized_characters_pronouns(store, campaign):
    # Given an already-finalized party member with no pronouns set (e.g.
    # created before this fix, or the player skipped the question)
    campaign.party = [Character(name="Kaelen", pronouns="")]
    await store.save(campaign)

    tools = {t.name: t for t in make_party_tools(campaign.id, store, None, None)}
    update_detail = tools["update_character_detail"]

    # When it's set after the fact
    result = await update_detail.ainvoke({
        "character_name": "Kaelen", "field": "pronouns", "value": "she/her",
    })
    assert "she/her" in result

    reloaded = await store.load(campaign.id)
    assert next(c for c in reloaded.party if c.name == "Kaelen").pronouns == "she/her"


@pytest.mark.asyncio
async def test_update_character_detail_refuses_mechanical_fields(store, campaign):
    # Given a finalized character
    campaign.party = [Character(name="Kaelen", race="Elf")]
    await store.save(campaign)

    tools = {t.name: t for t in make_party_tools(campaign.id, store, None, None)}
    update_detail = tools["update_character_detail"]

    # When a mechanical field is attempted (not one of the cosmetic-only fields)
    result = await update_detail.ainvoke({
        "character_name": "Kaelen", "field": "race", "value": "Dwarf",
    })

    # Then it refuses instead of silently mutating derived-stat-affecting data
    assert "isn't editable" in result
    reloaded = await store.load(campaign.id)
    assert next(c for c in reloaded.party if c.name == "Kaelen").race == "Elf"


@pytest.mark.asyncio
async def test_update_character_detail_clear_blanks_alignment_to_none(store, campaign):
    # Given a finalized character with an alignment set
    campaign.party = [Character(name="Kaelen", alignment="Chaotic Good")]
    await store.save(campaign)

    tools = {t.name: t for t in make_party_tools(campaign.id, store, None, None)}
    update_detail = tools["update_character_detail"]

    # When cleared
    await update_detail.ainvoke({"character_name": "Kaelen", "field": "alignment", "value": "CLEAR"})

    # Then it's really None (matching the field's str | None type), not an empty string
    reloaded = await store.load(campaign.id)
    assert next(c for c in reloaded.party if c.name == "Kaelen").alignment is None
