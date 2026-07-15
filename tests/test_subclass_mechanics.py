"""BDD-style coverage for subclass mechanical-feature grants (design.md item
#12's "still open" half, level-3 slice) — apply_subclass_features in
backend/tools/_helpers.py, wired into finalize_character (chargen.py),
generate_companion_character (companion.py), and level_up (levelup.py).
Name validation itself is already covered by test_subclass_validation.py.
"""
import pytest

from backend.data.fivee_options import SUBCLASS_BONUS_SPELLS, SUBCLASS_FEATURES, SUBCLASSES
from backend.models import AbilityScores, Character
from backend.stores.draft_store import DraftStore
from backend.tools._helpers import apply_subclass_features
from backend.tools.chargen import make_tools as make_chargen_tools
from backend.tools.levelup import make_tools as make_levelup_tools

from tests.conftest import make_character

def test_every_real_subclass_has_level_3_feature_data():
    for cls, subs in SUBCLASSES.items():
        for s in subs:
            assert s in SUBCLASS_FEATURES.get(cls, {}), f"missing feature data for {cls}/{s}"


def test_apply_subclass_features_grants_level_3_text():
    char = Character(
        name="Test", char_class="Fighter", subclass="Champion", level=3,
        ability_scores=AbilityScores(), max_hp=1, current_hp=1, ac=10,
    )
    apply_subclass_features(char)
    assert any("Improved Critical" in f for f in char.features)
    assert any("Remarkable Athlete" in f for f in char.features)


def test_apply_subclass_features_is_idempotent():
    char = Character(
        name="Test", char_class="Fighter", subclass="Champion", level=3,
        ability_scores=AbilityScores(), max_hp=1, current_hp=1, ac=10,
    )
    apply_subclass_features(char)
    first_len = len(char.features)
    apply_subclass_features(char)
    assert len(char.features) == first_len


def test_apply_subclass_features_no_ops_below_the_unlock_level():
    char = Character(
        name="Test", char_class="Fighter", subclass="Champion", level=2,
        ability_scores=AbilityScores(), max_hp=1, current_hp=1, ac=10,
    )
    apply_subclass_features(char)
    assert char.features == []


def test_apply_subclass_features_no_ops_for_unset_subclass():
    char = Character(
        name="Test", char_class="Fighter", subclass=None, level=5,
        ability_scores=AbilityScores(), max_hp=1, current_hp=1, ac=10,
    )
    apply_subclass_features(char)
    assert char.features == []


def test_apply_subclass_features_grants_bonus_prepared_spells_for_life_domain():
    char = Character(
        name="Test", char_class="Cleric", subclass="Life", level=3,
        ability_scores=AbilityScores(), max_hp=1, current_hp=1, ac=10,
    )
    apply_subclass_features(char)
    assert "Bless" in char.spells_prepared
    assert "Cure Wounds" in char.spells_prepared
    assert {s.name for s in char.spells_known} >= {"Bless", "Cure Wounds"}


def test_subclass_bonus_spells_only_reference_curated_spells():
    from backend.data.spells import ALL_SPELLS
    for cls_map in SUBCLASS_BONUS_SPELLS.values():
        for level_map in cls_map.values():
            for names in level_map.values():
                for name in names:
                    assert name in ALL_SPELLS, f"{name} not in curated ALL_SPELLS"


@pytest.mark.asyncio
async def test_finalize_character_applies_subclass_features(store, campaign):
    ds = DraftStore()
    tools = {t.name: t for t in make_chargen_tools(campaign.id, "player1", store, ds)}
    for field, value in [
        ("name", "Sir Champion"), ("race", "Human"), ("char_class", "Fighter"),
        ("background", "Soldier"), ("subclass", "Champion"),
    ]:
        await tools["update_character_draft"].ainvoke({"field": field, "value": value})
    await tools["update_ability_scores"].ainvoke({
        "strength": 14, "dexterity": 14, "constitution": 14,
        "intelligence": 10, "wisdom": 10, "charisma": 10,
    })
    result = await tools["finalize_character"].ainvoke({})
    assert "added to the party" in result
    reloaded = await store.load(campaign.id)
    char = reloaded.party[0]
    # A level-1 character shouldn't have level-3-only features yet.
    assert char.features == []


@pytest.mark.asyncio
async def test_level_up_reports_and_applies_newly_unlocked_subclass_features(store, campaign):
    char = make_character("Tarvokk", char_class="Fighter", level=2)
    campaign.party = [char]
    await store.save(campaign)

    tools = {t.name: t for t in make_levelup_tools(campaign.id, store)}
    result = await tools["level_up"].ainvoke({
        "character_name": "Tarvokk", "new_level": 3, "subclass": "Champion",
    })
    assert "New subclass features" in result
    assert "Improved Critical" in result

    reloaded = await store.load(campaign.id)
    reloaded_char = reloaded.party[0]
    assert any("Improved Critical" in f for f in reloaded_char.features)
