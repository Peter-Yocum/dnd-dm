"""BDD-style coverage for the combat action-economy engine (see
backend/tools/_helpers.py's check_and_spend_action_budget/
format_turn_budget_recap/advance_combatant_turn, and ActiveEffect in
backend/models.py) — built to guard against the "snuck in two attacks" bug
this system closes.

Each test is a Given/When/Then scenario against the real tool functions and
the real dev Postgres, not mocks — the same style of manual verification run
by hand throughout the session that motivated this suite, now permanent.
"""
import pytest

from backend.models import ActiveEffect, Attack, DamageType, Spell, SpellSlotLevel
from backend.tools.resolution import make_tools as make_resolution_tools
from backend.tools._helpers import advance_combatant_turn

from tests.conftest import make_character, make_monster, start_combat

pytestmark = pytest.mark.asyncio


async def test_a_second_attack_in_the_same_turn_is_refused(store, campaign, force_hit):
    # Given a character with a bow, mid-turn in an active encounter
    attacker = make_character(
        "Tarvokk",
        attacks=[Attack(name="Shortbow", to_hit_bonus=6, damage_dice="1d6+4",
                         damage_type=DamageType.PIERCING, action_type="action")],
    )
    goblin = make_monster("Goblin 1", ac=1)  # guaranteed hit
    campaign.party = [attacker]
    campaign.monsters = [goblin]
    start_combat(campaign, [("Tarvokk", "character", 15), ("Goblin 1", "monster", 10)])
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_attack = tools["resolve_attack"]

    # When they attack once
    first = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Goblin 1", "attack_name": "Shortbow",
    })
    # Then it resolves normally and the goblin takes damage
    assert "HIT" in first
    assert "Goblin 1" in first

    # When they try to attack again with their Action this same turn
    second = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Goblin 1", "attack_name": "Shortbow",
    })
    # Then the tool refuses instead of rolling a second attack
    assert "already used their Action this turn" in second


async def test_an_offhand_bonus_action_attack_does_not_consume_the_action(store, campaign, force_hit):
    # Given a character with both a primary (action) weapon and an offhand
    # (bonus_action) weapon
    attacker = make_character(
        "Tarvokk",
        attacks=[
            Attack(name="Shortsword", to_hit_bonus=5, damage_dice="1d6+2",
                   damage_type=DamageType.SLASHING, action_type="action"),
            Attack(name="Dagger (offhand)", to_hit_bonus=5, damage_dice="1d4+2",
                   damage_type=DamageType.PIERCING, action_type="bonus_action"),
        ],
    )
    goblin = make_monster("Goblin 1", ac=1)
    campaign.party = [attacker]
    campaign.monsters = [goblin]
    start_combat(campaign, [("Tarvokk", "character", 15), ("Goblin 1", "monster", 10)])
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_attack = tools["resolve_attack"]

    # When they use their Action on the primary weapon
    first = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Goblin 1", "attack_name": "Shortsword",
        "action_type": "action",
    })
    assert "HIT" in first

    # And then their Bonus Action on the offhand weapon, same turn
    second = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Goblin 1", "attack_name": "Dagger (offhand)",
        "action_type": "bonus_action",
    })
    # Then both are allowed — they're different resources
    assert "HIT" in second
    assert "already used" not in second

    # But a second Action-type attack afterward is still refused
    third = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Goblin 1", "attack_name": "Shortsword",
        "action_type": "action",
    })
    assert "already used their Action this turn" in third


async def test_haste_grants_and_later_expires_an_extra_action(store, campaign):
    # Given a hasted character mid-encounter
    attacker = make_character("Tarvokk")
    goblin = make_monster("Goblin 1")
    campaign.party = [attacker]
    campaign.monsters = [goblin]
    enc = start_combat(campaign, [("Tarvokk", "character", 15), ("Goblin 1", "monster", 10)])
    attacker.active_effects.append(ActiveEffect(name="Haste", duration_rounds=2, extra_actions=1))
    await store.save(campaign)

    def tarvokk_entry():
        return next(e for e in campaign.active_encounter.initiative_order if e.name == "Tarvokk")

    # When a full round passes and it becomes Tarvokk's turn again
    advance_combatant_turn(campaign, enc)  # -> Goblin's turn
    advance_combatant_turn(campaign, enc)  # -> Tarvokk's turn, round 2 (Haste: 2 -> 1 round left)

    # Then their turn budget reflects Haste's extra action automatically
    assert tarvokk_entry().actions_remaining == 2

    # When one more full round passes (Haste's duration_rounds=2 elapses:
    # it ticks down once per Tarvokk turn-start, so it expires on the
    # SECOND one, not the third)
    advance_combatant_turn(campaign, enc)  # Goblin
    advance_combatant_turn(campaign, enc)  # Tarvokk round 3 — Haste expires (1 -> 0)

    # Then the extra action is gone and the effect itself has been removed
    assert tarvokk_entry().actions_remaining == 1
    assert attacker.active_effects == []


async def test_a_killing_blow_actually_persists_the_monsters_hp(store, campaign, force_hit):
    # Given a heavily wounded goblin about to be finished off
    attacker = make_character(
        "Tarvokk",
        attacks=[Attack(name="Shortbow", to_hit_bonus=6, damage_dice="10d6",
                         damage_type=DamageType.PIERCING, action_type="action")],
    )
    goblin = make_monster("Goblin 1", max_hp=7, current_hp=1, ac=1)
    campaign.party = [attacker]
    campaign.monsters = [goblin]
    start_combat(campaign, [("Tarvokk", "character", 15), ("Goblin 1", "monster", 10)])
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_attack = tools["resolve_attack"]

    # When the attack lands
    result = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Goblin 1", "attack_name": "Shortbow",
    })
    assert "DEFEATED" in result

    # Then the monster's HP is really 0 in the persisted campaign, not just
    # narrated as dead — reload from the store to prove it round-tripped
    reloaded = await store.load(campaign.id)
    reloaded_goblin = next(m for m in reloaded.monsters if m.name == "Goblin 1")
    assert reloaded_goblin.current_hp == 0


async def test_cast_spell_derives_bonus_action_type_from_casting_time(store, campaign):
    # Given a caster who knows a bonus-action spell and has a slot for it
    caster = make_character(
        "Elara",
        spells_known=[Spell(name="Healing Word", level=1, casting_time="1 bonus action",
                             effect_dice="1d4", is_healing=True)],
        spell_slots={1: SpellSlotLevel(max=2, used=0)},
    )
    campaign.party = [caster]
    start_combat(campaign, [("Elara", "character", 12)])
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    cast_spell = tools["cast_spell"]

    # When cast without an explicit action_type override
    result = await cast_spell.ainvoke({
        "caster_name": "Elara", "spell_name": "Healing Word", "target_names": ["Elara"],
    })
    assert "casts Healing Word" in result

    # Then it only spent the Bonus Action, not the Action — a real Action is
    # still available this same turn. Reload from the store: cast_spell
    # mutated its own fresh load, not this test's in-memory `campaign`.
    reloaded = await store.load(campaign.id)
    entry = next(e for e in reloaded.active_encounter.initiative_order if e.name == "Elara")
    assert entry.actions_remaining == 1
    assert entry.bonus_actions_remaining == 0
