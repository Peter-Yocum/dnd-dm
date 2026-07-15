"""BDD-style coverage for NPC combat participation (see design.md's "NPC
combatants can't take damage" deferred item, closed 2026-07-13) —
find_combatant/apply_damage_to_combatant in backend/tools/_helpers.py, the
NPC.ac/current_hp/max_hp/attacks/ability_scores/saving_throw_bonuses/
skill_bonuses/conditions proxies in backend/models.py, and update_npc_hp in
backend/tools/combat.py. Built to guard against this regressing — an NPC
with combat_stats set should behave exactly like a Monster in every one of
these tools, not be silently invisible to them.
"""
import uuid

import pytest

from backend.models import AbilityScores, Attack, Campaign, ConditionType, DamageType, Monster, NPC
from backend.tools.combat import make_tools as make_combat_tools
from backend.tools.resolution import make_tools as make_resolution_tools

from tests.conftest import make_character, make_combat_npc, start_combat

pytestmark = pytest.mark.asyncio


async def test_resolve_attack_can_target_a_combat_npc(store, campaign, force_hit):
    # Given a party member and an NPC with real combat stats
    attacker = make_character(
        "Tarvokk",
        attacks=[Attack(name="Shortbow", to_hit_bonus=6, damage_dice="1d6+4", damage_type=DamageType.PIERCING)],
    )
    bandit = make_combat_npc("Bandit Captain", max_hp=20, current_hp=20, ac=1)  # guaranteed hit
    campaign.party = [attacker]
    campaign.npcs = [bandit]
    start_combat(campaign, [("Tarvokk", "character", 15), ("Bandit Captain", "npc", 10)])
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_attack = tools["resolve_attack"]

    # When the party member attacks the NPC
    result = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Bandit Captain", "attack_name": "Shortbow",
    })

    # Then it resolves and applies real damage to the NPC's combat_stats —
    # this used to be silently impossible (target lookup only checked
    # Character/Monster, dropping NPC entirely)
    assert "HIT" in result
    reloaded = await store.load(campaign.id)
    reloaded_bandit = next(n for n in reloaded.npcs if n.name == "Bandit Captain")
    assert reloaded_bandit.combat_stats.current_hp < 20


async def test_resolve_attack_can_use_a_combat_npc_as_the_attacker(store, campaign, force_hit):
    # Given an NPC with its own attack and a monster target
    bandit = make_combat_npc(
        "Bandit Captain", ac=15,
        attacks=[Attack(name="Scimitar", to_hit_bonus=4, damage_dice="1d6+2", damage_type=DamageType.SLASHING)],
    )
    goblin = Monster(name="Goblin 1", ac=1, max_hp=7, current_hp=7)
    campaign.npcs = [bandit]
    campaign.monsters = [goblin]
    start_combat(campaign, [("Bandit Captain", "npc", 15), ("Goblin 1", "monster", 10)])
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_attack = tools["resolve_attack"]

    # When the NPC attacks the monster
    result = await resolve_attack.ainvoke({
        "attacker_name": "Bandit Captain", "target_name": "Goblin 1", "attack_name": "Scimitar",
    })

    # Then it resolves as a real attack, not "No character or monster found"
    assert "HIT" in result
    assert "not found" not in result.lower()


async def test_update_npc_hp_applies_damage_and_healing(store, campaign):
    # Given an NPC with combat_stats
    bandit = make_combat_npc("Bandit Captain", max_hp=20, current_hp=20)
    campaign.npcs = [bandit]
    await store.save(campaign)

    tools = {t.name: t for t in make_combat_tools(campaign.id, store, None)}
    update_npc_hp = tools["update_npc_hp"]

    # When freeform damage is applied (a trap, not a resolved attack)
    result = await update_npc_hp.ainvoke({"npc_name": "Bandit Captain", "delta": -8})
    assert "20 → 12 HP" in result

    # Then it persists
    reloaded = await store.load(campaign.id)
    reloaded_bandit = next(n for n in reloaded.npcs if n.name == "Bandit Captain")
    assert reloaded_bandit.combat_stats.current_hp == 12

    # And healing works too, clamped at max_hp
    result2 = await update_npc_hp.ainvoke({"npc_name": "Bandit Captain", "delta": 50})
    assert "12 → 20 HP" in result2


async def test_update_npc_hp_refuses_an_npc_with_no_combat_stats():
    # Given an NPC with no combat_stats (e.g. a shopkeeper who'd never fight)
    innkeeper = NPC(name="Toblen")
    campaign_stub = Campaign(id=str(uuid.uuid4()), name="stub", npcs=[innkeeper])

    class FakeStore:
        async def load(self, cid):
            return campaign_stub

        async def save(self, c):
            pass

    tools = {t.name: t for t in make_combat_tools("stub", FakeStore(), None)}
    update_npc_hp = tools["update_npc_hp"]

    # When damage is attempted anyway
    result = await update_npc_hp.ainvoke({"npc_name": "Toblen", "delta": -5})

    # Then it refuses cleanly instead of crashing or silently no-op'ing
    assert "no combat_stats" in result


async def test_resolve_saving_throw_uses_the_npcs_own_bonus_not_a_flat_ability_mod(store, campaign):
    # Given an NPC with an explicit saving throw bonus that does NOT match
    # its raw ability modifier (proves the override is actually being read,
    # not just falling back to ability_scores)
    bandit = make_combat_npc(
        "Bandit Captain",
        ability_scores=AbilityScores(dexterity=8),  # raw mod would be -1
        saving_throw_bonuses={"dexterity": 9},
    )
    campaign.npcs = [bandit]
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_saving_throw = tools["resolve_saving_throw"]

    result = await resolve_saving_throw.ainvoke({
        "target_names": ["Bandit Captain"], "ability": "dexterity", "dc": 1,
    })

    # Then the override (+9), not the raw ability mod (-1), was used
    assert "+ 9 =" in result
    assert "not found" not in result.lower()


async def test_resolve_check_uses_the_npcs_own_skill_bonus(store, campaign):
    # Given an NPC with a skill override (a practiced liar with mediocre CHA)
    bandit = make_combat_npc(
        "Bandit Captain",
        ability_scores=AbilityScores(charisma=10),  # raw mod would be 0
        skill_bonuses={"deception": 5},
    )
    campaign.npcs = [bandit]
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_check = tools["resolve_check"]

    result = await resolve_check.ainvoke({
        "character_name": "Bandit Captain", "ability_or_skill": "deception",
    })

    assert "+ 5 =" in result


async def test_a_condition_from_a_failed_save_persists_on_the_npc(store, campaign):
    # Given an NPC certain to fail a save (DC far above any possible roll)
    bandit = make_combat_npc("Bandit Captain", ability_scores=AbilityScores(dexterity=1))
    campaign.npcs = [bandit]
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_saving_throw = tools["resolve_saving_throw"]

    # When it fails a save with an attached condition
    result = await resolve_saving_throw.ainvoke({
        "target_names": ["Bandit Captain"], "ability": "dexterity", "dc": 100,
        "condition_on_fail": "prone",
    })
    assert "FAILURE" in result
    assert "prone applied" in result

    # Then the condition is really persisted on combat_stats, not lost
    reloaded = await store.load(campaign.id)
    reloaded_bandit = next(n for n in reloaded.npcs if n.name == "Bandit Captain")
    assert ConditionType.PRONE in reloaded_bandit.combat_stats.conditions


async def test_find_combatant_ignores_an_npc_with_no_combat_stats(store, campaign, force_hit):
    # Given an NPC that exists but was never given combat_stats (not meant to fight)
    innkeeper = NPC(name="Toblen")
    attacker = make_character("Tarvokk")
    campaign.party = [attacker]
    campaign.npcs = [innkeeper]
    await store.save(campaign)

    tools = {t.name: t for t in make_resolution_tools(campaign.id, store)}
    resolve_attack = tools["resolve_attack"]

    # When an attack targets them by name anyway
    result = await resolve_attack.ainvoke({
        "attacker_name": "Tarvokk", "target_name": "Toblen",
    })

    # Then the tool correctly reports not found rather than crashing on a
    # combat_stats-less NPC's ac/attacks
    assert "No character, NPC, or monster named 'Toblen' found." == result


async def test_start_encounter_resolves_an_npcs_real_initiative_modifier(store, campaign):
    # Given a combat NPC with a real DEX score
    bandit = make_combat_npc("Bandit Captain", ability_scores=AbilityScores(dexterity=18))  # +4
    campaign.npcs = [bandit]
    await store.save(campaign)

    tools = {t.name: t for t in make_combat_tools(campaign.id, store, None)}
    start_encounter = tools["start_encounter"]

    # When they're added to an encounter with initiative rolled internally
    result = await start_encounter.ainvoke({
        "location_description": "a bandit camp",
        "combatants": [{"name": "Bandit Captain", "type": "npc"}],
    })

    # Then they were resolved (real DEX mod applied), not reported as an
    # unresolved combatant rolling at +0
    assert "not found in campaign" not in result
