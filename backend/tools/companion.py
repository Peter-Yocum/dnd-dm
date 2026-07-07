"""
Companion character generation — a DM-initiated way to add an NPC-controlled
party member, distinct from Session 0's player-driven chargen flow.

No per-player draft, no multi-turn collaboration: the DM/agent decides every
field directly in one call, informed by current party composition and the
active adventure's recommended party size (see get_campaign_summary). This
tool is available both during Session 0 (to round out the party right after
the human players finish) and in-game (if the party thins out mid-campaign).
"""

from langchain_core.tools import tool

from backend.data.spells import SPELL_REQUIREMENTS
from backend.models import AbilityScores, Character
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import (
    build_spells_known, derive_level1_stats, derive_saving_throw_proficiencies, derive_spellcasting_stats,
    find_npc,
)


def make_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def generate_companion_character(
        name: str,
        race: str,
        char_class: str,
        background: str,
        strength: int,
        dexterity: int,
        constitution: int,
        intelligence: int,
        wisdom: int,
        charisma: int,
        subclass: str | None = None,
        alignment: str | None = None,
        skill_proficiencies: str = "",
        spells_known: str = "",
        personality_note: str = "",
        appearance: str = "",
    ) -> str:
        """Create a level-1 DM-controlled companion character and add them
        to the party — for filling the party out toward the adventure's
        recommended size, not for player characters (use Session 0 for those).

        Before calling this, check get_campaign_summary for the current
        party's races/classes/backgrounds and the adventure's recommended
        player count, and choose a build that complements what's missing —
        don't duplicate an existing role (e.g. don't add a fourth Rogue to a
        party that has no healer). Use list_options / get_option_details
        (race, class, background) if you need to check what's available or
        get exact mechanical details — those tools work outside Session 0
        too, they just read shared reference data.

        strength/dexterity/etc: final ability scores, already including the
        chosen Background's ability score increase — species grant no
        ability score bonus in this ruleset (see get_option_details race
        for a specific species' traits).
        skill_proficiencies: comma-separated, e.g. "Perception, Survival".
        spells_known: comma-separated, ONLY for a spellcasting class — check
        list_options('spells <class>') first for the real menu and required
        counts (cantrips + level-1 spells together, same rule as Session 0).
        A wrong count or a name not on that class's menu is rejected with a
        corrective message and no character is created — no silent default,
        choose deliberately the same way you already do for race/class/
        background. Non-caster classes: leave this empty.
        appearance: a short physical description — this companion will be
        introduced to the player at the next session opening, so give them
        something visual and specific, not just a name and class.
        """
        campaign = await store.load(campaign_id)
        if not campaign:
            return "Error: campaign not found."

        existing = next((c for c in campaign.party if c.name.lower() == name.lower()), None)
        if existing:
            return f"A character named '{name}' already exists in the party. Choose a different name."
        if find_npc(campaign, name):
            return (
                f"'{name}' is already an NPC's name in this campaign — pick a different "
                f"name for this companion. Two characters sharing a name is confusing "
                f"and easy to mix up in play."
            )

        ab = AbilityScores(
            strength=strength, dexterity=dexterity, constitution=constitution,
            intelligence=intelligence, wisdom=wisdom, charisma=charisma,
        )
        skills = {s.strip().lower() for s in skill_proficiencies.split(",") if s.strip()}
        derived = derive_level1_stats(ab, char_class, skills)
        spell_stats = derive_spellcasting_stats(ab, char_class, proficiency_bonus=2)

        chosen_spell_names = [s.strip() for s in spells_known.split(",") if s.strip()]
        spell_objs, spells_prepared, spell_err = build_spells_known(char_class, chosen_spell_names)
        if char_class in SPELL_REQUIREMENTS and spell_err:
            return (
                f"Cannot create {name} — {spell_err} Call list_options('spells {char_class}') "
                f"for the real menu, then call generate_companion_character again with a "
                f"corrected spells_known list."
            )

        character = Character(
            name=name,
            race=race,
            char_class=char_class,
            subclass=subclass,
            background=background,
            alignment=alignment,
            appearance=appearance,
            level=1,
            ability_scores=ab,
            proficiency_bonus=2,
            max_hp=derived["max_hp"],
            current_hp=derived["max_hp"],
            ac=derived["ac"],
            passive_perception=derived["passive_perception"],
            spell_slots=derived["spell_slots"],
            hit_dice_total=derived["hit_dice_total"],
            hit_dice_remaining=1,
            attacks=derived["attacks"],
            inventory=derived["inventory"],
            currency=derived["currency"],
            skill_proficiencies=skills,
            saving_throw_proficiencies=derive_saving_throw_proficiencies(char_class),
            spellcasting_ability=spell_stats["spellcasting_ability"],
            spell_save_dc=spell_stats["spell_save_dc"],
            spell_attack_bonus=spell_stats["spell_attack_bonus"],
            spells_known=spell_objs,
            spells_prepared=spells_prepared,
            notes=personality_note,
            is_player_controlled=False,
        )

        campaign.party.append(character)
        await store.save(campaign)

        return (
            f"✓ {character.name} ({character.race} {character.char_class}) has joined the party "
            f"as a DM-controlled companion. HP: {character.max_hp}, AC: {character.ac}."
        )

    return [generate_companion_character]
