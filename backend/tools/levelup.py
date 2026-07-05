"""
Level-up tool — advances a character to a higher level, recomputing every
level-dependent stat in one call (HP, proficiency bonus, spell slots, weapon
to-hit bonuses, hit dice) from real data, the same "never let the model
invent a number" discipline chargen.py already applies to character
creation.

Built 2026-07-03 after a real live session where the DM narrated "The party
has reached Level 2" with nothing backing it — no leveling mechanism existed
anywhere in this codebase before this file, so the narrated milestone never
actually changed any character's stats.
"""

from langchain_core.tools import tool

from backend.data.fivee_options import HIT_DICE, SPELL_SLOTS_BY_LEVEL, proficiency_bonus_for_level
from backend.data.spells import ALL_SPELLS, SPELL_MENUS
from backend.models import SpellSlotLevel
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import derive_spellcasting_stats, find_char


def make_tools(campaign_id: str, store: CampaignStore) -> list:

    @tool
    async def level_up(
        character_name: str,
        new_level: int,
        new_spells_known: str = "",
        subclass: str = "",
    ) -> str:
        """Advance a PC or companion to a new level in ONE call — recomputes
        every level-dependent stat from real data (HP, proficiency bonus,
        spell slots, stored weapon to-hit bonuses, hit dice) instead of
        narrating a level-up with nothing behind it. Call this whenever the
        party actually reaches a new level (a real milestone or XP
        threshold), not just when it's narratively convenient to mention —
        never say a level-up happened without this tool call backing it.

        new_level: the character's new level, an absolute target (not a
        delta) — supports jumping more than one level in one call for a big
        milestone. Must be higher than the character's current level, max 20.

        new_spells_known: comma-separated NEW spell names to add to
        spells_known, ONLY if this class actually gains new known spells at
        this level — that varies by class/level and this app has no
        per-level spell-count table (only the level-1 requirement in
        SPELL_REQUIREMENTS), so check search_rules or the class's own
        leveling text if unsure whether any are gained. Each name must be on
        the class's menu (list_options('spells <class>')) and not already
        known. Leave empty if nothing new is gained, or the class isn't a
        caster. This app's spell data (ALL_SPELLS/SPELL_MENUS) only covers
        cantrips and level-1 spells today — if this level unlocks level-2+
        spell slots, there's no real level-2+ spell content to pick from yet;
        say so plainly in your resolution report rather than inventing one.

        subclass: set when the character selects a subclass at this level
        (commonly level 3) — pass the exact subclass name the player chose.
        Not validated against a fixed list.
        """
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        if new_level <= char.level:
            return f"{char.name} is already level {char.level} — new_level must be higher."
        if new_level > 20:
            return "5e characters cap at level 20."

        # Validate the new-spells request fully before mutating anything —
        # a rejected call must leave the character completely untouched.
        slot_table = SPELL_SLOTS_BY_LEVEL.get(char.char_class)
        canonical_new_spells: list[str] = []
        if slot_table:
            new_names = [s.strip() for s in new_spells_known.split(",") if s.strip()]
            if new_names:
                menu = SPELL_MENUS.get(char.char_class, {})
                valid_names = {name.lower(): name for tier in menu.values() for name in tier}
                known_names = {s.name for s in char.spells_known}
                bad = [n for n in new_names if n.lower() not in valid_names]
                if bad:
                    return (
                        f"Not on {char.char_class}'s spell menu: {', '.join(bad)}. "
                        f"Check list_options('spells {char.char_class}')."
                    )
                canonical_new_spells = [valid_names[n.lower()] for n in new_names]
                dupes = [n for n in canonical_new_spells if n in known_names]
                if dupes:
                    return f"{char.name} already knows: {', '.join(dupes)}."

        levels_gained = new_level - char.level
        hit_die = HIT_DICE.get(char.char_class, 8)
        con_mod = char.ability_scores.modifier(char.ability_scores.constitution)
        hp_gain = levels_gained * (hit_die // 2 + 1 + con_mod)

        old_level, old_prof = char.level, char.proficiency_bonus
        new_prof = proficiency_bonus_for_level(new_level)
        prof_delta = new_prof - old_prof

        char.max_hp += hp_gain
        char.current_hp += hp_gain
        char.proficiency_bonus = new_prof
        for atk in char.attacks:
            atk.to_hit_bonus += prof_delta
        char.hit_dice_remaining += levels_gained
        char.hit_dice_total = f"{new_level}d{hit_die}"
        char.level = new_level

        lines = [
            f"{char.name}: level {old_level} -> {new_level}.",
            f"HP: +{hp_gain} ({char.max_hp - hp_gain} -> {char.max_hp})",
            f"Proficiency bonus: +{old_prof} -> +{new_prof}",
        ]

        if slot_table:
            new_slots = slot_table.get(new_level, {})
            char.spell_slots = {lvl: SpellSlotLevel(max=count) for lvl, count in new_slots.items()}
            spell_stats = derive_spellcasting_stats(char.ability_scores, char.char_class, new_prof)
            char.spell_save_dc = spell_stats["spell_save_dc"]
            char.spell_attack_bonus = spell_stats["spell_attack_bonus"]
            lines.append(f"Spell slots now: {new_slots}")

            if canonical_new_spells:
                for n in canonical_new_spells:
                    char.spells_known.append(ALL_SPELLS[n].model_copy())
                    char.spells_prepared.append(n)
                lines.append(f"New spells known: {', '.join(canonical_new_spells)}")

        if subclass:
            char.subclass = subclass
            lines.append(f"Subclass: {subclass}")

        await store.save(campaign)
        return "\n".join(lines)

    return [level_up]
