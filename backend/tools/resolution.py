"""Atomic dice-resolution tools: roll + apply the result in one call, instead of
the LLM sequencing several low-level roll/apply/advance calls itself. Not
combat.py-scoped — resolve_check applies outside combat too (skill/ability
checks in exploration and social scenes)."""

import random

from langchain_core.tools import BaseTool, tool

from backend.models import (
    Character, ConditionType, DamageType, PendingAction, Skill, SKILL_ABILITY,
    SpellResolutionType, SpellSlotLevel,
)
from backend.stores.campaign_store import CampaignStore
from backend.tools._helpers import (
    advance_combatant_turn, apply_damage_to_character, apply_damage_to_monster,
    critical_damage_notation, find_char, find_monster, has_plausible_reaction,
    require_current_turn, roll_notation,
)

_ABILITIES = {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"}

_FOCUS_KEYWORDS = ("focus", "pouch", "holy symbol")


def _material_requirement(components: list[str]) -> str | None:
    """The material-component descriptor (e.g. "M (a pinch of sand)") if the
    spell has one, else None. Spell.components is free-form text like
    ["V", "S", "M (a pinch of sand)"] — no structured cost/consumed fields."""
    return next((c for c in components if c.strip().upper().startswith("M")), None)


def _has_focus_or_pouch(caster: Character) -> bool:
    """A spellcasting focus (arcane focus, holy symbol, druidic focus) or a
    component pouch substitutes for any material component with no listed
    gp cost (PHB) — this only checks whether the caster has ANY item that
    could serve that role, not whether it matches their specific class
    (already chosen at chargen/companion creation, not re-validated here)."""
    return any(
        any(k in item.name.lower() for k in _FOCUS_KEYWORDS)
        for item in caster.inventory
    )


def _has_named_material(caster: Character, material: str) -> bool:
    """A costly material component (a gp cost appears in its own
    description, e.g. "a diamond worth 300gp, which the spell consumes")
    can't be substituted by a focus/pouch — the caster needs that actual
    item. Best-effort match: strip an item name's parenthetical/cost suffix
    (e.g. "Diamond (300gp)" -> "diamond") and check that core word appears
    in the material's description — not a verbatim substring match, since
    an item's own naming convention (parens, cost) rarely matches the
    spell text's phrasing word-for-word. Still skip-on-doubt, not fuzzy
    NLP: if no item's core name shows up at all, refuse and let the
    DM/player decide, same principle as add_headers.py's heading
    classifier."""
    desc = material.lower()
    for item in caster.inventory:
        core = item.name.split("(")[0].strip().lower()
        if core and core in desc:
            return True
    return False


def _roll_d20(advantage: str) -> tuple[int, str]:
    """Roll 1d20, or 2d20 keeping the higher/lower for advantage/disadvantage.
    Returns (picked_face, breakdown). Only the picked die's face determines
    crit/fumble on an attack roll — the unchosen die is irrelevant to that."""
    adv = advantage.strip().lower()
    if adv not in ("advantage", "disadvantage"):
        adv = "none"
    if adv == "none":
        face = random.randint(1, 20)
        return face, f"{face}"
    a, b = random.randint(1, 20), random.randint(1, 20)
    picked = max(a, b) if adv == "advantage" else min(a, b)
    return picked, f"{picked} ({adv}: rolled {a}, {b})"


def _roll_to_hit(target_ac: int, to_hit_bonus: int, advantage: str) -> tuple[bool, bool, int, str]:
    """Roll one attack roll (weapon or spell). Returns (hit, is_crit, total,
    header_line) — deliberately doesn't roll damage, since a hit against a
    reaction-eligible target must pause BEFORE damage is rolled."""
    face, roll_desc = _roll_d20(advantage)
    total = face + to_hit_bonus
    is_nat20 = face == 20
    is_nat1 = face == 1
    hit = is_nat20 or (not is_nat1 and total >= target_ac)
    header = f"d20 {roll_desc} + {to_hit_bonus} = {total} vs AC {target_ac}"
    return hit, is_nat20, total, header


async def resolve_pending_action_impl(
    campaign,
    reaction_declared: str = "",
    ac_bonus: int = 0,
    damage_reduction: int = 0,
    damage_multiplier: float = 1.0,
) -> str:
    """Core logic for finishing a paused attack — factored out of the
    @tool-decorated resolve_pending_action so dm_agent.py's stale-pending
    auto-decline (mechanics_node) can call the exact same resolution instead
    of duplicating it. Takes an already-loaded `campaign`; does not save —
    callers own the store.save() call. Mutates `campaign` in place."""
    enc = campaign.active_encounter
    if not enc or not enc.pending_action:
        return "No pending action to resolve."
    pending = enc.pending_action
    target = find_char(campaign, pending.target_name) or find_monster(campaign, pending.target_name)
    if not target:
        enc.pending_action = None
        return f"Target '{pending.target_name}' no longer found — pending action cleared."

    lines = []
    new_ac = target.ac + ac_bonus
    still_hits = pending.was_natural_20 or pending.to_hit_total >= new_ac
    if not still_hits:
        lines.append(
            f"{pending.attacker_name}'s {pending.attack_name} vs {pending.target_name}: "
            f"reaction ({reaction_declared or 'declined'}) raises AC to {new_ac}, "
            f"attack total {pending.to_hit_total} now MISSES."
        )
    else:
        if pending.pending_damage_notation:
            dmg_total, dmg_breakdown = roll_notation(pending.pending_damage_notation)
        else:
            dmg_total, dmg_breakdown = 0, "no damage"
        dmg_total = max(0, dmg_total - damage_reduction)
        dmg_total = int(dmg_total * damage_multiplier)
        try:
            dtype = DamageType(pending.damage_type)
        except ValueError:
            dtype = DamageType.FORCE
        hp_msg = apply_damage_to_character(target, -dmg_total, is_critical=pending.was_natural_20) if isinstance(target, Character) \
            else apply_damage_to_monster(target, -dmg_total)
        reaction_note = f", reaction ({reaction_declared}) applied" if reaction_declared else ""
        lines.append(
            f"{pending.attacker_name}'s {pending.attack_name} vs {pending.target_name}: "
            f"damage {dmg_breakdown} {dtype.value}{reaction_note} — {hp_msg}"
        )

    if reaction_declared and isinstance(target, Character):
        target.reaction_available = False

    if pending.remaining_swings > 0:
        attacker = find_char(campaign, pending.attacker_name) or find_monster(campaign, pending.attacker_name)
        atk = next((a for a in attacker.attacks if a.name.lower() == pending.attack_name.lower()), None) if attacker else None
        if atk:
            for _ in range(pending.remaining_swings):
                hit, is_crit, total, header = _roll_to_hit(target.ac, atk.to_hit_bonus, "none")
                label = f"{pending.attacker_name} — {atk.name} vs {pending.target_name}: {header}"
                if not hit:
                    lines.append(f"{label} — MISS")
                    continue
                swing_dice = critical_damage_notation(atk.damage_dice) if is_crit else atk.damage_dice
                dmg_total, dmg_breakdown = roll_notation(swing_dice)
                hp_msg = apply_damage_to_character(target, -dmg_total, is_critical=is_crit) if isinstance(target, Character) \
                    else apply_damage_to_monster(target, -dmg_total)
                lines.append(
                    f"{label} — HIT{' (CRITICAL)' if is_crit else ''}, "
                    f"damage {dmg_breakdown} {atk.damage_type.value} — {hp_msg}"
                )

    if pending.attacker_wanted_end_turn:
        lines.append(advance_combatant_turn(campaign, enc))

    enc.pending_action = None
    return "\n".join(lines)


def _save_bonus(target, ability: str) -> int:
    """Save bonus for a Character (ability mod + proficiency if listed) or
    Monster (saving_throw_bonuses override if listed, else raw ability mod —
    matches how a real Monster Manual stat block only lists saves the creature
    is actually proficient in)."""
    ab = target.ability_scores
    mod = ab.modifier(getattr(ab, ability))
    if isinstance(target, Character):
        profs = target.saving_throw_proficiencies
        if ability in profs or ability[:3] in profs:
            mod += target.proficiency_bonus
        return mod
    override = target.saving_throw_bonuses.get(ability, target.saving_throw_bonuses.get(ability[:3]))
    return override if override is not None else mod


def make_tools(campaign_id: str, store: CampaignStore) -> list[BaseTool]:

    @tool
    async def resolve_attack(
        attacker_name: str,
        target_name: str,
        attack_name: str = "",
        spell_name: str = "",
        damage_dice_override: str = "",
        damage_type_override: str = "",
        advantage: str = "none",
        attack_count: int = 1,
        end_turn: bool = False,
    ) -> str:
        """Resolve one or more attacks (roll to-hit, apply crit/fumble rules,
        roll and apply damage) in a single call — the primary way to resolve
        any weapon or improvised-spell attack. Never roll to-hit and damage as
        separate roll_dice calls, and never apply the result with a separate
        update_character_hp/update_monster_hp call for an attack resolved here.

        Sourcing the attack's to-hit bonus and damage (in order):
        - attack_name: a named entry in the attacker's own attacks list
          (case-insensitive).
        - spell_name (no attack_name): an IMPROVISED spell attack — uses the
          attacker's spell_attack_bonus; damage_dice_override AND
          damage_type_override are REQUIRED (ground them via search_rules
          first). Prefer cast_spell instead whenever the caster's spells_known
          has a structured entry for this spell — it grounds the roll with no
          overrides needed.
        - neither given: falls back to the attacker's only attack, if they
          have exactly one.

        attack_count: for a Multiattack, resolve this many swings in ONE call,
        ALL against target_name (table rule — no split-targeting in a single
        call; call this again for a second target). Damage from all hits is
        summed into one HP update. Defaults to 1.

        advantage: "none" | "advantage" | "disadvantage".

        end_turn: pass True only when this is the LAST thing the attacking
        combatant does this turn (no bonus-action attack or extra Hasted
        action still coming) — folds in the same turn-advancement
        advance_initiative does. Leave False and call advance_initiative
        yourself once the combatant's whole turn is actually done.

        If the target is a player-controlled character with a real reaction
        option available (a prepared reaction spell with a remaining slot, or
        a reaction feature like Uncanny Dodge/Parry) and this attack would
        hit, resolution PAUSES instead of applying damage — the result says
        PENDING. Stop calling any more tools this turn when you see that; it
        continues later via resolve_pending_action.
        """
        campaign = await store.load(campaign_id)
        attacker = find_char(campaign, attacker_name) or find_monster(campaign, attacker_name)
        if not attacker:
            return f"No character or monster named '{attacker_name}' found."
        if isinstance(attacker, Character) and attacker.current_hp == 0:
            return f"{attacker_name} is unconscious at 0 HP and cannot act — call resolve_death_save for their turn instead."
        turn_issue = require_current_turn(campaign, attacker_name)
        if turn_issue:
            return turn_issue
        target = find_char(campaign, target_name) or find_monster(campaign, target_name)
        if not target:
            return f"No character or monster named '{target_name}' found."

        enc = campaign.active_encounter
        if enc and enc.pending_action:
            return (
                "Refusing — there's already a pending reaction on this encounter. "
                "Resolve it with resolve_pending_action before making another attack."
            )

        if attack_name:
            atk = next((a for a in attacker.attacks if a.name.lower() == attack_name.lower()), None)
            if not atk:
                names = ", ".join(a.name for a in attacker.attacks) or "(none)"
                return f"'{attacker_name}' has no attack named '{attack_name}'. Known attacks: {names}."
            to_hit_bonus, damage_dice, damage_type, display_name = atk.to_hit_bonus, atk.damage_dice, atk.damage_type, atk.name
        elif spell_name:
            if not isinstance(attacker, Character):
                return f"'{attacker_name}' is a monster — give it a real attack via create_monster instead of spell_name."
            if not damage_dice_override or not damage_type_override:
                return (
                    "An improvised spell attack needs both damage_dice_override and "
                    "damage_type_override (ground them via search_rules first) — or "
                    "use cast_spell if this spell has structured data in spells_known."
                )
            to_hit_bonus = attacker.spell_attack_bonus or 0
            damage_dice = damage_dice_override
            try:
                damage_type = DamageType(damage_type_override.lower())
            except ValueError:
                damage_type = DamageType.FORCE
            display_name = spell_name
        else:
            if len(attacker.attacks) != 1:
                names = ", ".join(a.name for a in attacker.attacks) or "(none)"
                return f"'{attacker_name}' has {len(attacker.attacks)} attacks — pass attack_name explicitly. Known: {names}."
            atk = attacker.attacks[0]
            to_hit_bonus, damage_dice, damage_type, display_name = atk.to_hit_bonus, atk.damage_dice, atk.damage_type, atk.name

        target_ac = target.ac
        attack_count = max(1, min(8, attack_count))
        eligible_for_pause = (
            isinstance(target, Character) and target.is_player_controlled
            and target.current_hp > 0  # unconscious creatures can't take reactions
            and target.reaction_available and has_plausible_reaction(target)
            and enc is not None and enc.is_active
        )

        swing_lines: list[str] = []
        total_damage = 0
        hits = 0
        any_crit = False
        for swing_idx in range(1, attack_count + 1):
            hit, is_crit, total, header = _roll_to_hit(target_ac, to_hit_bonus, advantage)
            label = f"{attacker_name} — {display_name} vs {target_name}: {header}"
            if not hit:
                swing_lines.append(f"{label} — MISS")
                continue

            swing_dice = critical_damage_notation(damage_dice) if is_crit else damage_dice

            if eligible_for_pause:
                remaining = attack_count - swing_idx
                enc.pending_action = PendingAction(
                    attacker_name=attacker_name,
                    target_name=target_name,
                    attack_name=display_name,
                    to_hit_total=total,
                    was_natural_20=is_crit,
                    target_ac_at_time=target_ac,
                    pending_damage_notation=swing_dice,
                    damage_type=damage_type.value,
                    remaining_swings=remaining,
                    attacker_wanted_end_turn=end_turn,
                    prompt_note=(
                        f"{attacker_name}'s {display_name} hit {target_name} (total "
                        f"{total} vs AC {target_ac}). {target_name}'s player may react "
                        f"before damage is applied — call resolve_pending_action once "
                        f"they've decided."
                    ),
                )
                await store.save(campaign)
                prior = ("\n".join(swing_lines) + "\n") if swing_lines else ""
                return (
                    f"{prior}{label} — HIT{' (CRITICAL)' if is_crit else ''}. PENDING — "
                    f"{target_name} has a reaction available. Stop calling tools this "
                    f"turn; do not apply damage or narrate the outcome yet."
                )

            dmg_total, dmg_breakdown = roll_notation(swing_dice)
            total_damage += dmg_total
            hits += 1
            any_crit = any_crit or is_crit
            swing_lines.append(
                f"{label} — HIT{' (CRITICAL)' if is_crit else ''}, damage {dmg_breakdown} {damage_type.value}"
            )

        result_lines = list(swing_lines)
        if hits:
            hp_msg = apply_damage_to_character(target, -total_damage, is_critical=any_crit) if isinstance(target, Character) \
                else apply_damage_to_monster(target, -total_damage)
            result_lines.append(hp_msg)
        if end_turn and enc and enc.is_active:
            result_lines.append(advance_combatant_turn(campaign, enc))

        await store.save(campaign)
        return "\n".join(result_lines)

    @tool
    async def resolve_saving_throw(
        target_names: list[str],
        ability: str,
        dc: int,
        on_fail_damage_dice: str = "",
        on_fail_damage_type: str = "",
        half_on_success: bool = False,
        condition_on_fail: str = "",
        end_turn: bool = False,
    ) -> str:
        """Resolve a saving throw for one or more targets in ONE call — the
        primary way to resolve any save-based effect (a trap, a spell save, an
        environmental hazard), including an AoE hitting several targets at
        once. Never roll saves one at a time with separate roll_dice calls
        when several targets are affected by the same effect. A target name
        that isn't found is skipped (reported inline) without dropping the
        rest of the group. No crit/fumble semantics — that's attack-roll-only.

        ability: strength, dexterity, constitution, intelligence, wisdom, or charisma.
        on_fail_damage_dice/on_fail_damage_type: optional — if the effect
        deals damage on a failed save, provide both (type defaults to force
        if dice are given but type is left blank).
        half_on_success: if true, a target who succeeds still takes half the
        on_fail_damage (rounded down) instead of none.
        condition_on_fail: optional condition name applied on a failed save.
        end_turn: pass True only when this is the last thing the acting
        combatant does this turn.
        """
        campaign = await store.load(campaign_id)
        ability = ability.strip().lower()
        if ability not in _ABILITIES:
            return f"'{ability}' is not a valid ability — use one of: {', '.join(sorted(_ABILITIES))}."

        dtype = None
        if on_fail_damage_dice:
            try:
                dtype = DamageType(on_fail_damage_type.lower()) if on_fail_damage_type else DamageType.FORCE
            except ValueError:
                dtype = DamageType.FORCE

        cond = None
        if condition_on_fail:
            try:
                cond = ConditionType(condition_on_fail.lower())
            except ValueError:
                cond = None

        lines = []
        for name in target_names:
            target = find_char(campaign, name) or find_monster(campaign, name)
            if not target:
                lines.append(f"{name}: not found — skipped.")
                continue

            bonus = _save_bonus(target, ability)
            face, roll_desc = _roll_d20("none")
            total = face + bonus
            success = total >= dc
            line = f"{name} — {ability} save: d20 {roll_desc} + {bonus} = {total} vs DC {dc} — {'SUCCESS' if success else 'FAILURE'}"

            if dtype is not None and on_fail_damage_dice and (not success or half_on_success):
                dmg_total, dmg_breakdown = roll_notation(on_fail_damage_dice)
                if success:
                    dmg_total //= 2
                if dmg_total > 0:
                    hp_msg = apply_damage_to_character(target, -dmg_total) if isinstance(target, Character) \
                        else apply_damage_to_monster(target, -dmg_total)
                    line += f", damage {dmg_breakdown}" + (" (halved)" if success else "") + f" {dtype.value} — {hp_msg}"

            if cond is not None and not success and cond not in target.conditions:
                target.conditions.append(cond)
                line += f" — {cond.value} applied"

            lines.append(line)

        enc = campaign.active_encounter
        if end_turn and enc and enc.is_active:
            lines.append(advance_combatant_turn(campaign, enc))

        await store.save(campaign)
        return "\n".join(lines)

    @tool
    async def resolve_check(
        character_name: str,
        ability_or_skill: str,
        dc: int | None = None,
        advantage: str = "none",
    ) -> str:
        """Resolve an ability check or skill check in ONE call — rolls d20
        plus the character's real modifier (ability score, proficiency,
        expertise) looked up from their sheet automatically, instead of a
        separate get_character call plus a manually-computed roll_dice call.
        No crit/fumble semantics — a natural 20/1 has no special effect on an
        ability check in 5e, only on attack rolls.

        ability_or_skill: a raw ability name (strength, dexterity, ...) or a
        skill name (perception, stealth, sleight of hand, ...).
        dc: optional — if given, the result also reports pass/fail.
        """
        campaign = await store.load(campaign_id)
        target = find_char(campaign, character_name) or find_monster(campaign, character_name)
        if not target:
            return f"No character or monster named '{character_name}' found."
        if isinstance(target, Character) and target.current_hp == 0:
            return f"{character_name} is unconscious at 0 HP and incapacitated — cannot take checks or actions."

        key = ability_or_skill.strip().lower().replace(" ", "_")
        ab = target.ability_scores

        try:
            skill = Skill(key)
        except ValueError:
            skill = None

        if skill is not None:
            ability_name = SKILL_ABILITY[skill]
            mod = ab.modifier(getattr(ab, ability_name))
            if isinstance(target, Character):
                if skill in target.skill_expertise:
                    mod += target.proficiency_bonus * 2
                elif skill in target.skill_proficiencies:
                    mod += target.proficiency_bonus
            else:
                override = target.skill_bonuses.get(key)
                if override is not None:
                    mod = override
            label = skill.value
        elif key in _ABILITIES:
            mod = ab.modifier(getattr(ab, key))
            label = key
        else:
            return f"'{ability_or_skill}' is not a recognized ability or skill."

        face, roll_desc = _roll_d20(advantage)
        total = face + mod
        line = f"{character_name} — {label} check: d20 {roll_desc} + {mod} = {total}"
        if dc is not None:
            line += f" vs DC {dc} — {'SUCCESS' if total >= dc else 'FAILURE'}"
        return line

    @tool
    async def resolve_death_save(character_name: str) -> str:
        """Roll a death saving throw for a character at 0 HP. This is the
        ONLY legal action for a combatant whose turn comes up while they're
        down — resolve_attack/cast_spell/resolve_check all refuse to act for
        a character at 0 HP and point back here instead. No modifiers: d20,
        10+ is a success, below 10 is a failure, a natural 1 counts as two
        failures. A natural 20 immediately regains consciousness with 1 HP
        (clearing the tally). 3 successes stabilizes them (still unconscious
        at 0 HP, but stops needing saves, until healed). 3 failures kills
        them. Damage taken while already down is handled automatically by
        apply_damage_to_character (via resolve_attack/resolve_saving_throw/
        resolve_pending_action) as its own failure(s), not through this tool —
        this one is only for the start-of-turn roll when nothing else
        triggered a save this round."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' found."
        if char.current_hp > 0:
            return f"{character_name} is not down (currently {char.current_hp} HP) — no death save needed."
        if char.death_save_successes >= 3 or char.death_save_failures >= 3:
            status = "stable" if char.death_save_successes >= 3 else "dead"
            return f"{character_name} is already {status} — no further death saves needed."

        face = random.randint(1, 20)
        if face == 20:
            char.current_hp = 1
            char.death_save_successes = 0
            char.death_save_failures = 0
            await store.save(campaign)
            return f"{character_name} — death save: d20 20 — NATURAL 20, regains consciousness with 1 HP!"

        if face == 1:
            char.death_save_failures = min(3, char.death_save_failures + 2)
            result = "FAILURE (natural 1, counts as 2)"
        elif face >= 10:
            char.death_save_successes = min(3, char.death_save_successes + 1)
            result = "SUCCESS"
        else:
            char.death_save_failures = min(3, char.death_save_failures + 1)
            result = "FAILURE"

        status = ""
        if char.death_save_failures >= 3:
            status = " — DIES (3 failures)"
        elif char.death_save_successes >= 3:
            status = " — STABLE (3 successes, still unconscious at 0 HP until healed)"

        await store.save(campaign)
        return (
            f"{character_name} — death save: d20 {face} — {result}. "
            f"Tally: {char.death_save_successes} successes, {char.death_save_failures} failures.{status}"
        )

    @tool
    async def resolve_pending_action(
        reaction_declared: str = "",
        ac_bonus: int = 0,
        damage_reduction: int = 0,
        damage_multiplier: float = 1.0,
    ) -> str:
        """Finish resolving an attack that paused for a reaction (see the
        PENDING result from resolve_attack/cast_spell, and the ⚠ PENDING
        REACTION line in the live encounter state). Call once the target's
        player has decided whether to react.

        reaction_declared: name/flavor of the reaction used, e.g. "Shield" —
        leave empty if the player declines (declining does not spend their
        reaction, so a later attack the same round can still offer one).
        ac_bonus: Shield-style — added to the target's AC and re-checked
        against the already-rolled attack total. Cannot turn a natural 20 into
        a miss.
        damage_reduction: Parry-style flat reduction, applied before the multiplier.
        damage_multiplier: Uncanny-Dodge-style, e.g. 0.5 to halve, 0.0 to negate.

        If this attack was part of a Multiattack, any remaining swings resolve
        now too (no further pause — a reaction is spent or declined once per
        round). If the original attack asked to end the turn, this call
        advances initiative for you.
        """
        campaign = await store.load(campaign_id)
        result = await resolve_pending_action_impl(
            campaign, reaction_declared, ac_bonus, damage_reduction, damage_multiplier,
        )
        await store.save(campaign)
        return result

    @tool
    async def cast_spell(
        caster_name: str,
        spell_name: str,
        target_names: list[str] | None = None,
        slot_level: int | None = None,
        advantage: str = "none",
        end_turn: bool = False,
        as_ritual: bool = False,
    ) -> str:
        """Cast a known spell in ONE call — consumes the spell slot, resolves
        the roll (attack, save, or automatic effect) using the spell's own
        stored data, and applies the result, all atomically. Prefer this over
        resolve_attack's spell_name fallback whenever the caster's
        spells_known has this spell — only cast_spell grounds the roll with
        zero model-supplied numbers, and it consumes the slot for you.

        target_names: required for an attack-roll or saving-throw spell
        (attack-roll uses only the first name); optional for an automatic
        spell — a self-targeting heal/utility defaults to the caster.
        slot_level: upcast by passing a slot level higher than the spell's own
        level; defaults to the spell's own level. Not compatible with
        as_ritual — a ritual cast is always at the spell's own level.
        end_turn: pass True only when this is the last thing the caster does
        this turn.
        as_ritual: cast a Spell.ritual=True spell as a Ritual per the 2024
        PHB's general rule — takes 10 minutes longer in fiction but spends no
        spell slot. Only pass this once the player has actually chosen to
        ritual-cast (see the mechanics prompt's guidance on offering it, not
        assuming it) — never silently default to it. Refuses for a cantrip
        (already slot-free), a non-ritual spell, or a spell that isn't
        currently prepared (the PHB's own requirement for ritual casting).

        If spells_known/casting stats aren't populated for this character
        (true for every character today — spell data population is a
        separate, not-yet-done pass), this fails with a clear message — use
        resolve_attack (spell_name + overrides) or resolve_saving_throw
        directly instead for an improvised spell effect.
        """
        campaign = await store.load(campaign_id)
        caster = find_char(campaign, caster_name)
        if not caster:
            return (
                f"No character named '{caster_name}' found — cast_spell only supports "
                f"PC/companion casters (a monster's spell-like attacks are baked into "
                f"its attacks list by create_monster instead)."
            )
        if caster.current_hp == 0:
            return f"{caster_name} is unconscious at 0 HP and cannot act — call resolve_death_save for their turn instead."
        turn_issue = require_current_turn(campaign, caster_name)
        if turn_issue:
            return turn_issue

        spell = next((s for s in caster.spells_known if s.name.lower() == spell_name.lower()), None)
        if not spell:
            return (
                f"'{caster_name}' has no known spell called '{spell_name}' (spells_known "
                f"is empty or doesn't include it — spell data population is a separate, "
                f"not-yet-done pass). Use resolve_attack (spell_name + damage overrides) "
                f"or resolve_saving_throw directly instead."
            )
        if caster.spells_prepared and spell.name not in caster.spells_prepared:
            return f"'{spell_name}' is known but not currently prepared."

        material = _material_requirement(spell.components)
        if material:
            costly = "gp" in material.lower()
            if costly and not _has_named_material(caster, material):
                return (
                    f"'{spell.name}' requires a specific material component ({material}) "
                    f"that a spellcasting focus or component pouch can't substitute for — "
                    f"{caster_name}'s inventory doesn't have it. Cast fails, no slot consumed."
                )
            if not costly and not _has_focus_or_pouch(caster):
                return (
                    f"'{spell.name}' requires a material component ({material}) and "
                    f"{caster_name} has no spellcasting focus or component pouch in "
                    f"inventory to substitute for it. Cast fails, no slot consumed."
                )

        if as_ritual:
            if spell.level == 0:
                return f"'{spell.name}' is a cantrip — it never uses a slot, ritual casting doesn't apply."
            if not spell.ritual:
                return f"'{spell.name}' isn't a ritual spell — cast it normally (as_ritual=False) instead."
            if slot_level is not None and slot_level != spell.level:
                return "A ritual cast can't be upcast — omit slot_level (or pass as_ritual=False to upcast normally)."

        target_names = target_names or []
        enc = campaign.active_encounter
        cast_level = slot_level or spell.level
        if spell.level > 0 and not as_ritual:
            slot = caster.spell_slots.get(cast_level)
            if not slot or slot.remaining <= 0:
                return f"'{caster_name}' has no remaining level {cast_level} spell slots."
            caster.spell_slots[cast_level] = SpellSlotLevel(max=slot.max, used=slot.used + 1)

        upcast_note = f" (upcast to level {cast_level})" if cast_level != spell.level else ""
        ritual_note = " as a Ritual (10 extra minutes, no spell slot spent)" if as_ritual else ""
        lines = [f"{caster_name} casts {spell.name}{upcast_note}{ritual_note}."]

        if spell.resolution_type == SpellResolutionType.ATTACK_ROLL:
            if not target_names:
                return "This is an attack-roll spell — target_names needs at least one name."
            target_name = target_names[0]
            target = find_char(campaign, target_name) or find_monster(campaign, target_name)
            if not target:
                return f"No character or monster named '{target_name}' found."
            target_ac = target.ac
            to_hit_bonus = caster.spell_attack_bonus or 0
            eligible_for_pause = (
                isinstance(target, Character) and target.is_player_controlled
                and target.current_hp > 0  # unconscious creatures can't take reactions
                and target.reaction_available and has_plausible_reaction(target)
                and enc is not None and enc.is_active
            )

            hit, is_crit, total, header = _roll_to_hit(target_ac, to_hit_bonus, advantage)
            label = f"{caster_name} — {spell.name} vs {target_name}: {header}"
            if not hit:
                lines.append(f"{label} — MISS")
            else:
                swing_dice = critical_damage_notation(spell.effect_dice) if (is_crit and spell.effect_dice) else spell.effect_dice
                if eligible_for_pause:
                    enc.pending_action = PendingAction(
                        attacker_name=caster_name,
                        target_name=target_name,
                        attack_name=spell.name,
                        to_hit_total=total,
                        was_natural_20=is_crit,
                        target_ac_at_time=target_ac,
                        pending_damage_notation=swing_dice,
                        damage_type=(spell.damage_type.value if spell.damage_type else "force"),
                        remaining_swings=0,
                        attacker_wanted_end_turn=end_turn,
                        prompt_note=(
                            f"{caster_name}'s {spell.name} hit {target_name} (total "
                            f"{total} vs AC {target_ac}). {target_name}'s player may "
                            f"react before damage is applied — call resolve_pending_action."
                        ),
                    )
                    await store.save(campaign)
                    lines.append(
                        f"{label} — HIT{' (CRITICAL)' if is_crit else ''}. PENDING — "
                        f"{target_name} has a reaction available. Stop calling tools "
                        f"this turn; do not apply damage or narrate the outcome yet."
                    )
                    return "\n".join(lines)
                elif spell.effect_dice:
                    dmg_total, dmg_breakdown = roll_notation(swing_dice)
                    hp_msg = apply_damage_to_character(target, -dmg_total, is_critical=is_crit) if isinstance(target, Character) \
                        else apply_damage_to_monster(target, -dmg_total)
                    dtype_str = spell.damage_type.value if spell.damage_type else "force"
                    lines.append(f"{label} — HIT{' (CRITICAL)' if is_crit else ''}, damage {dmg_breakdown} {dtype_str} — {hp_msg}")
                else:
                    lines.append(f"{label} — HIT{' (CRITICAL)' if is_crit else ''} (no damage).")

        elif spell.resolution_type == SpellResolutionType.SAVING_THROW:
            if not target_names:
                return "This is a saving-throw spell — target_names needs at least one name."
            if not spell.save_ability:
                return f"'{spell.name}' has no save_ability set — data incomplete, resolve manually via resolve_saving_throw."
            dc = caster.spell_save_dc or 0
            dtype_str = spell.damage_type.value if spell.damage_type else "force"
            for name in target_names:
                target = find_char(campaign, name) or find_monster(campaign, name)
                if not target:
                    lines.append(f"{name}: not found — skipped.")
                    continue
                bonus = _save_bonus(target, spell.save_ability)
                face, roll_desc = _roll_d20("none")
                total = face + bonus
                success = total >= dc
                line = f"{name} — {spell.save_ability} save: d20 {roll_desc} + {bonus} = {total} vs DC {dc} — {'SUCCESS' if success else 'FAILURE'}"
                if spell.effect_dice and (not success or spell.half_damage_on_success):
                    dmg_total, dmg_breakdown = roll_notation(spell.effect_dice)
                    if success:
                        dmg_total //= 2
                    if dmg_total > 0:
                        hp_msg = apply_damage_to_character(target, -dmg_total) if isinstance(target, Character) \
                            else apply_damage_to_monster(target, -dmg_total)
                        line += f", damage {dmg_breakdown}" + (" (halved)" if success else "") + f" {dtype_str} — {hp_msg}"
                if spell.condition_on_fail and not success:
                    try:
                        cond = ConditionType(spell.condition_on_fail.lower())
                        if cond not in target.conditions:
                            target.conditions.append(cond)
                            line += f" — {cond.value} applied"
                    except ValueError:
                        pass
                lines.append(line)

        else:  # AUTOMATIC
            if spell.effect_dice:
                recipients = target_names or [caster_name]
                for name in recipients:
                    target = find_char(campaign, name) or find_monster(campaign, name)
                    if not target:
                        lines.append(f"{name}: not found — skipped.")
                        continue
                    dmg_total, dmg_breakdown = roll_notation(spell.effect_dice)
                    amount = dmg_total if spell.is_healing else -dmg_total
                    hp_msg = apply_damage_to_character(target, amount) if isinstance(target, Character) \
                        else apply_damage_to_monster(target, amount)
                    kind = "healing" if spell.is_healing else (spell.damage_type.value if spell.damage_type else "damage")
                    lines.append(f"{name}: {dmg_breakdown} {kind} — {hp_msg}")
            else:
                lines.append(f"{spell.name} has no mechanical roll — narrate its established effect.")

        if end_turn and enc and enc.is_active:
            lines.append(advance_combatant_turn(campaign, enc))

        await store.save(campaign)
        return "\n".join(lines)

    return [
        resolve_attack, resolve_saving_throw, resolve_check, resolve_death_save,
        resolve_pending_action, cast_spell,
    ]
