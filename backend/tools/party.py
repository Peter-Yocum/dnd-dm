import re

from langchain_core.tools import tool

from backend.data.equipment import ARMOR, SHIELD_AC_BONUS, WEAPONS, weapon_reach_ft
from backend.models import ActiveEffect, Attack, ConditionType, Container, Currency, Item, SpellSlotLevel
from backend.rag.entity_resolution import find_candidate_matches
from backend.stores.campaign_store import CampaignStore
from backend.stores.graph_store import RelationGraphStore
from backend.stores.lore_store import LoreStore
from backend.tools._helpers import (
    all_campaign_item_names, apply_damage_to_character, apply_map_reveal_if_needed, char_summary,
    find_char, find_location, find_monster, with_ability_mod,
)
from backend.tools.loot_generator import ARMOR_BONUS_RARITY, WEAPON_LIKE_BONUS_RARITY


def _mark_loot_granted(campaign) -> None:
    """Flag the active encounter (if any) as having had loot manually granted
    mid-fight, so end_encounter's automatic roll (backend/tools/loot_generator.py)
    skips itself instead of paying out a second, unrelated pile on top."""
    enc = campaign.active_encounter
    if enc and enc.is_active:
        enc.loot_already_granted = True

# Matches an item_name that's actually currency mislabeled as a physical item
# ("Silver piece", "5 gp", "gold coins") — observed live: a mechanics model
# called add_item_to_character(item_name="Silver piece", quantity=5) instead
# of update_character_currency, landing 5 silver pieces in the inventory list
# with the Currency field left untouched at 0 gp.
_CURRENCY_ITEM_RE = re.compile(
    r"^\s*(?:gp|sp|cp|pp|ep)\s*$"
    r"|^\s*(?:gold|silver|copper|platinum|electrum)\s+pieces?\s*$"
    r"|^\s*(?:gold|silver|copper|platinum|electrum)\s+coins?\s*$",
    re.IGNORECASE,
)


def make_tools(
    campaign_id: str,
    store: CampaignStore,
    lore_store: LoreStore | None = None,
    books_in_play: list[str] | None = None,
    graph_store: RelationGraphStore | None = None,
) -> list:

    @tool
    async def get_party_status() -> str:
        """Get current HP, conditions, spell slots, and exhaustion for every party member.
        Call at the start of each session or whenever you need a party overview."""
        campaign = await store.load(campaign_id)
        if not campaign or not campaign.party:
            return "No party members found."
        lines = ["Party status:"] + [f"  {char_summary(c)}" for c in campaign.party]
        return "\n".join(lines)

    @tool
    async def get_character(character_name: str) -> str:
        """Get the full character sheet for one party member by name — ability scores,
        proficiencies, spells, inventory, features, and current status."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        ab = char.ability_scores
        lines = [
            f"=== {char.name} — {char.race} {char.char_class} {char.level} ===",
            f"HP: {char.current_hp}/{char.max_hp}  Temp HP: {char.temp_hp}  AC: {char.ac}  Speed: {char.speed}",
            f"STR {ab.strength}({ab.str_mod:+d})  DEX {ab.dexterity}({ab.dex_mod:+d})  CON {ab.constitution}({ab.con_mod:+d})",
            f"INT {ab.intelligence}({ab.int_mod:+d})  WIS {ab.wisdom}({ab.wis_mod:+d})  CHA {ab.charisma}({ab.cha_mod:+d})",
            f"Proficiency bonus: +{char.proficiency_bonus}  Passive Perception: {char.passive_perception}",
        ]
        if char.conditions:
            lines.append(f"Conditions: {', '.join(c.value for c in char.conditions)}")
        if char.exhaustion_level:
            lines.append(f"Exhaustion: {char.exhaustion_level}")
        if char.concentration:
            lines.append(f"Concentrating on: {char.concentration}")
        if char.spell_slots:
            slot_parts = [
                f"L{lvl}:{s.max-s.used}/{s.max}"
                for lvl, s in sorted(char.spell_slots.items()) if s.max > 0
            ]
            lines.append(f"Spell slots: {', '.join(slot_parts)}")
        if char.spells_known:
            by_level: dict[int, list[str]] = {}
            for sp in char.spells_known:
                by_level.setdefault(sp.level, []).append(sp.name)
            for lvl in sorted(by_level):
                label = "Cantrips" if lvl == 0 else f"Level {lvl}"
                lines.append(f"  {label}: {', '.join(by_level[lvl])}")
        if char.attacks:
            lines.append("Attacks: " + ", ".join(
                f"{a.name} ({a.to_hit_bonus:+d} to hit, {a.damage_dice} {a.damage_type.value})"
                for a in char.attacks
            ))
        if char.inventory:
            lines.append("Inventory: " + ", ".join(
                f"{i.name}" + (f" x{i.quantity}" if i.quantity > 1 else "")
                for i in char.inventory
            ))
        if char.features:
            lines.append("Features: " + ", ".join(char.features))
        if char.notes:
            lines.append(f"Notes: {char.notes}")
        return "\n".join(lines)

    @tool
    async def update_character_hp(character_name: str, delta: int) -> str:
        """Apply damage (negative delta) or healing (positive delta) to a character.
        Damage hits temporary HP first before reducing current HP. Healing cannot
        exceed max HP and does not restore temp HP. Call immediately after any
        HP-changing event."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        msg = apply_damage_to_character(char, delta)
        await store.save(campaign)
        return msg

    @tool
    async def update_character_detail(character_name: str, field: str, value: str) -> str:
        """Update a cosmetic detail on an ALREADY-FINALIZED party member — the
        in-game counterpart to Session 0's update_character_draft, for a field
        a player forgot to set (or wants to change) after character creation.

        Supported fields: pronouns, appearance, alignment, notes.

        Deliberately narrow — mechanical fields (race, class, ability scores,
        equipment, ...) aren't editable here; those cascade into derived
        stats (HP, AC, spell slots, ...) that this tool doesn't recompute, and
        changing them mid-campaign is a DM judgment call beyond "fix a
        forgotten detail." Use CLEAR as the value to blank a field."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        editable = {"pronouns", "appearance", "alignment", "notes"}
        key = field.strip().lower()
        if key not in editable:
            return f"'{field}' isn't editable here — supported fields: {', '.join(sorted(editable))}."
        cleared = value.strip().upper() == "CLEAR"
        new_value = (None if key == "alignment" else "") if cleared else value
        setattr(char, key, new_value)
        await store.save(campaign)
        return f"{char.name}'s {key} set to: {new_value or '(cleared)'}"

    @tool
    async def add_condition(character_name: str, condition: str) -> str:
        """Apply a condition to a character or DM-controlled companion.
        Valid conditions: blinded, charmed, deafened, frightened, grappled,
        incapacitated, invisible, paralyzed, petrified, poisoned, prone,
        restrained, stunned, unconscious."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        try:
            cond = ConditionType(condition.lower())
        except ValueError:
            return f"'{condition}' is not a valid condition."
        if cond in char.conditions:
            return f"{char.name} already has {cond.value}."
        char.conditions.append(cond)
        await store.save(campaign)
        return f"{char.name} is now {cond.value}."

    @tool
    async def remove_condition(character_name: str, condition: str) -> str:
        """Remove a condition from a character once it ends."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        try:
            cond = ConditionType(condition.lower())
        except ValueError:
            return f"'{condition}' is not a valid condition."
        if cond not in char.conditions:
            return f"{char.name} does not have {cond.value}."
        char.conditions.remove(cond)
        await store.save(campaign)
        return f"{char.name} is no longer {cond.value}."

    @tool
    async def apply_effect(
        target_name: str,
        name: str,
        duration_rounds: int | None = None,
        extra_actions: int = 0,
        extra_bonus_actions: int = 0,
        extra_reactions: int = 0,
        ac_bonus: int = 0,
        attack_bonus: int = 0,
        save_bonus: int = 0,
        source: str = "",
        notes: str = "",
    ) -> str:
        """Apply a structured buff/effect (Haste, Bless, Action Surge, and
        similar) to a character or monster — the counterpart to add_condition
        for beneficial effects, and the ONLY way anything like "+1 action per
        turn" actually changes the turn budget resolve_attack/cast_spell
        enforce (see check_and_spend_action_budget). duration_rounds=None
        means it lasts until remove_effect is called (e.g. concentration
        broken, dispelled) rather than expiring on its own. Effects with an
        extra_actions/extra_bonus_actions bonus apply automatically every
        turn the buff is still active (reconciled in advance_combatant_turn)
        and automatically stop the moment duration_rounds reaches 0 or
        remove_effect is called — no need to re-apply it each round."""
        campaign = await store.load(campaign_id)
        target = find_char(campaign, target_name) or find_monster(campaign, target_name)
        if not target:
            return f"No character or monster named '{target_name}' found."
        effect = ActiveEffect(
            name=name, source=source, duration_rounds=duration_rounds,
            extra_actions=extra_actions, extra_bonus_actions=extra_bonus_actions,
            extra_reactions=extra_reactions, ac_bonus=ac_bonus, attack_bonus=attack_bonus,
            save_bonus=save_bonus, notes=notes,
        )
        target.active_effects.append(effect)
        await store.save(campaign)
        return f"{target.name} is now affected by {name}" + (f" ({duration_rounds} round(s))" if duration_rounds else " (until removed)") + "."

    @tool
    async def remove_effect(target_name: str, effect_name: str) -> str:
        """Remove an active effect from a character or monster early — a
        dispel, a broken concentration, or any other early end not covered by
        its own duration_rounds expiring naturally."""
        campaign = await store.load(campaign_id)
        target = find_char(campaign, target_name) or find_monster(campaign, target_name)
        if not target:
            return f"No character or monster named '{target_name}' found."
        effect = next((e for e in target.active_effects if e.name.lower() == effect_name.lower()), None)
        if not effect:
            return f"{target.name} is not affected by '{effect_name}'."
        target.active_effects.remove(effect)
        await store.save(campaign)
        return f"{effect_name} ends for {target.name}."

    @tool
    async def use_spell_slot(character_name: str, level: int) -> str:
        """Expend one spell slot of the given level for a character.
        Call whenever a character casts a levelled spell."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        slot = char.spell_slots.get(level)
        if slot is None:
            return f"{char.name} has no level {level} spell slots."
        if slot.used >= slot.max:
            return f"{char.name} has no remaining level {level} slots (0/{slot.max})."
        char.spell_slots[level] = SpellSlotLevel(max=slot.max, used=slot.used + 1)
        await store.save(campaign)
        remaining = slot.max - slot.used - 1
        return f"{char.name} used a level {level} slot. Remaining: {remaining}/{slot.max}."

    @tool
    async def restore_spell_slots(character_name: str) -> str:
        """Restore all spell slots after a long rest. Also clears exhaustion by one
        level and resets death saves. Call after the party takes a long rest."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        for lvl, slot in char.spell_slots.items():
            char.spell_slots[lvl] = SpellSlotLevel(max=slot.max, used=0)
        char.exhaustion_level = max(0, char.exhaustion_level - 1)
        char.death_save_successes = 0
        char.death_save_failures = 0
        await store.save(campaign)
        return f"{char.name}'s spell slots restored. Exhaustion now {char.exhaustion_level}."

    @tool
    async def add_item_to_character(
        character_name: str, item_name: str, quantity: int = 1, force: bool = False,
        map_of_location: str = "",
    ) -> str:
        """Add an item to a character's inventory. Use when a character picks up,
        purchases, or is given an item. Refuses if item_name is actually currency
        (e.g. "Silver piece", "gp") — call update_character_currency instead. If
        a close-but-not-exact name match already exists somewhere in the
        campaign (in this campaign or the canon Lore Registry), returns a
        warning instead of adding a possible duplicate item under a slightly
        different name — call lookup_entity to check first, or pass
        force=True if this is genuinely a different item.

        map_of_location: pass an EXISTING location's name if this item is a
        map/chart of that place (e.g. "the party buys a hand-drawn map of
        the sewers") — unlocks that location in the Maps browser without
        the party needing to physically visit it. Leave blank for an
        ordinary item."""
        if _CURRENCY_ITEM_RE.match(item_name):
            return (
                f"'{item_name}' is currency, not a physical item — call "
                f"update_character_currency instead so it lands in {character_name}'s "
                f"actual gold total, not a fake inventory entry."
            )
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        map_location_id = None
        if map_of_location:
            map_loc = find_location(campaign, map_of_location)
            if not map_loc:
                return f"No location named '{map_of_location}' found — call create_location first if it's new."
            map_location_id = map_loc.id
        existing = next((i for i in char.inventory if i.name.lower() == item_name.lower()), None)
        if not existing and not force:
            existing_names = all_campaign_item_names(campaign)
            if lore_store is not None:
                existing_names += await lore_store.find_candidates(books_in_play or [], "item")
            matches = find_candidate_matches(item_name, existing_names)
            if matches:
                return (
                    f"'{item_name}' is a close match to existing item(s): {', '.join(matches)}. "
                    f"Call lookup_entity('{item_name}') to check first, or call "
                    f"add_item_to_character again with force=True if this is genuinely a different item."
                )
        if existing:
            existing.quantity += quantity
            new_item = existing
        else:
            new_item = Item(
                name=item_name, quantity=quantity,
                is_map=bool(map_location_id), map_location_id=map_location_id,
            )
            char.inventory.append(new_item)
        apply_map_reveal_if_needed(campaign, new_item)
        _mark_loot_granted(campaign)
        await store.save(campaign)
        if graph_store is not None:
            await graph_store.add_edge(
                campaign_id, "character", char.id, char.name, "item", new_item.id, new_item.name, "owns",
            )
        return f"Added {quantity}x {item_name} to {char.name}'s inventory." + (
            f" Unlocked '{map_of_location}' in the Maps browser." if map_location_id else ""
        )

    @tool
    async def remove_item_from_character(
        character_name: str, item_name: str, quantity: int = 1
    ) -> str:
        """Remove an item from a character's inventory. Use when an item is sold,
        consumed, or lost. Removes the inventory entry entirely once its quantity
        reaches zero."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        existing = next((i for i in char.inventory if i.name.lower() == item_name.lower()), None)
        if not existing:
            return f"{char.name} doesn't have {item_name}."
        removed = min(quantity, existing.quantity)
        existing.quantity -= removed
        if existing.quantity <= 0:
            char.inventory.remove(existing)
        await store.save(campaign)
        return f"Removed {removed}x {item_name} from {char.name}'s inventory."

    @tool
    async def update_character_currency(
        character_name: str, gp_delta: int, reason: str = ""
    ) -> str:
        """Adjust a character's gold. Use a positive gp_delta for gold gained
        (loot, quest reward, selling an item) and a negative gp_delta for gold
        spent (a purchase, a service, a bribe). Call this for every narrated
        transaction — never let a purchase happen without updating the
        character's gold. Refuses (without changing anything) if the character
        can't afford a negative delta."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        if gp_delta < 0 and char.currency.to_gp() < abs(gp_delta):
            return (
                f"{char.name} only has {char.currency.to_gp():.2f} gp worth of "
                f"currency — not enough to cover {abs(gp_delta)} gp."
            )
        prev = char.currency.gp
        char.currency.gp += gp_delta
        if gp_delta > 0:
            _mark_loot_granted(campaign)
        await store.save(campaign)
        msg = f"{char.name}'s gold: {prev} → {char.currency.gp} gp"
        if reason:
            msg += f" ({reason})"
        return msg

    @tool
    async def create_magic_item(
        character_name: str,
        item_name: str,
        base_item: str = "",
        bonus: int = 0,
        description: str = "",
        requires_attunement: bool = False,
        rarity: str = "",
        force: bool = False,
    ) -> str:
        """Give a character a special/magical item — found in a hoard, given as
        a reward, etc. If it's a magic weapon or armor variant (a "+1 Longsword",
        "+2 Chain Mail", "+1 Shield"), pass base_item as the exact name of a real
        weapon/armor (see search_rules for valid names — get_option_details is
        Session-0-only, not available mid-game) and bonus as the enchantment
        level — this grounds the
        item's stats in real base weapon/armor data instead of inventing them,
        and automatically adds a usable attack (for a weapon) or raises AC (for
        armor/a shield). For a wholly custom magic item with no weapon/armor
        equivalent (a wand, an amulet, a bag of holding), leave base_item empty
        and just describe it — it's added to inventory as flavor with no
        mechanical effect; narrate and resolve its powers reactively with the
        normal tools (update_character_hp, add_condition, etc.) if/when they
        trigger in play. If a close-but-not-exact name match already exists
        elsewhere in the campaign, returns a warning instead — call
        lookup_entity to check first, or pass force=True if this is genuinely
        a different item (e.g. a second +1 Longsword).

        rarity: 'common', 'uncommon', 'rare', 'very rare', 'legendary', or
        'artifact' — drives the item's color in the UI. Leave blank for a +N
        weapon/shield/armor variant (base_item + bonus set) — the real DMG
        scale is applied automatically (a +N weapon or shield is
        uncommon/rare/very rare for N=1/2/3; the same bonus on body armor is
        one tier higher, rare/very rare/legendary). Required for anything
        else (a wand, an amulet) — there's no formula to derive it from."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        if not force:
            existing_names = all_campaign_item_names(campaign)
            if lore_store is not None:
                existing_names += await lore_store.find_candidates(books_in_play or [], "item")
            matches = find_candidate_matches(item_name, existing_names)
            if matches:
                return (
                    f"'{item_name}' is a close match to existing item(s): {', '.join(matches)}. "
                    f"Call lookup_entity('{item_name}') to check first, or call "
                    f"create_magic_item again with force=True if this is genuinely a different item."
                )

        base_key = base_item.strip()
        weapon = WEAPONS.get(base_key) if base_key else None
        armor = ARMOR.get(base_key) if base_key else None
        is_shield = base_key.lower() == "shield"
        if not rarity and bonus in (1, 2, 3):
            scale = ARMOR_BONUS_RARITY if armor else WEAPON_LIKE_BONUS_RARITY
            rarity = scale[bonus]

        if weapon:
            is_ranged = "/" in weapon["range_ft"]
            to_hit_mod = char.ability_scores.dex_mod if (weapon["finesse"] or is_ranged) else char.ability_scores.str_mod
            char.attacks.append(Attack(
                name=item_name,
                to_hit_bonus=char.proficiency_bonus + to_hit_mod + bonus,
                damage_dice=with_ability_mod(weapon["damage_dice"], to_hit_mod + bonus),
                damage_type=weapon["damage_type"],
                range_ft=weapon["range_ft"],
                reach_ft=weapon_reach_ft(weapon),
                notes=description,
            ))
            result = f"{item_name} added to {char.name}'s attacks (grounded on {base_key})."
        elif armor or is_shield:
            char.ac += bonus
            result = f"{char.name}'s AC increased by {bonus} ({item_name}, grounded on {base_key})."
        else:
            result = f"{item_name} added to {char.name}'s inventory (no mechanical effect — narrate its power as needed)."

        new_item = Item(
            name=item_name,
            description=description,
            magical=True,
            requires_attunement=requires_attunement,
            rarity=rarity,
        )
        char.inventory.append(new_item)
        _mark_loot_granted(campaign)
        await store.save(campaign)
        if graph_store is not None:
            await graph_store.add_edge(
                campaign_id, "character", char.id, char.name, "item", new_item.id, new_item.name, "owns",
            )
        return result

    @tool
    async def add_weapon_attack(
        character_name: str, item_name: str, base_item: str, action_type: str = "action",
    ) -> str:
        """Grant a character a real, usable Attack from a mundane (non-magical)
        weapon already in their inventory — call this once a character intends to
        actually fight with a looted/purchased mundane weapon, so resolve_attack
        has a grounded attack_name to use instead of falling back to Unarmed
        Strike or, worse, an invented one. base_item must be a real weapon name
        (search_rules if unsure) — this grounds to-hit/
        damage in the character's real proficiency bonus and STR/DEX modifier,
        the same way create_magic_item grounds a magical weapon's stats, just
        with no enchantment bonus. Refuses if item_name isn't already in the
        character's inventory (add_item_to_character/reveal_loot distribution
        first) or if base_item isn't a recognized weapon, or if this character
        already has an attack with this exact name (no-op, not a duplicate).
        For a magical weapon, use create_magic_item instead.

        action_type: pass "bonus_action" for an off-hand/Two-Weapon-Fighting-style
        attack — defaults to "action" for a character's primary weapon."""
        campaign = await store.load(campaign_id)
        char = find_char(campaign, character_name)
        if not char:
            return f"No character named '{character_name}' in the party."
        if any(a.name.lower() == item_name.lower() for a in char.attacks):
            return f"{char.name} already has an attack named '{item_name}' — not adding a duplicate."
        if not any(i.name.lower() == item_name.lower() for i in char.inventory):
            return f"{char.name} doesn't have '{item_name}' in their inventory — add it first."
        weapon = WEAPONS.get(base_item.strip())
        if not weapon:
            return f"'{base_item}' isn't a recognized weapon — check the exact name via search_rules."
        is_ranged = "/" in weapon["range_ft"]
        to_hit_mod = char.ability_scores.dex_mod if (weapon["finesse"] or is_ranged) else char.ability_scores.str_mod
        to_hit_bonus = char.proficiency_bonus + to_hit_mod
        char.attacks.append(Attack(
            name=item_name,
            to_hit_bonus=to_hit_bonus,
            damage_dice=with_ability_mod(weapon["damage_dice"], to_hit_mod),
            damage_type=weapon["damage_type"],
            range_ft=weapon["range_ft"],
            action_type=action_type,
            reach_ft=weapon_reach_ft(weapon),
        ))
        await store.save(campaign)
        return (
            f"{item_name} added to {char.name}'s attacks (grounded on {base_item}, "
            f"+{to_hit_bonus} to hit, {weapon['damage_dice']} {weapon['damage_type'].value})."
        )

    @tool
    async def reveal_loot(
        source_name: str,
        items: list[dict] | None = None,
        currency: dict | None = None,
    ) -> str:
        """Reveal loot found from a shared source — a defeated enemy's body, a
        searched container/area — where more than one party member could plausibly
        claim it. Decide concrete contents NOW (real item names/quantities, a real
        coin amount — never leave a "pouch of coins" vague) and pass them here.
        This records the find and returns an unassigned loot block for the
        narrator to show the player; it does NOT put anything in anyone's
        inventory yet. Once the player says who takes what, call
        add_item_to_character/update_character_currency/remove_item_from_character
        per their allocation — never assign shared loot to a character on your own
        guess. For a solo find only one character could possibly want (a locked
        box only the searching rogue could reach, coins found while alone), skip
        this and call add_item_to_character/update_character_currency directly
        instead.

        items: list of {"name": str, "quantity": int (default 1), "description": str (optional)}
        currency: {"pp": int, "gp": int, "ep": int, "sp": int, "cp": int} — omit unused denominations.
        """
        campaign = await store.load(campaign_id)
        container = Container(
            name=source_name,
            is_open=True,
            contents=[
                Item(name=i["name"], quantity=i.get("quantity", 1), description=i.get("description", ""))
                for i in (items or [])
            ],
            currency=Currency(**(currency or {})),
        )
        campaign.containers.append(container)
        _mark_loot_granted(campaign)
        await store.save(campaign)

        lines = [f"💰 Loot found — {source_name} (unassigned):"]
        lines += [
            f"  - {i.name}" + (f" x{i.quantity}" if i.quantity > 1 else "")
            for i in container.contents
        ]
        coins = ", ".join(f"{v} {k}" for k, v in (currency or {}).items() if v)
        if coins:
            lines.append(f"  - {coins}")
        lines.append("Ask the party how to split it before assigning anything.")
        return "\n".join(lines)

    @tool
    async def get_unassigned_loot() -> str:
        """List every shared find from reveal_loot that still has unclaimed contents
        or currency, across the whole campaign (not just this encounter). Call this
        before resolving a player's allocation of a find ("I'll take the pouch",
        "add it to my inventory") if its exact contents aren't already in your
        recent context — reveal_loot's own result can scroll out of view by the
        time the party gets around to splitting it up, and guessing at contents
        instead of checking here is exactly how a claimed item silently fails to
        reach anyone's actual inventory. Does not mutate anything."""
        campaign = await store.load(campaign_id)
        if not campaign or not campaign.containers:
            return "No unassigned loot recorded."
        lines = []
        for c in campaign.containers:
            coins = ", ".join(f"{v} {k}" for k, v in c.currency.model_dump().items() if v)
            if not c.contents and not coins:
                continue
            lines.append(f"{c.name}:")
            lines += [
                f"  - {i.name}" + (f" x{i.quantity}" if i.quantity > 1 else "")
                for i in c.contents
            ]
            if coins:
                lines.append(f"  - {coins}")
        if not lines:
            return "No unassigned loot recorded."
        return "\n".join(lines)

    return [
        get_party_status,
        get_character,
        get_unassigned_loot,
        update_character_hp,
        update_character_detail,
        add_condition,
        remove_condition,
        apply_effect,
        remove_effect,
        use_spell_slot,
        restore_spell_slots,
        add_item_to_character,
        remove_item_from_character,
        update_character_currency,
        create_magic_item,
        add_weapon_attack,
        reveal_loot,
    ]
