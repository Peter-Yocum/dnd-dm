from langchain_core.tools import tool

from backend.models import (
    Attack, Campaign, CombatantPosition, CombatantType, CoverType, DamageType,
    Encounter, InitiativeEntry, Monster, MonsterSize, MonsterType, ZoneType,
)
from backend.stores.campaign_store import CampaignStore
from backend.stores.lore_store import LoreStore
from backend.tools._helpers import (
    advance_combatant_turn, apply_damage_to_monster, find_char, find_monster,
    monster_summary, roll_notation,
)


def build_encounter_context(campaign: Campaign) -> str | None:
    """Live, ground-truth summary of the active encounter — round, initiative
    order, the real registered name/HP/AC/attacks of every monster, tactical
    positions, and any pending reaction — for the mechanics model. Auto-injected
    into every mechanics-node invocation during combat (see dm_agent.py's
    _make_mechanics_modifier) rather than a tool the model has to remember to
    call every turn. Returns None when there's no active encounter, so the
    injection point can skip cleanly."""
    enc = campaign.active_encounter
    if not enc or not enc.is_active:
        return None

    lines = []
    if enc.pending_action:
        lines.append(f"⚠ PENDING REACTION: {enc.pending_action.prompt_note}")
    lines.append(f"=== Active Encounter — Round {enc.round} — {enc.location_description} ===")
    lines.append("Initiative order:")
    for e in enc.initiative_order:
        marker = "→" if e.is_current_turn else " "
        lines.append(f"  {marker} [{e.initiative:>2}] {e.name} ({e.combatant_type.value})")

    monster_names = {e.name for e in enc.initiative_order if e.combatant_type == CombatantType.MONSTER}
    if monster_names:
        lines.append("\nMonsters:")
        for name in monster_names:
            monster = find_monster(campaign, name)
            lines.append(f"  {monster_summary(monster)}" if monster else f"  {name}: no stat block found — this is a bug, do not re-create it, investigate instead")

    if enc.combatant_positions:
        lines.append("\nPositions:")
        for p in enc.combatant_positions:
            cover = f", {p.cover.value} cover" if p.cover.value != "none" else ""
            lines.append(f"  {p.name}: {p.zone.value}{cover}")

    return "\n".join(lines)


def make_tools(
    campaign_id: str,
    store: CampaignStore,
    lore_store: LoreStore | None = None,
    books_in_play: list[str] | None = None,
) -> list:

    @tool
    async def create_monster(
        name: str,
        ac: int,
        max_hp: int,
        attacks: list[dict],
        size: str = "medium",
        monster_type: str = "humanoid",
        cr: str = "0",
        xp: int = 0,
        ac_description: str = "",
        count: int = 1,
    ) -> str:
        """Create a monster stat block and add it to the campaign, so it can
        actually take damage in combat via resolve_attack/update_monster_hp. Call
        this for every new opponent BEFORE start_encounter — pass your best
        AC/HP/attacks estimate; if this name matches a canonical stat block in
        the precomputed Lore Registry (Monster Manual, Volo's, Mordenkainen's,
        Tasha's — see scripts/extract_entities.py --source-type core), those
        EXACT numbers silently override whatever you passed, so you don't need
        to get every digit right yourself — just get the name right. If no
        canonical entry exists (homebrew, an adventure-unique creature), your
        passed numbers are used as-is — search_rules first for a real stat
        block when one might exist, and be prepared to say so is a DM
        improvisation if asked, same as an ungrounded rules ruling.

        count: create this many identical copies in one call (e.g. 3 goblins),
        auto-named "{name} 1".."{name} N" — leave at the default 1 for a single
        monster with no numeric suffix. Clamped to 1-20.

        attacks: list of {"name": str, "to_hit_bonus": int, "damage_dice": str,
        "damage_type": str} — damage_type must be one of: acid, bludgeoning,
        cold, fire, force, lightning, necrotic, piercing, poison, psychic,
        radiant, slashing, thunder.
        """
        campaign = await store.load(campaign_id)

        count = max(1, min(20, count))
        names = [name] if count == 1 else [f"{name} {i}" for i in range(1, count + 1)]

        existing_lower = {m.name.lower() for m in campaign.monsters}
        collisions = [n for n in names if n.lower() in existing_lower]
        if collisions:
            return f"A monster named '{collisions[0]}' already exists — use a different name (e.g. '{name} 2')."

        canon_note = ""
        if lore_store is not None:
            canon = await lore_store.find_by_name_or_alias(books_in_play or [], name, entity_type="monster")
            if canon:
                profile = canon.rolled_up_profile
                if profile.get("ac"):
                    ac = int(profile["ac"])
                if profile.get("hp"):
                    max_hp = int(profile["hp"])
                if profile.get("challenge_rating"):
                    cr = str(profile["challenge_rating"])
                if profile.get("attacks"):
                    attacks = profile["attacks"]
                canon_note = f" [canonical stats from '{canon.book_slug}']"

        try:
            size_enum = MonsterSize(size.lower())
        except ValueError:
            size_enum = MonsterSize.MEDIUM
        try:
            type_enum = MonsterType(monster_type.lower())
        except ValueError:
            type_enum = MonsterType.HUMANOID

        built_attacks = []
        for a in attacks:
            try:
                dtype = DamageType(a.get("damage_type", "bludgeoning").lower())
            except ValueError:
                dtype = DamageType.BLUDGEONING
            built_attacks.append(Attack(
                name=a["name"],
                to_hit_bonus=int(a.get("to_hit_bonus", 0)),
                damage_dice=a.get("damage_dice", "1d4"),
                damage_type=dtype,
            ))

        created = []
        for n in names:
            monster = Monster(
                name=n,
                size=size_enum,
                monster_type=type_enum,
                ac=ac,
                ac_description=ac_description or None,
                max_hp=max_hp,
                current_hp=max_hp,
                attacks=[a.model_copy() for a in built_attacks],
                cr=cr,
                xp=xp,
                reaction_available=True,
            )
            campaign.monsters.append(monster)
            created.append(monster)
        await store.save(campaign)
        return f"{', '.join(m.name for m in created)} created (AC {ac}, {max_hp} HP, CR {cr}){canon_note}."

    @tool
    async def start_encounter(
        location_description: str,
        combatants: list[dict],
        difficulty: str = "medium",
        xp_budget: int = 0,
    ) -> str:
        """Begin a combat encounter. Pass a list of combatants, each with:
          {"name": str, "type": "character"|"monster"|"npc",
           "initiative_override": int (optional), "surprised": bool (optional)}
        Initiative is rolled internally (d20 + the combatant's own initiative
        modifier/DEX) unless initiative_override is given — no need to call
        roll_dice per combatant first. Sort order is computed automatically.

        surprised: pass True for any combatant caught unaware when the ambush
        began (an unaware guard, a party that walked into an ambush without
        noticing it). Per the 5e surprise rule, this is NOT a separate turn or
        round before initiative — a surprised combatant simply rolls their own
        initiative with disadvantage (2d20, keep the lower) instead of a
        normal roll. Everyone still acts in one combined initiative order;
        surprise just tends to push the surprised side later in it. Leave
        False for anyone already aware a fight was starting.
        Clears any previous encounter and refreshes every combatant's reaction."""
        campaign = await store.load(campaign_id)

        entries: list[InitiativeEntry] = []
        unresolved: list[str] = []
        surprised_names: list[str] = []
        for c in combatants:
            try:
                ctype = CombatantType(c.get("type", "monster").lower())
            except ValueError:
                ctype = CombatantType.MONSTER

            override = c.get("initiative_override", c.get("initiative"))
            if override is not None:
                initiative_value = int(override)
            else:
                char = find_char(campaign, c["name"])
                monster = None if char else find_monster(campaign, c["name"])
                if char:
                    modifier = char.initiative_modifier or char.ability_scores.dex_mod
                elif monster:
                    modifier = monster.ability_scores.dex_mod
                else:
                    modifier = 0
                    unresolved.append(c["name"])
                sign = "+" if modifier >= 0 else ""
                dice_notation = f"2d20kl1{sign}{modifier}" if c.get("surprised") else f"1d20{sign}{modifier}"
                initiative_value, _ = roll_notation(dice_notation)

            if c.get("surprised"):
                surprised_names.append(c["name"])

            entries.append(InitiativeEntry(
                name=c["name"],
                combatant_type=ctype,
                initiative=initiative_value,
            ))
        entries.sort(key=lambda e: e.initiative, reverse=True)
        if entries:
            entries[0].is_current_turn = True

        enc = Encounter(
            location_description=location_description,
            round=1,
            is_active=True,
            initiative_order=entries,
            difficulty=difficulty,  # type: ignore[arg-type]
            xp_budget=xp_budget,
        )
        if campaign.current_location_id:
            enc.location_id = campaign.current_location_id

        campaign.active_encounter = enc

        # Fresh reactions for everyone at the start of a new encounter — a
        # Character/Monster is a long-lived record that may carry a stale False
        # left over from a previous encounter that never advanced past whoever
        # used a reaction.
        for entry in entries:
            combatant = find_char(campaign, entry.name) or find_monster(campaign, entry.name)
            if combatant is not None:
                combatant.reaction_available = True

        await store.save(campaign)

        order = "\n".join(
            f"  {'→' if e.is_current_turn else ' '} [{e.initiative:>2}] {e.name} ({e.combatant_type.value})"
            + (" — surprised" if e.name in surprised_names else "")
            for e in entries
        )
        result = f"Encounter started (round 1). Initiative:\n{order}"
        if unresolved:
            result += f"\n(Note: {', '.join(unresolved)} not found in campaign — rolled at +0.)"
        return result

    @tool
    async def advance_initiative() -> str:
        """Advance to the next combatant's turn. Call at the end of a turn that
        wasn't already ended via resolve_attack's/resolve_saving_throw's
        end_turn=True. Automatically increments the round counter when the order
        wraps around and refreshes the next combatant's reaction."""
        campaign = await store.load(campaign_id)
        enc = campaign.active_encounter
        if not enc or not enc.is_active:
            return "No active encounter."
        msg = advance_combatant_turn(campaign, enc)
        await store.save(campaign)
        return msg

    @tool
    async def end_encounter(xp_awarded: int = 0) -> str:
        """End the current combat encounter. Pass xp_awarded to record what the
        party earned; leave 0 if XP is not used or will be tracked separately."""
        campaign = await store.load(campaign_id)
        if not campaign.active_encounter or not campaign.active_encounter.is_active:
            return "No active encounter to end."
        enc = campaign.active_encounter
        enc.is_active = False
        enc.xp_awarded = xp_awarded
        campaign.active_encounter = None
        await store.save(campaign)
        msg = f"Encounter ended after {enc.round} round(s)."
        if xp_awarded:
            msg += f" {xp_awarded} XP awarded."
        return msg

    @tool
    async def update_monster_hp(monster_name: str, delta: int) -> str:
        """Apply damage (negative delta) or healing (positive delta) to a monster
        in the active encounter. Not tied to a specific listed Attack — prefer
        resolve_attack for a monster taking damage from an actual attack roll;
        use this for freeform damage (falling, traps, fire, poison)."""
        campaign = await store.load(campaign_id)
        monster = find_monster(campaign, monster_name)
        if not monster:
            return f"No monster named '{monster_name}' found."
        msg = apply_damage_to_monster(monster, delta)
        await store.save(campaign)
        return msg

    @tool
    async def set_combatant_position(
        combatant_name: str,
        zone: str,
        cover: str = "none",
        notes: str = "",
    ) -> str:
        """Update a combatant's tactical position in the active encounter.
        zone: melee | adjacent | near | far | distant
        cover: none | half | three_quarters | total"""
        campaign = await store.load(campaign_id)
        enc = campaign.active_encounter
        if not enc or not enc.is_active:
            return "No active encounter."
        try:
            zone_type = ZoneType(zone.lower())
        except ValueError:
            return f"'{zone}' is not a valid zone. Use: {', '.join(z.value for z in ZoneType)}."
        try:
            cover_type = CoverType(cover.lower())
        except ValueError:
            cover_type = CoverType.NONE

        existing = next((p for p in enc.combatant_positions if p.name.lower() == combatant_name.lower()), None)
        if existing:
            existing.zone = zone_type
            existing.cover = cover_type
            existing.notes = notes
        else:
            enc.combatant_positions.append(CombatantPosition(
                name=combatant_name,
                zone=zone_type,
                cover=cover_type,
                notes=notes,
            ))
        await store.save(campaign)
        cover_str = f", {cover_type.value} cover" if cover_type != CoverType.NONE else ""
        return f"{combatant_name} is now at {zone_type.value} range{cover_str}."

    return [create_monster, start_encounter, advance_initiative, end_encounter, update_monster_hp, set_combatant_position]
