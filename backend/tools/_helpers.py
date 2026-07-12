"""Shared lookup utilities used across all tool modules."""
import json
import random
import re
from pathlib import Path

from backend.data.equipment import ARMOR, SHIELD_AC_BONUS, STARTING_KITS, WEAPONS
from backend.data.fivee_options import CLASSES, HIT_DICE, STARTING_SPELL_SLOTS
from backend.data.spells import ALL_SPELLS, SPELL_MENUS, SPELL_REQUIREMENTS, SPELLCASTING_ABILITY
from backend.models import (
    AbilityScores, Attack, Campaign, Character, Container, Currency, DamageType, Encounter,
    Item, Location, LocationConnection, Monster, NPC, Quest, Spell, SpellSlotLevel, TimeOfDay,
)

ADVENTURES_DIR = Path("docs/source/adventures")


def read_adventure_meta(slug: str) -> dict:
    """Read _meta.json for one adventure folder. Falls back to a bare
    title-cased name with empty description/levels/recommended_players/
    opening_hook if the folder or _meta.json is missing. Shared by main.py's
    campaign-creation picker and the agent-facing tools/prompts below, so
    both read the same file the same way.

    opening_hook (added 2026-07-04): a short, DM-authored, verified-against-
    the-real-text directive describing how THIS specific adventure's story
    actually begins (e.g. Out of the Abyss: the party starts as manacled
    prisoners in the drow outpost of Velkynvelve — not a free choice of
    starting location). Distinct from `description`, which is a general
    back-of-the-book pitch for the campaign-creation picker UI and isn't
    reliably phrased as a concrete opening-scene directive. Added because
    session kickoff (build_session_kickoff_message) needs a deterministic,
    always-available anchor for the opening scene — a semantic RAG search
    for "the adventure's opening" was tested live and found unreliable
    (surfaced the wrong section for several adventures, e.g. Icewind Dale,
    Ghosts of Saltmarsh), so a curated field beats an auto-retrieved one for
    this specific purpose.

    opening_location / opening_section_marker (added 2026-07-05): same
    "hand-curate what RAG proved unreliable for" precedent as opening_hook,
    one level deeper — world_prep.py's opening-scene NPC/site-detail seeding
    needs the exact name of the starting location (opening_location, e.g.
    "Velkynvelve") and the exact `## Chapter N: <title>` heading marking
    where the opening chapter begins in the source markdown
    (opening_section_marker, e.g. "## Chapter 1: Prisoners of the Drow") so
    it can deterministically extract the whole chapter — a name-proximity
    search (semantic or literal) was confirmed to miss most of a roster
    whose entries don't repeat the location's own name nearby. Both default
    to "" for any adventure without this curation yet; world_prep.py treats
    that as "skip this phase," not an error."""
    meta_file = ADVENTURES_DIR / slug / "_meta.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text())
    else:
        meta = {}
    return {
        "name": meta.get("name", slug.replace("-", " ").title()),
        "description": meta.get("description", ""),
        "levels": meta.get("levels", ""),
        "recommended_players": meta.get("recommended_players", ""),
        "opening_hook": meta.get("opening_hook", ""),
        "opening_location": meta.get("opening_location", ""),
        "opening_section_marker": meta.get("opening_section_marker", ""),
    }


def find_char(campaign: Campaign, name: str) -> Character | None:
    n = name.lower()
    return next((c for c in campaign.party if c.name.lower() == n), None)


def find_npc(campaign: Campaign, name: str) -> NPC | None:
    n = name.lower()
    return next((c for c in campaign.npcs if c.name.lower() == n), None)


def find_monster(campaign: Campaign, name: str) -> Monster | None:
    n = name.lower()
    return next((m for m in campaign.monsters if m.name.lower() == n), None)


def find_quest(campaign: Campaign, name: str) -> Quest | None:
    n = name.lower()
    return next((q for q in campaign.quests if q.name.lower() == n), None)


def find_location(campaign: Campaign, name: str) -> Location | None:
    n = name.lower()
    return next((l for l in campaign.locations if l.name.lower() == n), None)


def find_container(campaign: Campaign, name: str) -> Container | None:
    n = name.lower()
    return next((c for c in campaign.containers if c.name.lower() == n), None)


def find_container_by_id(campaign: Campaign, container_id: str) -> Container | None:
    return next((c for c in campaign.containers if c.id == container_id), None)


def assign_container_item(
    campaign: Campaign, container: Container, item_id: str, character: Character,
) -> str:
    """Move one item straight from an unassigned Container (reveal_loot/
    end_encounter's shared find) into a character's inventory — a direct UI
    action, not a tool call, so there's no LLM narration step for it to get
    lost in (the repeated "narrated loot never actually landed" bug this
    exists to route around entirely). Drops the empty container once both
    its contents and currency are exhausted, so the loot panel doesn't keep
    showing a used-up find."""
    item = next((i for i in container.contents if i.id == item_id), None)
    if not item:
        raise ValueError(f"No item with id '{item_id}' in container '{container.name}'.")
    container.contents.remove(item)
    character.inventory.append(item)
    if not container.contents and not container.currency.to_gp():
        campaign.containers.remove(container)
    return f"{item.name} moved to {character.name}'s inventory."


def assign_container_currency(
    campaign: Campaign, container: Container, denom: str, amount: int, character: Character,
) -> str:
    """Move up to `amount` of one currency denomination from an unassigned
    Container into a character's purse — same direct, non-agent action as
    assign_container_item, for the coin side of a shared find."""
    if denom not in ("cp", "sp", "ep", "gp", "pp"):
        raise ValueError(f"Unknown denomination '{denom}'.")
    available = getattr(container.currency, denom)
    moved = min(amount, available)
    if moved <= 0:
        raise ValueError(f"Container '{container.name}' has no {denom} left to assign.")
    setattr(container.currency, denom, available - moved)
    setattr(character.currency, denom, getattr(character.currency, denom) + moved)
    if not container.contents and not container.currency.to_gp():
        campaign.containers.remove(container)
    return f"{moved} {denom} moved to {character.name}."


def find_connection(loc: Location, destination_name: str) -> LocationConnection | None:
    n = destination_name.lower()
    return next((c for c in loc.connections if c.to_location_name.lower() == n), None)


def find_item_anywhere(campaign: Campaign, name: str) -> tuple[Item, str] | None:
    """Search every Item-holding place in the campaign — Character.inventory,
    NPC.inventory, Container.contents — for an exact (case-insensitive) name
    match. Returns (item, holder_description) or None. Unlike find_char/
    find_npc/etc., items aren't a single top-level campaign list, so this
    checks all three holder types rather than one."""
    n = name.lower()
    for char in campaign.party:
        for item in char.inventory:
            if item.name.lower() == n:
                return item, f"{char.name}'s inventory"
    for npc in campaign.npcs:
        for item in npc.inventory:
            if item.name.lower() == n:
                return item, f"{npc.name}'s inventory"
    for container in campaign.containers:
        for item in container.contents:
            if item.name.lower() == n:
                return item, f"container '{container.name}'"
    return None


def all_campaign_item_names(campaign: Campaign) -> list[str]:
    """All item names currently held anywhere in the campaign — used for
    fuzzy dedup blocking before creating a new item."""
    names = []
    for char in campaign.party:
        names += [i.name for i in char.inventory]
    for npc in campaign.npcs:
        names += [i.name for i in npc.inventory]
    for container in campaign.containers:
        names += [i.name for i in container.contents]
    return names


_OPPOSITES = {
    "north": "south", "south": "north", "east": "west", "west": "east",
    "up": "down", "down": "up", "in": "out", "out": "in",
}


def opposite_direction(direction: str) -> str:
    """Best-effort reverse of a cardinal direction; free-text directions
    ("the coast road") just return "" — to_location_name carries orientation."""
    return _OPPOSITES.get(direction.strip().lower(), "")


_TIME_STEPS = [
    TimeOfDay.DAWN, TimeOfDay.MORNING, TimeOfDay.MIDDAY, TimeOfDay.AFTERNOON,
    TimeOfDay.DUSK, TimeOfDay.EVENING, TimeOfDay.NIGHT, TimeOfDay.MIDNIGHT,
]  # 8 steps, ~3 hours each


def advance_clock(campaign: Campaign, hours: float) -> int:
    """Advance campaign.days_elapsed/time_of_day by `hours` of game time.
    Mutates campaign in place; returns the number of days advanced."""
    steps = round(hours / 3)
    total = _TIME_STEPS.index(campaign.time_of_day) + steps
    days_advanced, new_idx = divmod(total, 8)
    campaign.days_elapsed += days_advanced
    campaign.time_of_day = _TIME_STEPS[new_idx]
    return days_advanced


def apply_long_rest(campaign: Campaign) -> str:
    """Whole-party long rest (8 hours), mutating campaign in place: full HP,
    all spell slots restored, exhaustion -1, death saves cleared, hit dice
    regained (real 5e rule: half your total hit dice, rounded down, minimum
    1) — same effects `restore_spell_slots` already applies per-character,
    but bulk, deterministic, and also fixes `last_long_rest_day` (which nothing
    else in this codebase ever sets, leaving the UI's rest-status line
    permanently stuck on "No long rest today"). Returns a plain summary,
    no narration — deliberately not routed through any LLM, since this is
    pure bulk arithmetic with a known correct answer, not a narrative choice."""
    lines = []
    for char in campaign.party:
        healed = char.max_hp - char.current_hp
        char.current_hp = char.max_hp
        for lvl, slot in char.spell_slots.items():
            char.spell_slots[lvl] = SpellSlotLevel(max=slot.max, used=0)
        char.exhaustion_level = max(0, char.exhaustion_level - 1)
        char.death_save_successes = 0
        char.death_save_failures = 0
        total_dice = int(char.hit_dice_total.split("d")[0]) if "d" in char.hit_dice_total else char.level
        regained = max(1, total_dice // 2)
        old_remaining = char.hit_dice_remaining
        char.hit_dice_remaining = min(total_dice, char.hit_dice_remaining + regained)
        lines.append(
            f"{char.name}: +{healed} HP (now {char.current_hp}/{char.max_hp}), "
            f"spell slots restored, hit dice {old_remaining}→{char.hit_dice_remaining}"
        )
    advance_clock(campaign, 8)
    campaign.last_long_rest_day = campaign.days_elapsed
    return "The party takes a long rest.\n" + "\n".join(lines)


def apply_short_rest(campaign: Campaign) -> str:
    """Whole-party short rest (1 hour), mutating campaign in place. Real 5e
    short rest lets each player choose how many Hit Dice to spend for
    healing (roll + CON mod each); this app has no per-character interactive
    UI for that choice, so — matching the same "fixed average, no rolling"
    simplification already used for level_up's HP gain — every character
    automatically spends just enough hit dice (average value per die) to
    reach full HP, capped at however many dice they have remaining; no dice
    are wasted healing past max_hp. Warlock spell slots (the only class
    whose slots recharge on a short rest, not just a long one — Pact Magic)
    are restored; every other class's slots are untouched, correctly
    matching real rules. Returns a plain summary, no narration, same
    no-LLM reasoning as apply_long_rest."""
    lines = []
    for char in campaign.party:
        hit_die_size = int(char.hit_dice_total.split("d")[1]) if "d" in char.hit_dice_total else 8
        con_mod = char.ability_scores.modifier(char.ability_scores.constitution)
        per_die = max(1, hit_die_size // 2 + 1 + con_mod)
        dice_spent = 0
        healed = 0
        missing_hp = char.max_hp - char.current_hp
        if missing_hp > 0 and char.hit_dice_remaining > 0:
            dice_needed = -(-missing_hp // per_die)  # ceil division
            dice_spent = min(dice_needed, char.hit_dice_remaining)
            healed = min(missing_hp, per_die * dice_spent)
            char.current_hp += healed
            char.hit_dice_remaining -= dice_spent
        note = f"{char.name}: "
        note += f"+{healed} HP ({dice_spent} hit dice spent)" if dice_spent else "no healing needed/no hit dice left"
        if char.char_class == "Warlock" and char.spell_slots:
            for lvl, slot in char.spell_slots.items():
                char.spell_slots[lvl] = SpellSlotLevel(max=slot.max, used=0)
            note += ", Pact Magic slots restored"
        lines.append(note)
    advance_clock(campaign, 1)
    return "The party takes a short rest.\n" + "\n".join(lines)


def with_ability_mod(dice: str, modifier: int) -> str:
    """Combine a base weapon-table damage dice string with a flat modifier,
    formatted the way _DICE_RE parses it back ("1d6"+4 -> "1d6+4", "1d6"+-1
    -> "1d6-1", "1d6"+0 -> "1d6" unchanged). Every PC weapon-damage call site
    (chargen's starting weapon, unarmed strike, add_weapon_attack,
    create_magic_item) uses this — 2026-07-11: previously none of them did,
    which meant Attack.damage_dice only ever carried the bare weapon-table
    die (equipment.py's WEAPONS, e.g. Shortbow's "1d6") with the character's
    STR/DEX modifier silently never added at any point in the pipeline
    (resolve_attack rolls damage_dice as stored — see resolution.py), while
    to_hit_bonus correctly included it the whole time. Caught live: a
    level-1 Shortbow user with DEX 18 (+4) rolling "6 piercing" damage that
    should have been 7-10. Previously described as "a known simplification"
    (see unarmed_strike_attack's old docstring) but a real accuracy gap, not
    a deliberate design choice — every character with a positive modifier
    (nearly all of them) was underdealing damage.

    Caveat for a future ASI/stat-change feature: unlike to_hit_bonus (a
    patchable int, see levelup.py), "1d6+4" carries no record of which part
    is weapon vs. modifier — recompute from the weapon table rather than
    delta-patching the string."""
    if modifier > 0:
        return f"{dice}+{modifier}"
    if modifier < 0:
        return f"{dice}{modifier}"
    return dice


def unarmed_strike_attack(str_mod: int) -> Attack:
    """Every creature can always make an unarmed strike (PHB) regardless of
    what weapon, if any, it's currently carrying — a baseline every
    character should have, not just one a DM remembers to add by hand when
    a character loses or is stripped of their weapon (e.g. captured at the
    start of an adventure). 1d4 bludgeoning rather than RAW's flat 1 damage:
    roll_notation requires a real die (2-1000 sides) so a flat value isn't
    representable anyway — 1d4 matches the lightest real weapons (Dagger)
    rather than inventing a new convention. STR mod IS included (see
    with_ability_mod) — RAW unarmed strike is "1 + STR mod", and now that
    every other attack in this app correctly includes its modifier, leaving
    this one out would make it disproportionately weak by comparison."""
    return Attack(
        name="Unarmed Strike",
        to_hit_bonus=2 + str_mod,  # proficiency bonus is always +2 at level 1
        damage_dice=with_ability_mod("1d4", str_mod),
        damage_type=DamageType.BLUDGEONING,
        range_ft="5",
    )


def _starting_equipment(ab: AbilityScores, char_class: str, dex_mod: int, str_mod: int) -> dict:
    """Derive attacks/inventory/currency/ac from the class's default kit
    (backend/data/equipment.py). Split out of derive_level1_stats purely for
    readability — always called from there, not a public entry point."""
    kit = STARTING_KITS.get(char_class)
    if not kit:
        # Unknown class name (bad data, homebrew) — unarmed and unarmored
        # rather than guessing; matches HIT_DICE's own defensive fallback.
        return {"attacks": [unarmed_strike_attack(str_mod)], "inventory": [], "currency": Currency(), "ac": 10 + dex_mod}

    weapon_name = kit["weapon"]
    weapon = WEAPONS[weapon_name]
    is_ranged = "/" in weapon["range_ft"]
    to_hit_mod = dex_mod if (weapon["finesse"] or is_ranged) else str_mod
    attacks = [Attack(
        name=weapon_name,
        to_hit_bonus=2 + to_hit_mod,  # proficiency bonus is always +2 at level 1
        damage_dice=with_ability_mod(weapon["damage_dice"], to_hit_mod),
        damage_type=weapon["damage_type"],
        range_ft=weapon["range_ft"],
    ), unarmed_strike_attack(str_mod)]

    inventory = [Item(name=weapon_name)]
    ac = 10 + dex_mod
    if kit["armor"]:
        armor = ARMOR[kit["armor"]]
        inventory.append(Item(name=kit["armor"]))
        if armor["dex_cap"] is None:
            ac = armor["base_ac"] + dex_mod
        elif armor["dex_cap"] == 0:
            ac = armor["base_ac"]
        else:
            ac = armor["base_ac"] + min(dex_mod, armor["dex_cap"])
    if kit["shield"]:
        inventory.append(Item(name="Shield"))
        ac += SHIELD_AC_BONUS
    inventory += [Item(name=g) for g in kit["gear"]]

    return {
        "attacks": attacks,
        "inventory": inventory,
        "currency": Currency(gp=kit["gold"]),
        "ac": ac,
    }


def derive_level1_stats(ab: AbilityScores, char_class: str, skill_proficiencies: set[str]) -> dict:
    """Compute level-1 derived stats (HP, AC, passive perception, spell
    slots, hit dice, starting equipment) from ability scores + class. Shared
    by the Session 0 finalize_character flow (backend/tools/chargen.py) and
    DM-generated companion characters (backend/tools/companion.py) — same
    math, two entry points, since a companion is a level-1 character just
    like a new PC, just built in one direct call instead of a multi-turn
    draft.

    AC and starting gear come from the class's default kit
    (backend/data/equipment.py) rather than assuming unarmored — a known
    simplification: this doesn't model class-specific Unarmored Defense
    (Barbarian CON, Monk WIS), those classes just get the plain unarmored
    formula like everyone else with no armor in their kit."""
    hit_die = HIT_DICE.get(char_class, 8)
    con_mod = ab.modifier(ab.constitution)
    max_hp = max(1, hit_die + con_mod)
    dex_mod = ab.modifier(ab.dexterity)
    str_mod = ab.modifier(ab.strength)

    wis_mod = ab.modifier(ab.wisdom)
    perc_prof = "perception" in skill_proficiencies
    passive_perception = 10 + wis_mod + (2 if perc_prof else 0)

    slot_map = STARTING_SPELL_SLOTS.get(char_class, {})
    spell_slots = {lvl: SpellSlotLevel(max=count) for lvl, count in slot_map.items()}

    equipment = _starting_equipment(ab, char_class, dex_mod, str_mod)

    return {
        "max_hp": max_hp,
        "ac": equipment["ac"],
        "passive_perception": passive_perception,
        "spell_slots": spell_slots,
        "hit_dice_total": f"1d{hit_die}",
        "attacks": equipment["attacks"],
        "inventory": equipment["inventory"],
        "currency": equipment["currency"],
    }


def derive_saving_throw_proficiencies(char_class: str) -> set[str]:
    """Look up a class's two saving throw proficiencies (fivee_options.CLASSES,
    e.g. Ranger -> ["Strength", "Dexterity"]) and lowercase them to match the
    ability-name keys resolve_saving_throw's _save_bonus checks against.
    Empty set for an unrecognized class rather than raising, matching
    derive_level1_stats' HIT_DICE.get(..., 8) fallback style."""
    return {s.lower() for s in CLASSES.get(char_class, {}).get("saving_throws", [])}


def derive_spellcasting_stats(ab: AbilityScores, char_class: str, proficiency_bonus: int) -> dict:
    """Compute spellcasting_ability/spell_save_dc/spell_attack_bonus from final
    ability scores + class — pure arithmetic (8 + prof + mod / prof + mod),
    independent of which specific spells get chosen. All None for a
    non-casting class (not in SPELLCASTING_ABILITY). Split out from
    derive_level1_stats as a sibling, not folded in, since that function's
    callers don't carry proficiency_bonus or care about spell selection."""
    ability_name = SPELLCASTING_ABILITY.get(char_class)
    if not ability_name:
        return {"spellcasting_ability": None, "spell_save_dc": None, "spell_attack_bonus": None}
    mod = ab.modifier(getattr(ab, ability_name))
    return {
        "spellcasting_ability": ability_name,
        "spell_save_dc": 8 + proficiency_bonus + mod,
        "spell_attack_bonus": proficiency_bonus + mod,
    }


def build_spells_known(char_class: str, chosen_names: list[str]) -> tuple[list[Spell], list[str], str | None]:
    """Turn a list of chosen spell names (from a chargen draft or a
    DM-authored companion call) into real Spell objects, looked up
    case-insensitively against SPELL_MENUS[char_class]. Returns
    (spell_objects, same_names_for_spells_prepared, error_or_None) — the
    caller decides whether to treat a non-None error as fatal (chargen's
    finalize_character already validates before this is called, so an error
    here should never actually surface to a player; companion generation
    treats it as a real rejection, matching finalize_character's own
    no-silent-default philosophy).

    For Wizard specifically, ALL chosen names (up to the required 6-spell
    spellbook count) come back in both returned lists — this app has no
    "prepare 4 of 6 daily" reselection tool, so gating to 4 castable would
    make the other 2 permanently inaccessible, worse than not distinguishing
    spellbook-vs-prepared at all. See spells.py's module docstring."""
    menu = SPELL_MENUS.get(char_class)
    if not menu:
        return [], [], None  # non-caster: nothing to build, not an error

    valid_names = {name.lower(): name for tier in menu.values() for name in tier}
    resolved: list[Spell] = []
    invalid: list[str] = []
    for chosen in chosen_names:
        canonical = valid_names.get(chosen.strip().lower())
        if canonical is None:
            invalid.append(chosen)
        else:
            resolved.append(ALL_SPELLS[canonical].model_copy())

    if invalid:
        return [], [], f"Not on {char_class}'s spell menu: {', '.join(invalid)}."

    counts: dict[int, int] = {}
    for spell in resolved:
        counts[spell.level] = counts.get(spell.level, 0) + 1
    required = SPELL_REQUIREMENTS.get(char_class, {})
    mismatches = []
    for tier, needed in required.items():
        got = counts.get(tier, 0)
        if got != needed:
            label = "cantrip(s)" if tier == 0 else f"level-{tier} spell(s)"
            mismatches.append(f"{needed} {label} required, got {got}")
    if mismatches:
        return [], [], f"{char_class} spell count mismatch: {'; '.join(mismatches)}."

    names = [s.name for s in resolved]
    return resolved, names, None


def char_summary(char: Character) -> str:
    """One-line status line for a character."""
    pct = char.current_hp / char.max_hp if char.max_hp else 0
    if char.death_save_failures >= 3:
        health = "DEAD"
    elif char.current_hp == 0:
        health = "DOWNED"
    elif pct < 0.25:
        health = "CRITICAL"
    elif pct < 0.5:
        health = "BLOODIED"
    else:
        health = "OK"

    role = "PC" if char.is_player_controlled else "DM-controlled"
    conds = ", ".join(c.value for c in char.conditions)
    parts = [
        f"{char.name} [{char.char_class} {char.level}, {role}]",
        f"{char.current_hp}/{char.max_hp} HP",
        f"AC {char.ac}",
        health,
    ]
    if conds:
        parts.append(conds)
    if char.exhaustion_level:
        parts.append(f"Exhaustion {char.exhaustion_level}")
    if char.spell_slots:
        slots = [
            f"L{lvl}:{s.max - s.used}/{s.max}"
            for lvl, s in sorted(char.spell_slots.items())
            if s.max > 0
        ]
        parts.append("Slots: " + ", ".join(slots))
    return " | ".join(parts)


def monster_summary(monster: Monster) -> str:
    """One-line status line for a monster, mirroring char_summary — used by
    get_active_encounter so the mechanics model can always ground itself
    against the real registered monster (exact name, current HP, attacks)
    instead of relying on its own conversation memory of what it created."""
    pct = monster.current_hp / monster.max_hp if monster.max_hp else 0
    if monster.current_hp == 0:
        health = "DEFEATED"
    elif pct < 0.25:
        health = "CRITICAL"
    elif pct < 0.5:
        health = "BLOODIED"
    else:
        health = "OK"
    parts = [
        f"{monster.name} [{monster.size.value} {monster.monster_type.value}, CR {monster.cr}]",
        f"{monster.current_hp}/{monster.max_hp} HP",
        f"AC {monster.ac}",
        health,
    ]
    if monster.attacks:
        parts.append("Attacks: " + ", ".join(
            f"{a.name} ({a.to_hit_bonus:+d} to hit, {a.damage_dice} {a.damage_type.value})"
            for a in monster.attacks
        ))
    return " | ".join(parts)


# ─── Dice engine (shared by dice.py's roll_dice and resolution.py's resolve_* tools) ──

_DICE_RE = re.compile(r'^(\d+)?d(\d+)(?:k([hl])(\d+))?([+-]\d+)?$', re.IGNORECASE)


def roll_notation(notation: str) -> tuple[int, str]:
    """Parse and roll dice notation. Returns (total, breakdown_string). Moved out of
    dice.py's roll_dice tool so resolution.py's resolve_* tools can roll dice
    directly without importing a private function across modules."""
    m = _DICE_RE.match(notation.strip())
    if not m:
        raise ValueError(
            f"Unrecognised notation '{notation}'. "
            "Examples: d20, 1d20+5, 2d6-1, 4d6kh3"
        )
    n = int(m.group(1) or 1)
    sides = int(m.group(2))
    keep_dir = (m.group(3) or "").lower()
    keep_n = int(m.group(4)) if m.group(4) else None
    modifier = int(m.group(5)) if m.group(5) else 0

    if not (1 <= n <= 100):
        raise ValueError("Number of dice must be 1–100.")
    if not (2 <= sides <= 1000):
        raise ValueError("Die sides must be 2–1000.")

    rolls = [random.randint(1, sides) for _ in range(n)]

    if keep_dir and keep_n is not None:
        reverse = keep_dir == "h"
        ranked = sorted(range(n), key=lambda i: rolls[i], reverse=reverse)
        keep_idx = set(ranked[:keep_n])
        kept = [rolls[i] for i in range(n) if i in keep_idx]
        # Dropped dice shown in parentheses
        parts = [
            str(rolls[i]) if i in keep_idx else f"({rolls[i]})"
            for i in range(n)
        ]
        total = sum(kept) + modifier
    else:
        parts = [str(r) for r in rolls]
        total = sum(rolls) + modifier

    breakdown = "[" + ", ".join(parts) + "]"
    if modifier:
        breakdown += f" {'+' if modifier > 0 else ''}{modifier}"
    breakdown += f" = {total}"
    return total, breakdown


def critical_damage_notation(notation: str) -> str:
    """Double the dice-count component of a damage notation for a 2024-rules
    critical hit (double the dice, not the flat modifier — "2d6+3" -> "4d6+3").
    Returns the input unchanged if it doesn't parse; grounded Attack.damage_dice
    never uses keep-highest/lowest syntax, so this is a safe transform in
    practice."""
    m = _DICE_RE.match(notation.strip())
    if not m:
        return notation
    n = int(m.group(1) or 1)
    sides = m.group(2)
    keep = f"k{m.group(3)}{m.group(4)}" if m.group(3) and m.group(4) else ""
    modifier = m.group(5) or ""
    return f"{n * 2}d{sides}{keep}{modifier}"


# ─── Shared damage application (used by party.py/combat.py's HP tools AND every ──
# ─── resolution.py tool that applies damage, so there's one source of truth) ────

def apply_damage_to_character(char: Character, amount: int, is_critical: bool = False) -> str:
    """Apply damage (negative amount) or healing (positive amount) to a character.
    Damage hits temp HP first. Mutates `char` in place; does not save — callers
    own the store.save() call.

    is_critical: whether the hit causing this damage was a critical (only
    relevant when the character is already at 0 HP — a crit against a downed
    character costs 2 death save failures instead of 1, per 5e rules).
    Callers with no crit concept (freeform damage, saving throws) leave this
    False; resolve_attack/resolve_pending_action pass their own crit flag.

    Handles the full damage-while-at-0-HP interaction, not just the initial
    drop to 0: damage that drops an already-conscious character to 0 just
    downs them (unchanged from before); damage against an ALREADY-downed
    character instead counts as 1 death save failure (2 on a crit), or kills
    them outright if a single hit's damage is >= max_hp (5e's massive damage
    instant-death rule) — a simplification when several swings from one
    resolve_attack call are summed into one HP update (only the batch's own
    is_critical/total matters, not each individual swing), acceptable since a
    downed ally eating multiple attacks in one action is a rare table
    scenario. Healing back above 0 HP clears the death save tally, per RAW.

    A character with 3+ death save failures is truly dead, not just downed —
    ordinary healing (this function's positive-amount path) does NOT revive
    them; only a dedicated resurrection spell (Revivify, Raise Dead, ...)
    should, and that isn't built yet (needs a "time since death" field to
    enforce windows like Revivify's 1 minute — see design.md's deferred
    list). Mutating a dead character here is always a no-op."""
    if char.death_save_failures >= 3:
        return f"{char.name} is dead — ordinary healing has no effect. Only a resurrection spell (Revivify, Raise Dead, ...) can restore them, and that isn't a supported tool yet."

    prev_hp, prev_temp = char.current_hp, char.temp_hp
    was_already_down = char.current_hp == 0

    if amount < 0:
        dmg = abs(amount)
        instant_death = was_already_down and dmg >= char.max_hp
        if char.temp_hp > 0:
            absorbed = min(char.temp_hp, dmg)
            char.temp_hp -= absorbed
            dmg -= absorbed
        char.current_hp = max(0, char.current_hp - dmg)
    else:
        instant_death = False
        char.current_hp = min(char.max_hp, char.current_hp + amount)

    msg = f"{char.name}: {prev_hp} → {char.current_hp} HP"
    if prev_temp != char.temp_hp:
        msg += f" (temp HP: {prev_temp} → {char.temp_hp})"

    if char.current_hp > 0:
        if char.death_save_successes or char.death_save_failures:
            char.death_save_successes = 0
            char.death_save_failures = 0
    elif amount < 0:
        if instant_death:
            char.death_save_failures = 3
            msg += " — INSTANT DEATH (damage while down met or exceeded max HP)"
        elif was_already_down:
            fails = 2 if is_critical else 1
            char.death_save_failures = min(3, char.death_save_failures + fails)
            msg += (
                f" — {fails} death save failure(s) from taking damage while down "
                f"(tally: {char.death_save_successes} successes, {char.death_save_failures} failures)"
            )
            if char.death_save_failures >= 3:
                msg += " — DIES"
        else:
            msg += " — DOWNED, make death saving throws"
    return msg


def apply_damage_to_monster(monster: Monster, amount: int) -> str:
    """Apply damage (negative amount) or healing (positive amount) to a monster.
    Mutates `monster` in place; does not save — callers own the store.save() call."""
    prev = monster.current_hp
    monster.current_hp = max(0, min(monster.max_hp, monster.current_hp + amount))
    msg = f"{monster.name}: {prev} → {monster.current_hp} HP"
    if monster.current_hp == 0:
        msg += " — DEFEATED"
    return msg


# ─── Reaction gating (resolution.py's resolve_attack pause check) ───────────────

_REACTION_FEATURE_KEYWORDS = (
    "uncanny dodge", "parry", "deflect missiles", "riposte", "war caster",
)


def has_plausible_reaction(char: Character) -> bool:
    """Cheap, deterministic, zero-LLM-cost check for whether a character has any
    real reaction option available right now (a prepared reaction spell with a
    remaining slot, or a known reaction-granting feature/feat) — used only to
    gate resolve_attack's pause-for-reaction logic. Without this, EVERY hit
    against EVERY player-controlled character would pause play regardless of
    whether they have anything to react with, since reaction_available alone
    only tracks "hasn't been spent this round," not "has an option at all"."""
    prepared = set(char.spells_prepared) if char.spells_prepared else None
    for spell in char.spells_known:
        if prepared is not None and spell.name not in prepared:
            continue
        if spell.casting_time != "1 reaction":
            continue
        if spell.level == 0:
            return True
        slot = char.spell_slots.get(spell.level)
        if slot is not None and slot.remaining > 0:
            return True
    features_blob = " ".join(char.features).lower()
    return any(kw in features_blob for kw in _REACTION_FEATURE_KEYWORDS)


def require_current_turn(campaign: Campaign, actor_name: str) -> str | None:
    """Returns an error string if `actor_name` isn't the current combatant during
    an active encounter (case-insensitive match against InitiativeEntry.name), or
    None if the check passes / there's no active encounter to check against.
    Observed live: the mechanics model kept resolving the same player character's
    attack turn after turn — the printed "Round 1 / Initiative Order" block never
    actually advanced, since nothing enforced that the attacker in resolve_attack/
    cast_spell matched whose turn it actually was. A soft correction_note nudge
    (see dm_agent.py's guardrails) wasn't enough to stop this on a
    less-compliant local model; this makes it a hard tool-level refusal instead."""
    enc = campaign.active_encounter
    if not enc or not enc.is_active or not enc.initiative_order:
        return None
    current = next((e for e in enc.initiative_order if e.is_current_turn), None)
    if not current or current.name.lower() == actor_name.lower():
        return None
    return (
        f"It isn't {actor_name}'s turn — the current turn belongs to {current.name} "
        f"(round {enc.round}). Resolve {current.name}'s turn (or whichever combatant "
        f"is actually up) before acting as {actor_name} again — check the "
        f"[LIVE ENCOUNTER STATE] block for the real initiative order rather than "
        f"repeating a previous turn."
    )


# ─── Turn advancement (combat.py's advance_initiative AND resolution.py's ───────
# ─── end_turn=True path — one source of truth for "what happens when a turn ends") ─

def advance_combatant_turn(campaign: Campaign, enc: Encounter) -> str:
    """Advance to the next combatant's turn in `enc`, incrementing the round on
    wraparound, and refresh the new current combatant's reaction_available (a
    reaction refreshes at the start of YOUR turn, not everyone's). Also ticks
    down and expires the new current combatant's ActiveEffects, then resets
    their action-economy budget (InitiativeEntry.actions_remaining/
    bonus_actions_remaining) from 1 + whatever effects are still active —
    see check_and_spend_action_budget for where that budget gets spent, and
    ActiveEffect's docstring for why this is a numeric sum rather than a
    hardcoded "Haste = +1 action" special case. Mutates `enc` and the
    affected Character/Monster in place; does not save."""
    order = enc.initiative_order
    if not order:
        return "Initiative order is empty."

    current_idx = next((i for i, e in enumerate(order) if e.is_current_turn), 0)
    order[current_idx].is_current_turn = False
    next_idx = (current_idx + 1) % len(order)
    order[next_idx].is_current_turn = True

    if next_idx == 0:
        enc.round += 1

    current = order[next_idx]
    combatant = find_char(campaign, current.name) or find_monster(campaign, current.name)
    expired_notes = []
    if combatant is not None:
        combatant.reaction_available = True

        still_active = []
        for effect in combatant.active_effects:
            if effect.duration_rounds is None:
                still_active.append(effect)
                continue
            remaining = effect.duration_rounds - 1
            if remaining <= 0:
                expired_notes.append(f"{effect.name} fades from {combatant.name}.")
            else:
                effect.duration_rounds = remaining
                still_active.append(effect)
        combatant.active_effects = still_active

        current.actions_remaining = 1 + sum(e.extra_actions for e in still_active)
        current.bonus_actions_remaining = 1 + sum(e.extra_bonus_actions for e in still_active)

    msg = (
        f"Round {enc.round} — {current.name}'s turn "
        f"(initiative {current.initiative}, {current.combatant_type.value})."
    )
    if expired_notes:
        msg += " " + " ".join(expired_notes)
    return msg


def spell_action_type(casting_time: str) -> str:
    """Derive a spell's action-economy classification from its own
    `casting_time` field (e.g. "1 action", "1 bonus action", "1 reaction") —
    no separate structured field needed on Spell. Anything else (rituals,
    "1 minute", "10 minutes", ...) defaults to "action" since those aren't
    combat-turn options in the first place."""
    ct = casting_time.strip().lower()
    if "bonus action" in ct:
        return "bonus_action"
    if "reaction" in ct:
        return "reaction"
    return "action"


def check_and_spend_action_budget(
    campaign: Campaign, actor_name: str, action_type: str,
) -> str | None:
    """Returns a refusal string if `actor_name` has already spent this turn's
    Action or Bonus Action (per `action_type`, "action" | "bonus_action"), or
    None (after decrementing the matching InitiativeEntry counter) if the
    spend is legal. No-op (returns None, spends nothing) outside an active
    encounter or for a reaction — reactions are tracked via
    Character/Monster.reaction_available, not here. Same contract/shape as
    require_current_turn, called at the same call sites in resolution.py:
    a hard, code-level backstop for 5e's one-Action-plus-one-Bonus-Action
    turn structure, which previously existed only as prompt prose the model
    could (and did) ignore — see the "snuck in two attacks" incident this
    was built to close."""
    if action_type not in ("action", "bonus_action"):
        return None
    enc = campaign.active_encounter
    if not enc or not enc.is_active or not enc.initiative_order:
        return None
    entry = next((e for e in enc.initiative_order if e.name.lower() == actor_name.lower()), None)
    if entry is None:
        return None

    field = "actions_remaining" if action_type == "action" else "bonus_actions_remaining"
    label = "Action" if action_type == "action" else "Bonus Action"
    remaining = getattr(entry, field)
    if remaining <= 0:
        other_field = "bonus_actions_remaining" if action_type == "action" else "actions_remaining"
        other_label = "Bonus Action" if action_type == "action" else "Action"
        other_remaining = getattr(entry, other_field)
        return (
            f"{actor_name} has already used their {label} this turn (Round {enc.round}). "
            + (
                f"They still have their {other_label} available — resolve that, or end "
                f"the turn, instead of a second {label}."
                if other_remaining > 0
                else f"They have no {label} or {other_label} left — end the turn instead."
            )
        )
    setattr(entry, field, remaining - 1)
    return None


def format_turn_budget_recap(campaign: Campaign, actor_name: str) -> str:
    """Short structured line rendering the just-updated action-economy state
    for `actor_name` — folded into the mechanics resolution report right
    after a successful check_and_spend_action_budget call, so the narrator's
    "so you still have your bonus action" framing (however it phrases it)
    is always a rendering of the same InitiativeEntry the enforcement code
    just mutated, not a separately-authored guess. Returns "" outside an
    active encounter (nothing to recap)."""
    enc = campaign.active_encounter
    if not enc or not enc.is_active:
        return ""
    entry = next((e for e in enc.initiative_order if e.name.lower() == actor_name.lower()), None)
    if entry is None:
        return ""
    combatant = find_char(campaign, actor_name) or find_monster(campaign, actor_name)
    parts = [
        f"Action {'available' if entry.actions_remaining > 0 else 'used'}",
        f"Bonus Action {'available' if entry.bonus_actions_remaining > 0 else 'used'}",
    ]
    if combatant is not None:
        parts.append("Reaction available" if combatant.reaction_available else "Reaction used")
        for effect in combatant.active_effects:
            duration = f"{effect.duration_rounds} round(s) remaining" if effect.duration_rounds is not None else "ongoing"
            parts.append(f"{effect.name}: {duration}")
    return f"\n[Turn state — {actor_name}: " + ", ".join(parts) + ".]"
