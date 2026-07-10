"""
Combat loot generation — rolls real DMG treasure (backend/data/treasure_tables.py)
for the monsters a party just defeated, then enriches it with any adventure-specific
item already tied to one of those monsters or the encounter's location in the canon
Lore Registry (LoreStore). Wired into end_encounter (backend/tools/combat.py) so this
happens automatically and deterministically every time combat ends — see design.md's
Evolution section for why (no reliance on the model remembering to invent loot).

RNG design (this is homebrew, not from the DMG — documented here since it isn't
transcribed like treasure_tables.py's data): individual treasure is rolled per
defeated monster at its own CR tier, matching DMG's own per-monster/per-group
convention. Whether a treasure hoard *additionally* drops is gated by
hoard_drop_chance(), scaled off the toughest defeated monster's CR — low tiers rarely
warrant a hoard beyond pocket change, high tiers almost always do. This is the knob
that makes "beefier monster -> better chance at valuable/magical loot" real.

Magic item rarity/attunement aren't in the DMG's own roll tables (see
treasure_tables.py's docstring) — RARITY_BY_TABLE assigns one rarity per table letter
(A=common..I=legendary, the DMG's own rough correspondence between table letter and
the point at which items start appearing in higher-CR hoards) and _looks_attuned
uses a name-keyword heuristic. Both are best-effort approximations, not a per-item
transcription of all ~250 items' actual individual rarity/attunement — good enough for
a homebrew loot roll; not a substitute for the book if exact fidelity ever matters.

One exception, verified directly against source rather than approximated: the generic
"Weapon/Ammunition/Shield/Armor, +N" entries (Tables F/G/H) DO have an explicit rarity
in the DMG's own text (`AMMUNITION, +1, +2, OR +3` etc.), and it isn't a flat per-table
value — confirmed at "docs/source/core/D&D 5E - Dungeon Master's Guide.md" lines
6134/8585/9269: a weapon/ammunition/shield bonus is uncommon(+1)/rare(+2)/very
rare(+3), but the same bonus on body armor is one tier higher —
rare(+1)/very rare(+2)/legendary(+3). There is no official "+4"/legendary weapon
bonus; legendary-tier weapons in the DMG are specific named items (Vorpal Sword, Holy
Avenger, etc.), not a numeric enchantment. _bonus_rarity_override implements this
exactly and takes priority over RARITY_BY_TABLE's per-table default whenever it
applies.
"""
import random
import re

from backend.data.equipment import WEAPONS
from backend.data.treasure_tables import (
    ART_OBJECT_TABLES, GEM_TABLES, INDIVIDUAL_TREASURE_TABLES, MAGIC_ITEM_TABLES,
    TREASURE_HOARD_TABLES,
)
from backend.models import Currency, Item, Monster
from backend.stores.lore_store import LoreStore
from backend.tools._helpers import roll_notation

RARITY_BY_TABLE = {
    "A": "common", "B": "uncommon", "C": "rare", "D": "very rare", "E": "legendary",
    "F": "rare", "G": "very rare", "H": "very rare", "I": "legendary",
}

WEAPON_LIKE_BONUS_RARITY = {1: "uncommon", 2: "rare", 3: "very rare"}
ARMOR_BONUS_RARITY = {1: "rare", 2: "very rare", 3: "legendary"}
_WEAPON_LIKE_BONUS_RE = re.compile(r"^(?:Weapon|Ammunition|Shield),\s*\+(\d)\b")
_ARMOR_BONUS_RE = re.compile(r"^Armor,\s*\+(\d)\b")


def _bonus_rarity_override(name: str) -> str | None:
    m = _WEAPON_LIKE_BONUS_RE.match(name)
    if m:
        return WEAPON_LIKE_BONUS_RARITY.get(int(m.group(1)))
    m = _ARMOR_BONUS_RE.match(name)
    if m:
        return ARMOR_BONUS_RARITY.get(int(m.group(1)))
    return None

_ATTUNEMENT_HINTS = (
    "ring", "rod", "staff", "wand", "amulet", "cloak", "boots", "belt", "gauntlet",
    "glove", "helm", "circlet", "headband", "crown", "horn of", "instrument of",
    "bracers", "brooch", "mantle", "robe", "cape", "ioun stone", "periapt",
    "medallion", "necklace", "eyes of", "figurine", "slayer", "sun blade",
    "sword of", "dagger of venom", "mace of", "flame tongue", "oathbow",
    "luck blade", "defender", "vorpal", "holy avenger", "frost brand",
    "dwarven thrower", "nine lives stealer", "berserker axe", "scimitar of speed",
    "hammer of thunderbolts", "wings of flying", "armor of", "dragon scale mail",
    "carpet of flying", "crystal ball",
)


def _looks_attuned(name: str) -> bool:
    n = name.lower()
    return any(hint in n for hint in _ATTUNEMENT_HINTS)


def cr_to_numeric(cr: str) -> float:
    cr = cr.strip()
    if "/" in cr:
        num, den = cr.split("/", 1)
        return int(num) / int(den)
    return float(cr)


def cr_to_tier(cr: str) -> int:
    val = cr_to_numeric(cr)
    if val <= 4:
        return 1
    if val <= 10:
        return 2
    if val <= 16:
        return 3
    return 4


def _roll_currency_cells(cells: dict) -> Currency:
    kwargs = {}
    for denom, (dice, mult) in cells.items():
        total, _ = roll_notation(dice)
        kwargs[denom] = total * mult
    return Currency(**kwargs)


def _pick_row(entries: list[tuple], roll: int):
    for entry in entries:
        lo, hi = entry[0], entry[1]
        if lo <= roll <= hi:
            return entry
    raise ValueError(f"no row covers roll {roll} in {entries!r}")


def roll_individual_treasure(cr: str) -> Currency:
    """One creature's (or one identical group's, per DMG convention) pocket change."""
    tier = cr_to_tier(cr)
    entries = INDIVIDUAL_TREASURE_TABLES[tier]
    _, _, cells = _pick_row(entries, random.randint(1, 100))
    return _roll_currency_cells(cells)


def roll_magic_item(table_letter: str) -> Item:
    """One roll on Magic Item Table A-I, resolving a nested sub-roll (e.g. Table G's
    "Figurine of wondrous power") if the entry that hits has one."""
    entries = MAGIC_ITEM_TABLES[table_letter]
    lo, hi, name, sub = _pick_row(entries, random.randint(1, 100))
    if sub:
        die = max(s_hi for _, s_hi, _ in sub)
        _, _, sub_name = _pick_row(sub, random.randint(1, die))
        name = f"{name.split(' (roll d')[0]} ({sub_name})"

    rarity = _bonus_rarity_override(name) or RARITY_BY_TABLE[table_letter]

    # Armor/shield/ammunition bonus entries already name a concrete type (e.g.
    # "Armor, +1 chain mail") — only the generic weapon entry doesn't (the DMG
    # leaves picking a weapon type to the DM), so roll one for a real name
    # ("+2 Longsword") instead of the bare placeholder. Also reformat all of
    # these from the DMG's table-cell phrasing ("Armor, +1 chain mail") into a
    # natural item name ("+1 Chain Mail"), matching how a rolled weapon reads.
    is_weapon = name.startswith("Weapon, +")
    is_shield = name.startswith("Shield, +")
    armor_m = re.match(r"^Armor,\s*\+(\d)\s+(.+)$", name)
    if is_weapon:
        bonus = name.split("+", 1)[1]
        name = f"+{bonus} {random.choice(list(WEAPONS.keys()))}"
    elif is_shield:
        bonus = name.split("+", 1)[1]
        name = f"+{bonus} Shield"
    elif armor_m:
        name = f"+{armor_m.group(1)} {armor_m.group(2).title()}"

    return Item(
        name=name,
        quantity=1,
        magical=True,
        requires_attunement=_looks_attuned(name),
        rarity=rarity,
        item_type="weapon" if is_weapon else ("armor" if (is_shield or armor_m) else "wondrous"),
    )


def _roll_gems_or_art(spec: tuple[str, int, str]) -> list[Item]:
    dice, value_gp, kind = spec
    count, _ = roll_notation(dice)
    table = GEM_TABLES if kind == "gems" else ART_OBJECT_TABLES
    names = table[value_gp]
    return [
        Item(name=random.choice(names), quantity=1, value_gp=value_gp, item_type="misc")
        for _ in range(count)
    ]


def hoard_drop_chance(cr: str) -> float:
    """Homebrew scaling, not DMG rules — see module docstring. Tier 1 hoards are
    genuinely rare beyond an individual monster's own pocket change; tier 4 hoards are
    near-certain, matching "a dragon's hoard" expectations."""
    return {1: 0.15, 2: 0.40, 3: 0.70, 4: 0.95}[cr_to_tier(cr)]


def roll_treasure_hoard(cr: str) -> tuple[Currency, list[Item]]:
    tier = cr_to_tier(cr)
    data = TREASURE_HOARD_TABLES[tier]
    currency = _roll_currency_cells(data["coins"])
    items: list[Item] = []

    lo, hi, gemart, magic_rolls = _pick_row(data["rows"], random.randint(1, 100))
    if gemart:
        items += _roll_gems_or_art(gemart)
    for table_letter, count_notation in magic_rolls:
        # "roll once" parses to the literal count "1", not dice notation.
        count = int(count_notation) if count_notation.isdigit() else roll_notation(count_notation)[0]
        items += [roll_magic_item(table_letter) for _ in range(count)]

    return currency, items


class LootResult:
    def __init__(self, currency: Currency | None = None, items: list[Item] | None = None, notes: list[str] | None = None):
        self.currency = currency or Currency()
        self.items = items or []
        self.notes = notes or []

    def is_empty(self) -> bool:
        return self.currency.to_gp() == 0 and not self.items


def generate_encounter_loot(defeated_monsters: list[Monster]) -> LootResult:
    """Individual treasure per defeated monster (own CR tier) plus, gated by
    hoard_drop_chance() on the toughest one, a single treasure hoard roll."""
    if not defeated_monsters:
        return LootResult()

    result = LootResult()
    for monster in defeated_monsters:
        coins = roll_individual_treasure(monster.cr)
        if coins.to_gp() > 0:
            result.currency = Currency(
                cp=result.currency.cp + coins.cp, sp=result.currency.sp + coins.sp,
                ep=result.currency.ep + coins.ep, gp=result.currency.gp + coins.gp,
                pp=result.currency.pp + coins.pp,
            )
            result.notes.append(f"{monster.name} was carrying pocket change.")

    toughest = max(defeated_monsters, key=lambda m: cr_to_numeric(m.cr))
    if random.random() < hoard_drop_chance(toughest.cr):
        hoard_currency, hoard_items = roll_treasure_hoard(toughest.cr)
        result.currency = Currency(
            cp=result.currency.cp + hoard_currency.cp, sp=result.currency.sp + hoard_currency.sp,
            ep=result.currency.ep + hoard_currency.ep, gp=result.currency.gp + hoard_currency.gp,
            pp=result.currency.pp + hoard_currency.pp,
        )
        result.items += hoard_items
        if hoard_currency.to_gp() > 0 or hoard_items:
            result.notes.append(f"A larger cache turned up, sized to {toughest.name}'s challenge rating.")

    return result


async def enrich_with_adventure_loot(
    lore_store: LoreStore | None,
    books_in_play: list[str] | None,
    defeated_monsters: list[Monster],
    location_name: str | None,
    location_aliases: list[str] | None = None,
) -> list[Item]:
    """Adventure-specific items already tied to one of these monsters or this
    location in the canon Lore Registry — guaranteed additions, not chance-gated,
    since a published adventure ties a specific item to a specific
    monster/location for a real story reason (see prompts.py's Loot section).
    Matches on rolled_up_profile's owned_by/found_at fields (see
    scripts/extract_entities.py's ItemExtractor) against defeated monster names, the
    encounter's location name, and its known aliases — best-effort substring match
    (including aliases specifically so a location referred to differently in the
    extracted item's own text still hits), not a hard link. A published adventure's
    "found near/under X" phrasing can still diverge enough from this to miss — see
    prompts.py's Loot section for the residual search_adventure_literal backstop
    this doesn't replace."""
    if not lore_store or not books_in_play:
        return []

    monster_names = {m.name.lower() for m in defeated_monsters}
    haystacks = set(monster_names)
    if location_name:
        haystacks.add(location_name.lower())
    for alias in location_aliases or []:
        haystacks.add(alias.lower())

    found: list[Item] = []
    for book in books_in_play:
        for entity in await lore_store.all_for_book(book, "item"):
            profile = entity.rolled_up_profile or {}
            owned_by = str(profile.get("owned_by") or "").lower()
            found_at = str(profile.get("found_at") or "").lower()
            if any(h and (h in owned_by or h in found_at) for h in haystacks):
                found.append(Item(
                    name=entity.canonical_name,
                    quantity=1,
                    description=profile.get("description", ""),
                    magical=bool(profile.get("magical", False)),
                    requires_attunement=bool(profile.get("requires_attunement", False)),
                    item_type=profile.get("item_type", "misc"),
                    rarity=profile.get("rarity", ""),
                    lore_entity_id=entity.id,
                    aliases=entity.aliases,
                    source_chunk_ids=entity.source_chunk_ids,
                    spoiler_tier="player_discovered",
                ))
    return found
