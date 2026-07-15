"""
Starting equipment reference data — one reasonable default kit per class,
plus the base weapon/armor stats needed to compute attacks and AC from it.

Not a full PHB equipment-choice branching table (see design.md for the scope
decision) — the goal is "every character starts with class-appropriate,
mechanically-usable gear," not "replicate every PHB equipment option." A
player who wants something different can buy/sell in-game once the economy
tools are available.

WEAPONS/ARMOR also double as the grounding data for the mechanics model's
create_magic_item tool (backend/tools/party.py) — a "+1 Longsword" is looked
up here, never invented. Also the backing data for the item-detail popup
(GET /campaigns/{id}/item-detail, backend/main.py) — category/properties/
weight/value/mastery let it show a real stat block on click, not just a name.

category/weapon_type/properties/versatile_damage_dice/mastery/weight_lbs/
value_gp (weapons) and category/str_requirement/stealth_disadvantage/
weight_lbs/value_gp (armor) transcribed and cross-checked directly against
the in-repo 2024 PHB text's own weapon/armor tables
(`docs/source/core/D&D 5.5E - Player's Handbook.md`), same discipline as
spells.py — only for the entries already listed above, not the full PHB list
(see the scope note above).
"""

from backend.models import DamageType

WEAPONS: dict[str, dict] = {
    "Greataxe":        {"damage_dice": "1d12", "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["heavy", "two-handed"], "versatile_damage_dice": None, "mastery": "cleave", "weight_lbs": 7, "value_gp": 30},
    "Longsword":       {"damage_dice": "1d8",  "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["versatile"], "versatile_damage_dice": "1d10", "mastery": "sap", "weight_lbs": 3, "value_gp": 15},
    "Rapier":          {"damage_dice": "1d8",  "damage_type": DamageType.PIERCING,    "range_ft": "5",      "finesse": True,
                         "category": "martial", "weapon_type": "melee", "properties": ["finesse"], "versatile_damage_dice": None, "mastery": "vex", "weight_lbs": 2, "value_gp": 25},
    "Scimitar":        {"damage_dice": "1d6",  "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": True,
                         "category": "martial", "weapon_type": "melee", "properties": ["finesse", "light"], "versatile_damage_dice": None, "mastery": "nick", "weight_lbs": 3, "value_gp": 25},
    "Shortsword":      {"damage_dice": "1d6",  "damage_type": DamageType.PIERCING,    "range_ft": "5",      "finesse": True,
                         "category": "martial", "weapon_type": "melee", "properties": ["finesse", "light"], "versatile_damage_dice": None, "mastery": "vex", "weight_lbs": 2, "value_gp": 10},
    "Mace":            {"damage_dice": "1d6",  "damage_type": DamageType.BLUDGEONING, "range_ft": "5",      "finesse": False,
                         "category": "simple", "weapon_type": "melee", "properties": [], "versatile_damage_dice": None, "mastery": "sap", "weight_lbs": 4, "value_gp": 5},
    "Quarterstaff":    {"damage_dice": "1d6",  "damage_type": DamageType.BLUDGEONING, "range_ft": "5",      "finesse": False,
                         "category": "simple", "weapon_type": "melee", "properties": ["versatile"], "versatile_damage_dice": "1d8", "mastery": "topple", "weight_lbs": 4, "value_gp": 0.2},
    "Dagger":          {"damage_dice": "1d4",  "damage_type": DamageType.PIERCING,    "range_ft": "20/60",  "finesse": True,
                         "category": "simple", "weapon_type": "melee", "properties": ["finesse", "light", "thrown"], "versatile_damage_dice": None, "mastery": "nick", "weight_lbs": 1, "value_gp": 2},
    "Warhammer":       {"damage_dice": "1d8",  "damage_type": DamageType.BLUDGEONING, "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["versatile"], "versatile_damage_dice": "1d10", "mastery": "push", "weight_lbs": 5, "value_gp": 15},
    "Handaxe":         {"damage_dice": "1d6",  "damage_type": DamageType.SLASHING,    "range_ft": "20/60",  "finesse": True,
                         "category": "simple", "weapon_type": "melee", "properties": ["light", "thrown"], "versatile_damage_dice": None, "mastery": "vex", "weight_lbs": 2, "value_gp": 5},
    "Shortbow":        {"damage_dice": "1d6",  "damage_type": DamageType.PIERCING,    "range_ft": "80/320", "finesse": False,
                         "category": "simple", "weapon_type": "ranged", "properties": ["ammunition", "two-handed"], "versatile_damage_dice": None, "mastery": "vex", "weight_lbs": 2, "value_gp": 25},
    "Longbow":         {"damage_dice": "1d8",  "damage_type": DamageType.PIERCING,    "range_ft": "150/600", "finesse": False,
                         "category": "martial", "weapon_type": "ranged", "properties": ["ammunition", "heavy", "two-handed"], "versatile_damage_dice": None, "mastery": "slow", "weight_lbs": 2, "value_gp": 50},
    "Light Crossbow":  {"damage_dice": "1d8",  "damage_type": DamageType.PIERCING,    "range_ft": "80/320", "finesse": False,
                         "category": "simple", "weapon_type": "ranged", "properties": ["ammunition", "loading", "two-handed"], "versatile_damage_dice": None, "mastery": "slow", "weight_lbs": 5, "value_gp": 25},
    "Sickle":          {"damage_dice": "1d4",  "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": False,
                         "category": "simple", "weapon_type": "melee", "properties": ["light"], "versatile_damage_dice": None, "mastery": "nick", "weight_lbs": 2, "value_gp": 1},
    "Spear":           {"damage_dice": "1d6",  "damage_type": DamageType.PIERCING,    "range_ft": "20/60",  "finesse": False,
                         "category": "simple", "weapon_type": "melee", "properties": ["thrown", "versatile"], "versatile_damage_dice": "1d8", "mastery": "sap", "weight_lbs": 3, "value_gp": 1},
    "Greatsword":      {"damage_dice": "2d6",  "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["heavy", "two-handed"], "versatile_damage_dice": None, "mastery": "graze", "weight_lbs": 6, "value_gp": 50},
    "Flail":           {"damage_dice": "1d8",  "damage_type": DamageType.BLUDGEONING, "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": [], "versatile_damage_dice": None, "mastery": "sap", "weight_lbs": 2, "value_gp": 10},
    "Javelin":         {"damage_dice": "1d6",  "damage_type": DamageType.PIERCING,    "range_ft": "30/120", "finesse": False,
                         "category": "simple", "weapon_type": "melee", "properties": ["thrown"], "versatile_damage_dice": None, "mastery": "slow", "weight_lbs": 2, "value_gp": 0.5},
    # Real reach weapons (the "reach" property doubles melee reach to 10 ft,
    # see weapon_reach_ft below) — not used by any default STARTING_KITS
    # entry, added for add_weapon_attack/create_magic_item lookups and
    # opportunity-attack reach calculation.
    "Glaive":          {"damage_dice": "1d10", "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["heavy", "reach", "two-handed"], "versatile_damage_dice": None, "mastery": "graze", "weight_lbs": 6, "value_gp": 20},
    "Halberd":         {"damage_dice": "1d10", "damage_type": DamageType.SLASHING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["heavy", "reach", "two-handed"], "versatile_damage_dice": None, "mastery": "cleave", "weight_lbs": 6, "value_gp": 20},
    "Lance":           {"damage_dice": "1d10", "damage_type": DamageType.PIERCING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["heavy", "reach", "two-handed"], "versatile_damage_dice": None, "mastery": "topple", "weight_lbs": 6, "value_gp": 10},
    "Pike":            {"damage_dice": "1d10", "damage_type": DamageType.PIERCING,    "range_ft": "5",      "finesse": False,
                         "category": "martial", "weapon_type": "melee", "properties": ["heavy", "reach", "two-handed"], "versatile_damage_dice": None, "mastery": "push", "weight_lbs": 18, "value_gp": 5},
}


def weapon_reach_ft(weapon: dict) -> int:
    """10 ft for a real reach weapon (Glaive/Halberd/Lance/Pike — the
    'reach' property), 5 ft standard melee reach otherwise. Used wherever
    an Attack gets built from a WEAPONS entry (chargen, add_weapon_attack,
    create_magic_item) so Attack.reach_ft reflects the real weapon."""
    return 10 if "reach" in weapon.get("properties", []) else 5

# A weapon's range_ft containing "/" (short/long range, e.g. "80/320") marks
# it ranged — ranged weapons always use DEX for to-hit regardless of the
# finesse flag, which only matters for melee weapons.

ARMOR: dict[str, dict] = {
    "Leather Armor":   {"base_ac": 11, "dex_cap": None, "type": "light",  "category": "light",  "str_requirement": None, "stealth_disadvantage": False, "weight_lbs": 10, "value_gp": 10},
    "Studded Leather": {"base_ac": 12, "dex_cap": None, "type": "light",  "category": "light",  "str_requirement": None, "stealth_disadvantage": False, "weight_lbs": 13, "value_gp": 45},
    "Chain Shirt":     {"base_ac": 13, "dex_cap": 2,    "type": "medium", "category": "medium", "str_requirement": None, "stealth_disadvantage": False, "weight_lbs": 20, "value_gp": 50},
    "Scale Mail":      {"base_ac": 14, "dex_cap": 2,    "type": "medium", "category": "medium", "str_requirement": None, "stealth_disadvantage": True,  "weight_lbs": 45, "value_gp": 50},
    "Chain Mail":      {"base_ac": 16, "dex_cap": 0,    "type": "heavy",  "category": "heavy",  "str_requirement": 13,   "stealth_disadvantage": True,  "weight_lbs": 55, "value_gp": 75},
}

SHIELD_AC_BONUS = 2

# One default kit per class, matching each class's real 2024 PHB "Choose A or
# B" Starting Equipment row, Option A — transcribed and cross-checked directly
# against the in-repo PHB text (docs/source/core/D&D 5.5E - Player's
# Handbook.md), same discipline as WEAPONS/ARMOR above and spells.py.
# "weapon" becomes the character's one mechanical Attack; "gear" carries
# everything else Option A grants (including any second/third weapon a class
# gets, e.g. Rogue's shortbow+dagger alongside its shortsword) as flavor/
# utility items with no separate mechanical stats. Good enough to guarantee
# "not empty-handed, with the real Option A loadout"; players can equip
# alternates via the in-game economy tools if they want more.
STARTING_KITS: dict[str, dict] = {
    "Barbarian": {"weapon": "Greataxe",    "armor": None,             "shield": False, "gear": ["Explorer's Pack", "Handaxe (x4)"],                                                      "gold": 15},
    "Bard":      {"weapon": "Dagger",      "armor": "Leather Armor",  "shield": False, "gear": ["Entertainer's Pack", "Musical Instrument", "Dagger"],                                   "gold": 19},
    "Cleric":    {"weapon": "Mace",        "armor": "Chain Shirt",    "shield": True,  "gear": ["Priest's Pack", "Holy Symbol"],                                                         "gold": 7},
    "Druid":     {"weapon": "Sickle",      "armor": "Leather Armor",  "shield": True,  "gear": ["Explorer's Pack", "Herbalism Kit", "Druidic Focus (Quarterstaff)"],                     "gold": 9},
    "Fighter":   {"weapon": "Greatsword",  "armor": "Chain Mail",     "shield": False, "gear": ["Dungeoneer's Pack", "Flail", "Javelin (x8)"],                                           "gold": 4},
    "Monk":      {"weapon": "Spear",       "armor": None,             "shield": False, "gear": ["Explorer's Pack", "Dagger (x5)", "Artisan's Tools or Musical Instrument"],             "gold": 11},
    "Paladin":   {"weapon": "Longsword",   "armor": "Chain Mail",     "shield": True,  "gear": ["Priest's Pack", "Holy Symbol", "Javelin (x6)"],                                         "gold": 9},
    "Ranger":    {"weapon": "Longbow",     "armor": "Studded Leather","shield": False, "gear": ["Explorer's Pack", "Quiver of 20 Arrows", "Scimitar", "Shortsword", "Druidic Focus (Sprig of Mistletoe)"], "gold": 7},
    "Rogue":     {"weapon": "Shortsword",  "armor": "Leather Armor",  "shield": False, "gear": ["Burglar's Pack", "Thieves' Tools", "Shortbow", "Quiver of 20 Arrows", "Dagger (x2)"],   "gold": 8},
    "Sorcerer":  {"weapon": "Dagger",      "armor": None,             "shield": False, "gear": ["Dungeoneer's Pack", "Arcane Focus (Crystal)", "Spear", "Dagger"],                       "gold": 28},
    "Warlock":   {"weapon": "Dagger",      "armor": "Leather Armor",  "shield": False, "gear": ["Scholar's Pack", "Arcane Focus (Orb)", "Book (Occult Lore)", "Sickle", "Dagger"],       "gold": 15},
    "Wizard":    {"weapon": "Dagger",      "armor": None,             "shield": False, "gear": ["Scholar's Pack", "Spellbook", "Robe", "Arcane Focus (Quarterstaff)", "Dagger"],         "gold": 5},
}
