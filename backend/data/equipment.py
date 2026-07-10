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
}

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

# One default kit per class. "weapon" becomes the character's one starting
# Attack; "gear" items are flavor/utility only (no mechanical stats) — a
# class that traditionally gets two weapons (e.g. Rogue's rapier + shortbow)
# only gets one as a real Attack, the other listed as gear. Good enough to
# guarantee "not empty-handed"; players can equip alternates via the
# in-game economy tools if they want more.
STARTING_KITS: dict[str, dict] = {
    "Barbarian": {"weapon": "Greataxe",      "armor": None,             "shield": False, "gear": ["Explorer's Pack", "Javelin (x4)"],                       "gold": 10},
    "Bard":      {"weapon": "Rapier",        "armor": "Leather Armor",  "shield": False, "gear": ["Diplomat's Pack", "Lute"],                                "gold": 15},
    "Cleric":    {"weapon": "Mace",          "armor": "Scale Mail",     "shield": True,  "gear": ["Priest's Pack", "Holy Symbol"],                           "gold": 10},
    "Druid":     {"weapon": "Scimitar",      "armor": "Leather Armor",  "shield": True,  "gear": ["Explorer's Pack", "Druidic Focus (Sprig of Mistletoe)"], "gold": 10},
    "Fighter":   {"weapon": "Longsword",     "armor": "Chain Mail",     "shield": True,  "gear": ["Explorer's Pack"],                                        "gold": 10},
    "Monk":      {"weapon": "Quarterstaff",  "armor": None,             "shield": False, "gear": ["Explorer's Pack", "Darts (x10)"],                         "gold": 5},
    "Paladin":   {"weapon": "Longsword",     "armor": "Chain Mail",     "shield": True,  "gear": ["Priest's Pack", "Holy Symbol"],                           "gold": 10},
    "Ranger":    {"weapon": "Shortbow",      "armor": "Leather Armor",  "shield": False, "gear": ["Explorer's Pack", "Quiver of 20 Arrows"],                 "gold": 10},
    "Rogue":     {"weapon": "Rapier",        "armor": "Leather Armor",  "shield": False, "gear": ["Burglar's Pack", "Thieves' Tools", "Shortbow"],           "gold": 10},
    "Sorcerer":  {"weapon": "Dagger",        "armor": None,             "shield": False, "gear": ["Dungeoneer's Pack", "Arcane Focus (Crystal)"],            "gold": 15},
    "Warlock":   {"weapon": "Light Crossbow","armor": "Leather Armor",  "shield": False, "gear": ["Scholar's Pack", "Arcane Focus (Rod)"],                   "gold": 15},
    "Wizard":    {"weapon": "Quarterstaff",  "armor": None,             "shield": False, "gear": ["Scholar's Pack", "Spellbook", "Arcane Focus (Wand)"],     "gold": 15},
}
