"""
Hardcoded 2024 ("5.5E") PHB options for character creation.

Used by the chargen tools so the DM agent can explain choices without
hitting ChromaDB every time. `search_rules` remains the fallback for
edge cases, optional sourcebooks, or disputed rulings.

Key 2024 rules change baked into this data: species (formerly "races")
grant no ability score bonus. Ability score increases now come from your
chosen Background (+2/+1, or +1/+1/+1, among three background-specific
abilities), alongside an Origin feat. The RACES dict name and the
'race'/'races' tool-category strings are kept as-is for interface
stability with Character.race and the existing chargen tool signatures —
only the content changed to reflect 2024 rules.
"""

# ── Ability score methods ─────────────────────────────────────────────────────

ABILITY_SCORE_METHODS = {
    "rolled": (
        "Roll four d6s and record the total of the highest three, six times. "
        "Assign the six results to your ability scores in any order. This can "
        "yield very high scores but is random — you might get a powerful "
        "character or a weak one. Best for groups that enjoy variance and "
        "don't mind inequality between characters."
    ),
    "standard_array": (
        "Assign these six values to your scores in any order: 15, 14, 13, 12, 10, 8. "
        "Predictable and balanced — every character using standard array has the same "
        "total stats. Good when the group wants fairness without math."
    ),
    "point_buy": (
        "Start with 8 in every score and spend 27 points to raise them. "
        "Scores cost: 8→9 (1pt), 9→10 (1pt), 10→11 (1pt), 11→12 (1pt), "
        "12→13 (1pt), 13→14 (2pt), 14→15 (2pt — max score before any background "
        "increase). Most flexible and equitable — you design exactly the "
        "character you want."
    ),
}

STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]

POINT_BUY_COSTS = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4, 13: 5, 14: 7, 15: 9}

# ── Species (2024 PHB — no ability score bonuses; see module docstring) ───────

RACES: dict[str, dict] = {
    "Aasimar": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Celestial Resistance: resistance to Necrotic and Radiant damage",
            "Darkvision 60 ft",
            "Healing Hands: touch a creature and roll a number of d4s equal to your "
            "Proficiency Bonus, creature regains that many HP; once per long rest",
            "Light Bearer: know the Light cantrip (CHA)",
            "Celestial Revelation (level 3): as a Bonus Action, transform once per "
            "long rest for 1 minute — choose Heavenly Wings (Fly Speed = Speed), "
            "Inner Radiance (shed light, Radiant damage aura), or Necrotic Shroud "
            "(frighten nearby foes on a failed CHA save)",
        ],
        "flavor": "Mortals carrying a spark of the Upper Planes, marked by a celestial "
                  "presence somewhere in their bloodline or a divine blessing on their life.",
        "good_for": "Any CHA-based class (Paladin, Cleric, Sorcerer, Warlock) wants the "
                    "free healing and radiant utility; the transformation choice suits "
                    "front-line or support builds equally well.",
    },
    "Dragonborn": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Draconic Ancestry: choose a dragon type, which sets your damage type — "
            "Black/Copper: Acid, Blue/Bronze: Lightning, Brass/Gold/Red: Fire, "
            "Green: Poison, Silver/White: Cold",
            "Breath Weapon: replace one attack with a 15 ft cone or 30 ft line "
            "(DEX save DC 8 + CON mod + Proficiency Bonus), 1d10 damage scaling to "
            "4d10 by level 17, uses equal to Proficiency Bonus per long rest",
            "Damage Resistance: resistance to your draconic damage type",
            "Darkvision 60 ft",
            "Draconic Flight (level 5): Bonus Action, Fly Speed = Speed for 10 minutes, "
            "once per long rest",
        ],
        "flavor": "Proud, honorable, and dragon-touched. Dragonborn don't have a homeland "
                  "in most settings but carry themselves with innate dignity and purpose.",
        "good_for": "Paladin, Fighter, Sorcerer (Draconic Bloodline especially thematic).",
    },
    "Dwarf": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Darkvision 120 ft",
            "Dwarven Resilience: resistance to Poison damage, advantage on saves "
            "against being Poisoned",
            "Dwarven Toughness: +1 max HP now, +1 more every level thereafter",
            "Stonecunning: Bonus Action, Tremorsense 60 ft for 10 minutes while on or "
            "touching stone, uses equal to Proficiency Bonus per long rest",
        ],
        "flavor": "Stalwart, stubborn, and deeply proud of their craft and clan. Dwarves "
                  "are reliable allies and relentless enemies. They hold grudges for centuries.",
        "good_for": "Any front-line class — Dwarven Toughness scales HP every level, "
                    "and 120 ft darkvision plus poison resistance suit dungeon-heavy play.",
    },
    "Elf": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Darkvision 60 ft",
            "Elven Lineage: choose Drow, High Elf, or Wood Elf (see subraces) — grants "
            "a cantrip and bonus spells at higher levels",
            "Fey Ancestry: advantage on saving throws against the Charmed condition",
            "Keen Senses: proficiency in Insight, Perception, or Survival (your choice)",
            "Trance: only needs 4 hours of meditation instead of 8 hours of sleep",
        ],
        "subraces": {
            "Drow": "Cantrip: Dancing Lights (CHA). At levels 3 and 5, gain Faerie Fire "
                    "and Darkness once per long rest each. Best for charisma casters.",
            "High Elf": "Cantrip of your choice from the Wizard list (INT). At levels 3 "
                        "and 5, gain a 1st- and 2nd-level Wizard spell once per long rest "
                        "each. Best for INT casters and utility-minded characters.",
            "Wood Elf": "Cantrip: Druidcraft (WIS). At levels 3 and 5, gain Longstrider "
                        "and Pass Without Trace once per long rest each. Best for "
                        "rangers, druids, and stealthy characters.",
        },
        "flavor": "Ancient, graceful, and attuned to magic and nature. Elves live for centuries "
                  "and approach life with patience and elegance.",
        "good_for": "High Elf → Wizard/Arcane Trickster. Wood Elf → Ranger/Druid/Monk. "
                    "Drow → Warlock/Sorcerer/Bard.",
    },
    "Gnome": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Darkvision 60 ft",
            "Gnomish Cunning: advantage on Intelligence, Wisdom, and Charisma saving "
            "throws against magic",
            "Gnomish Lineage: choose Forest Gnome or Rock Gnome (see subraces)",
        ],
        "subraces": {
            "Forest Gnome": "Cantrip: Minor Illusion. Can also cast Speak with Animals "
                            "without a slot. Charismatic and sneaky trickster.",
            "Rock Gnome": "Cantrips: Mending and Prestidigitation. Can build a Tiny "
                          "clockwork device (Tinker's Tools). The more scholarly and "
                          "inventor-minded option.",
        },
        "flavor": "Curious, energetic, and delighted by ideas and invention. Gnomes see the world "
                  "as endlessly fascinating and bring that enthusiasm everywhere.",
        "good_for": "Wizard, Artificer, Arcane Trickster Rogue.",
    },
    "Goliath": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 35,
        "traits": [
            "Giant Ancestry: choose one Giant ancestry boon (see subraces), usable a "
            "number of times equal to Proficiency Bonus per long rest",
            "Large Form (level 5): Bonus Action, become Large for 10 minutes once per "
            "long rest — +10 ft speed and advantage on Strength checks while active",
            "Powerful Build: advantage on checks to end the Grappled condition, counts "
            "as one size larger when determining carrying capacity",
        ],
        "subraces": {
            "Cloud's Jaunt": "Teleport 30 feet to an unoccupied space you can see.",
            "Fire's Burn": "On a hit, deal an extra 1d10 Fire damage.",
            "Frost's Chill": "On a hit, deal an extra 1d6 Cold damage and reduce the "
                             "target's speed by 10 ft until the start of your next turn.",
            "Hill's Tumble": "On a hit, knock the target Prone.",
            "Stone's Endurance": "Reaction: reduce damage taken by 1d12 + CON modifier.",
            "Storm's Thunder": "Reaction when hit by an attack: deal 1d8 Thunder damage "
                               "back to the attacker.",
        },
        "flavor": "Tall, powerfully built, and shaped by a distant mountain or highland "
                  "homeland. Goliaths value personal strength and communal survival.",
        "good_for": "Barbarian, Fighter, or any melee build that wants extra mobility "
                    "and a built-in combat trick from the ancestry boon.",
    },
    "Halfling": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Brave: advantage on saving throws against the Frightened condition",
            "Halfling Nimbleness: move through the space of any creature that is a "
            "size larger than you",
            "Luck: when you roll a 1 on the d20 for a D20 Test, reroll and use the "
            "new result",
            "Naturally Stealthy: can hide even while obscured only by a creature "
            "that is at least one size larger than you",
        ],
        "flavor": "Cheerful, practical, and surprisingly brave for their size. Halflings have a "
                  "knack for getting out of trouble and a love of simple comforts.",
        "good_for": "Rogue, Bard, or any DEX-based class that wants extra reliability "
                    "on rolls.",
    },
    "Human": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Resourceful: gain Heroic Inspiration whenever you finish a Long Rest",
            "Skillful: proficiency in one skill of your choice",
            "Versatile: one Origin feat of your choice (Skilled is a strong default)",
        ],
        "flavor": "The most adaptable and ambitious of species, humans are found everywhere "
                  "and excel at everything. 2024 traded the old flat ability bonus for "
                  "genuine build flexibility — a free skill plus a free Origin feat.",
        "good_for": "Any class — the free Origin feat lets you double up on your "
                    "background's feat theme or pick something completely different.",
    },
    "Orc": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Adrenaline Rush: Bonus Action Dash plus temporary HP equal to your "
            "Proficiency Bonus, uses equal to Proficiency Bonus per short or long rest",
            "Darkvision 120 ft",
            "Relentless Endurance: when reduced to 0 HP but not killed outright, drop "
            "to 1 HP instead, once per long rest",
        ],
        "flavor": "Powerful and resilient, with a cultural emphasis on strength, "
                  "community, and survival. A core PHB species in 2024, no longer a "
                  "'Half-Orc' variant of Human.",
        "good_for": "Barbarian (exceptional — Adrenaline Rush stacks with Rage mobility), "
                    "Fighter, Paladin.",
    },
    "Tiefling": {
        "asi": "None — species grant no ability score bonus in this ruleset; "
               "your Background grants +2/+1 (or +1/+1/+1) among three listed "
               "abilities instead.",
        "speed": 30,
        "traits": [
            "Darkvision 60 ft",
            "Fiendish Legacy: choose Abyssal, Chthonic, or Infernal (see subraces) — "
            "grants a resistance, a cantrip, and bonus spells at higher levels",
            "Otherworldly Presence: know the Thaumaturgy cantrip",
        ],
        "subraces": {
            "Abyssal": "Resistance to Poison damage. Cantrip: Poison Spray. At levels 3 "
                       "and 5, gain Ray of Sickness and Hold Person once per long rest each.",
            "Chthonic": "Resistance to Necrotic damage. Cantrip: Chill Touch. At levels "
                       "3 and 5, gain False Life and Ray of Enfeeblement once per long "
                       "rest each.",
            "Infernal": "Resistance to Fire damage. Cantrip: Fire Bolt. At levels 3 and "
                       "5, gain Hellish Rebuke and Darkness once per long rest each.",
        },
        "flavor": "Descended from a pact with a fiend somewhere in the family line, "
                  "tieflings carry a heritage that makes others distrust them. Great for "
                  "players who want an outsider backstory with strong roleplay hooks.",
        "good_for": "Warlock (natural fit), Sorcerer, Bard, Paladin (fallen/redemption arc).",
    },
}

# ── Classes (2024 PHB) ──────────────────────────────────────────────────────────
# Every class now picks its subclass at level 3, a deliberate 2024 unification
# (verified directly in the source text for 9 of 12 classes; inferred with high
# confidence for the remaining 3 — Cleric, Monk, Sorcerer, Wizard — which follow
# the identical "LEVEL 3: <Class> Subclass" pattern).

CLASSES: dict[str, dict] = {
    "Barbarian": {
        "hit_die": 12,
        "primary_ability": "Strength",
        "saving_throws": ["Strength", "Constitution"],
        "armor": "Light armor, medium armor, shields",
        "weapons": "Simple weapons, martial weapons",
        "skills": {"count": 2, "from": ["Animal Handling", "Athletics", "Intimidation",
                                          "Nature", "Perception", "Survival"]},
        "level_1_features": [
            "Rage (Bonus Action): resistance to Bludgeoning/Piercing/Slashing damage, "
            "bonus melee damage, advantage on Strength checks/saves",
            "Weapon Mastery: use the mastery property of two weapon types you're "
            "proficient with",
            "Unarmored Defense: AC = 10 + DEX mod + CON mod when wearing no armor",
        ],
        "level_3_features": ["Subclass: Berserker, Wild Heart, World Tree, or Zealot"],
        "playstyle": "Get in the enemy's face and be impossible to kill. Rage turns you into "
                     "a damage-sponge that hits extremely hard. Minimal complexity at low "
                     "levels — great for new players.",
        "good_for": "Players who want to be a front-line melee powerhouse with high HP.",
    },
    "Bard": {
        "hit_die": 8,
        "primary_ability": "Charisma",
        "saving_throws": ["Dexterity", "Charisma"],
        "armor": "Light armor",
        "weapons": "Simple weapons, hand crossbows, longswords, rapiers, shortswords",
        "skills": {"count": 3, "from": "any skills"},
        "level_1_features": [
            "Bardic Inspiration (Bonus Action, uses = CHA mod/long rest): give an ally "
            "a d6 to add to a roll",
            "Spellcasting (CHA): spells from the Bard list",
        ],
        "level_3_features": ["Subclass: Dance, Glamour, Lore, or Valor"],
        "playstyle": "The ultimate support and social character. You know a little of everything "
                     "— healing, buffs, debuffs, damage, and social skills. Three skill "
                     "proficiencies and any skills means you're the most versatile character "
                     "in the party.",
        "good_for": "Players who want to be the face of the party, love roleplay, and prefer "
                    "enabling others over doing damage themselves.",
    },
    "Cleric": {
        "hit_die": 8,
        "primary_ability": "Wisdom",
        "saving_throws": ["Wisdom", "Charisma"],
        "armor": "Light armor, medium armor, shields",
        "weapons": "Simple weapons",
        "skills": {"count": 2, "from": ["History", "Insight", "Medicine", "Persuasion", "Religion"]},
        "level_1_features": [
            "Spellcasting (WIS): 3 cantrips, 4 prepared level-1+ spells to start, "
            "growing with level",
            "Divine Order: choose Protector (martial weapon and heavy armor training) "
            "or Thaumaturge (extra cantrip plus an Intelligence check bonus)",
        ],
        "level_3_features": ["Subclass (Domain): Life, Light, Trickery, or War"],
        "playstyle": "One of the most powerful classes in 5e. Clerics can heal, buff, debuff, "
                     "deal damage, and wear armor — all from the same character. Domain "
                     "choice dramatically changes your playstyle (Life = healer, War = "
                     "frontliner, etc.)",
        "good_for": "Players who want to be indispensable to the party and enjoy thematic "
                    "roleplay centered on their deity.",
    },
    "Druid": {
        "hit_die": 8,
        "primary_ability": "Wisdom",
        "saving_throws": ["Intelligence", "Wisdom"],
        "armor": "Light armor, shields (non-metal only)",
        "weapons": "Clubs, daggers, darts, javelins, maces, quarterstaffs, scimitars, sickles, slings, spears",
        "skills": {"count": 2, "from": ["Arcana", "Animal Handling", "Insight", "Medicine",
                                          "Nature", "Perception", "Religion", "Survival"]},
        "level_1_features": [
            "Spellcasting (WIS): 2 cantrips and 4 prepared level-1+ spells to start, "
            "prepared from the full Druid list",
            "Druidic: secret language known only to druids",
        ],
        "level_2_features": [
            "Wild Shape: transform into beasts (limited by CR and level)",
            "Wild Companion: cast Find Familiar without a slot, once per long rest",
        ],
        "level_3_features": ["Subclass (Circle): Land, Moon, Sea, or Stars"],
        "playstyle": "Flexible nature casters with extraordinary utility. Wild Shape is "
                     "transformative — Moon Druids can tank in beast form early. Land Druids "
                     "focus on powerful spells. You'll always have the right tool for the "
                     "situation.",
        "good_for": "Players who love the idea of shapeshifting, control spells, and being "
                    "deeply connected to the natural world.",
    },
    "Fighter": {
        "hit_die": 10,
        "primary_ability": "Strength or Dexterity",
        "saving_throws": ["Strength", "Constitution"],
        "armor": "All armor, shields",
        "weapons": "Simple weapons, martial weapons",
        "skills": {"count": 2, "from": ["Acrobatics", "Animal Handling", "Athletics", "History",
                                          "Insight", "Intimidation", "Perception", "Survival"]},
        "level_1_features": [
            "Fighting Style: choose one specialization",
            "Second Wind (Bonus Action): regain 1d10 + Fighter level HP, 2 uses at level 1",
            "Weapon Mastery: use the mastery property of two weapon types you're "
            "proficient with",
        ],
        "level_2_features": ["Action Surge: take one extra action, once per short/long rest"],
        "level_3_features": ["Subclass: Battle Master, Champion, Eldritch Knight, or Psi Warrior"],
        "playstyle": "The most combat-efficient class. Simple to play but deep to optimize. "
                     "Action Surge is one of the most powerful level 2 features in the game. "
                     "Psi Warrior (psionic combat tricks) is new to the core PHB in 2024.",
        "good_for": "Players who want consistent, reliable combat performance. Great for "
                    "beginners due to simplicity; Battle Master is great for tacticians.",
    },
    "Monk": {
        "hit_die": 8,
        "primary_ability": "Dexterity and Wisdom",
        "saving_throws": ["Strength", "Dexterity"],
        "armor": "None (AC = 10 + DEX + WIS unarmored)",
        "weapons": "Simple weapons, martial weapons with the Light property",
        "skills": {"count": 2, "from": ["Acrobatics", "Athletics", "History", "Insight",
                                          "Religion", "Stealth"]},
        "level_1_features": [
            "Martial Arts: use DEX for unarmed strikes/monk weapons, unarmed strike "
            "die scales with level",
            "Unarmored Defense: AC = 10 + DEX mod + WIS mod",
        ],
        "level_2_features": [
            "Monk's Focus (Focus Points = Monk level, replaces old ki points): fuel for "
            "Flurry of Blows, Patient Defense, Step of the Wind",
            "Uncanny Metabolism: regain expended Focus Points once per long rest as a "
            "Bonus Action after rolling Initiative",
        ],
        "level_3_features": ["Subclass (Warrior of): Mercy, Shadow, The Elements, or The Open Hand"],
        "playstyle": "A highly mobile skirmisher who strikes fast and often. Resource "
                     "management around Focus Points. Requires DEX + WIS to be high — two "
                     "ability scores to optimize. Rewarding but demanding.",
        "good_for": "Players who want to feel like a martial arts master with strong mobility.",
    },
    "Paladin": {
        "hit_die": 10,
        "primary_ability": "Strength and Charisma",
        "saving_throws": ["Wisdom", "Charisma"],
        "armor": "All armor, shields",
        "weapons": "Simple weapons, martial weapons",
        "skills": {"count": 2, "from": ["Athletics", "Insight", "Intimidation", "Medicine",
                                          "Persuasion", "Religion"]},
        "level_1_features": [
            "Lay on Hands: pool of HP equal to 5 × Paladin level, heal with a touch "
            "as a Bonus Action",
            "Spellcasting (CHA): 2 prepared level-1+ spells to start — moved from level "
            "2 in 2024",
            "Weapon Mastery: use the mastery property of two weapon types you're "
            "proficient with",
        ],
        "level_2_features": [
            "Fighting Style",
            "Paladin's Smite: spend a spell slot to add 2d8+ radiant damage on a hit",
        ],
        "level_3_features": ["Channel Divinity", "Subclass (Oath): Devotion, Glory, "
                              "The Ancients, or Vengeance"],
        "playstyle": "A frontline warrior with divine power. Getting Spellcasting at level 1 "
                     "(instead of 2014's level 2) makes early Paladins noticeably stronger. "
                     "Divine Smite still makes critical hits devastating.",
        "good_for": "Players who want a noble, idealistic warrior with both healing and combat "
                    "power, and love the idea of a sacred vow.",
    },
    "Ranger": {
        "hit_die": 10,
        "primary_ability": "Dexterity and Wisdom",
        "saving_throws": ["Strength", "Dexterity"],
        "armor": "Light armor, medium armor, shields",
        "weapons": "Simple weapons, martial weapons",
        "skills": {"count": 3, "from": ["Animal Handling", "Athletics", "Insight", "Investigation",
                                          "Nature", "Perception", "Stealth", "Survival"]},
        "level_1_features": [
            "Spellcasting (WIS): 2 prepared level-1+ spells to start — moved from level "
            "2 in 2024",
            "Favored Enemy: always have Hunter's Mark prepared, cast it twice without a "
            "spell slot (regains on long rest)",
            "Weapon Mastery: use the mastery property of two weapon types you're "
            "proficient with",
        ],
        "level_2_features": ["Deft Explorer", "Fighting Style"],
        "level_3_features": ["Subclass: Beast Master, Fey Wanderer, Gloom Stalker, or Hunter"],
        "playstyle": "The wilderness scout. Getting Spellcasting and free Hunter's Mark uses "
                     "at level 1 (instead of 2014's level 2) makes Rangers noticeably more "
                     "capable out of the gate. Three skills and WIS spellcasting keep you "
                     "versatile.",
        "good_for": "Players who love the idea of a hunter, scout, or beastmaster archetype.",
    },
    "Rogue": {
        "hit_die": 8,
        "primary_ability": "Dexterity",
        "saving_throws": ["Dexterity", "Intelligence"],
        "armor": "Light armor",
        "weapons": "Simple weapons, martial weapons with the Finesse or Light property",
        "skills": {"count": 4, "from": ["Acrobatics", "Athletics", "Deception", "Insight",
                                          "Intimidation", "Investigation", "Perception", "Performance",
                                          "Persuasion", "Sleight of Hand", "Stealth"]},
        "level_1_features": [
            "Expertise: double proficiency on 2 skills of your choice",
            "Sneak Attack: extra 1d6 damage once per turn on a hit with advantage (or "
            "an ally within 5 ft) using a Finesse or Ranged weapon, scales with level",
            "Thieves' Cant: secret language of the criminal underworld",
            "Weapon Mastery: use the mastery property of two weapon types you're "
            "proficient with",
        ],
        "level_2_features": ["Cunning Action: Dash, Disengage, or Hide as a Bonus Action"],
        "level_3_features": ["Subclass: Arcane Trickster, Assassin, Soulknife, or Thief"],
        "playstyle": "The most skill-heavy class in the game. Four skill proficiencies and "
                     "Expertise mean you're exceptional at what you're good at. Sneak Attack "
                     "makes you deal surprisingly large damage. Cunning Action gives unmatched "
                     "tactical mobility.",
        "good_for": "Players who want to be the party's skill expert, infiltrator, or assassin.",
    },
    "Sorcerer": {
        "hit_die": 6,
        "primary_ability": "Charisma",
        "saving_throws": ["Constitution", "Charisma"],
        "armor": "None",
        "weapons": "Daggers, darts, slings, quarterstaffs, light crossbows",
        "skills": {"count": 2, "from": ["Arcana", "Deception", "Insight", "Intimidation",
                                          "Persuasion", "Religion"]},
        "level_1_features": [
            "Spellcasting (CHA): 4 cantrips, 2 prepared level-1+ spells to start",
            "Innate Sorcery (Bonus Action, twice per long rest): advantage on spell "
            "attack rolls and +1 spell save DC for 1 minute",
        ],
        "level_2_features": [
            "Font of Magic: Sorcery Points (level equal), can convert to/from spell slots",
        ],
        "level_3_features": ["Subclass (Sorcery): Aberrant, Clockwork, Draconic, or Wild Magic"],
        "playstyle": "Natural-born magic users who can twist their spells in ways no other "
                     "class can. Fewer spells known than Wizard but more flexibility per "
                     "spell via Metamagic. Innate Sorcery gives an early-level nova option.",
        "good_for": "Players who want a powerful arcane caster with a dramatic innate-magic backstory.",
    },
    "Warlock": {
        "hit_die": 8,
        "primary_ability": "Charisma",
        "saving_throws": ["Wisdom", "Charisma"],
        "armor": "Light armor",
        "weapons": "Simple weapons",
        "skills": {"count": 2, "from": ["Arcana", "Deception", "History", "Intimidation",
                                          "Investigation", "Nature", "Religion"]},
        "level_1_features": [
            "Otherworldly Patron: Archfey, Celestial, Fiend, or Great Old One — shapes "
            "your bonus spells and flavor",
            "Pact Magic (CHA): very few spell slots but they recharge on a SHORT rest",
            "Eldritch Invocations: at least one from level 1 — moved earlier than "
            "2014's level 2",
        ],
        "level_3_features": ["Subclass: same choice as Otherworldly Patron at level 1 — "
                              "features unlock further at level 3"],
        "playstyle": "Short rest recharge means you spam your 1-2 spell slots then recover "
                     "quickly. Eldritch Blast (learned via the Wizard/Sorcerer/Warlock spell "
                     "list) is one of the best cantrips in the game. Invocations let you "
                     "customize heavily from level 1 onward.",
        "good_for": "Players who love the pact/patron backstory and want a social/magical "
                    "character with a dark edge.",
    },
    "Wizard": {
        "hit_die": 6,
        "primary_ability": "Intelligence",
        "saving_throws": ["Intelligence", "Wisdom"],
        "armor": "None",
        "weapons": "Daggers, darts, slings, quarterstaffs, light crossbows",
        "skills": {"count": 2, "from": ["Arcana", "History", "Insight", "Investigation",
                                          "Medicine", "Religion"]},
        "level_1_features": [
            "Spellcasting (INT): largest spell list in the game; spellbook with 6 spells "
            "at level 1",
            "Arcane Recovery: regain spell slots equal to half your level (rounded up), "
            "once per long rest",
        ],
        "level_3_features": ["Subclass (School): Abjurer, Diviner, Evoker, or Illusionist"],
        "playstyle": "The most versatile arcane class — you can prepare different spells each "
                     "day and scribe new ones into your spellbook from scrolls. Fragile (d6 "
                     "hit die, no armor) but the right spell at the right time can end an "
                     "encounter before it starts.",
        "good_for": "Players who enjoy reading, planning, and having the perfect tool for every problem.",
    },
}

# ── Backgrounds (2024 PHB — 16 total) ────────────────────────────────────────
# Each background now grants: an ability score increase among three named
# abilities (+2/+1 or +1/+1/+1, never above 20), an Origin feat, two skill
# proficiencies, and one tool proficiency. `feature` holds a short summary of
# what the granted Origin feat actually does — see get_option_details, which
# prints ability_scores and feat alongside it.

BACKGROUNDS: dict[str, dict] = {
    "Acolyte": {
        "ability_scores": ["Intelligence", "Wisdom", "Charisma"],
        "feat": "Magic Initiate (Cleric)",
        "skills": ["Insight", "Religion"],
        "tools": ["Calligrapher's Supplies"],
        "languages": None,
        "feature": "Magic Initiate (Cleric): learn 2 Cleric cantrips plus one 1st-level "
                   "Cleric spell you can cast once per long rest without a slot.",
        "flavor": "Served a temple or religious order. Know the rituals, hierarchy, and "
                  "doctrine. Ideal for Clerics and Paladins, or characters with a complex "
                  "relationship with faith.",
    },
    "Artisan": {
        "ability_scores": ["Strength", "Dexterity", "Intelligence"],
        "feat": "Crafter",
        "skills": ["Investigation", "Persuasion"],
        "tools": ["Artisan's Tools (your choice)"],
        "languages": None,
        "feature": "Crafter: proficiency with three Artisan's Tools, a 20% discount on "
                   "nonmagical purchases, and the ability to craft temporary gear on a "
                   "long rest.",
        "flavor": "A workshop apprentice who learned a trade from the ground up. Good for "
                  "characters with a craft-driven backstory.",
    },
    "Charlatan": {
        "ability_scores": ["Dexterity", "Constitution", "Charisma"],
        "feat": "Skilled",
        "skills": ["Deception", "Sleight of Hand"],
        "tools": ["Forgery Kit"],
        "languages": None,
        "feature": "Skilled: proficiency in any combination of three skills or tools "
                   "of your choice.",
        "flavor": "A con artist, grifter, or fraudster. You've learned to read people and tell them "
                  "exactly what they want to hear. Great for Rogues and Bards.",
    },
    "Criminal": {
        "ability_scores": ["Dexterity", "Constitution", "Intelligence"],
        "feat": "Alert",
        "skills": ["Sleight of Hand", "Stealth"],
        "tools": ["Thieves' Tools"],
        "languages": None,
        "feature": "Alert: add your Proficiency Bonus to Initiative rolls, and you can "
                   "swap your rolled Initiative with a willing ally's.",
        "flavor": "You've broken the law, whether as a thief, smuggler, spy, or assassin. "
                  "You know how the underworld works. Natural fit for Rogues.",
    },
    "Entertainer": {
        "ability_scores": ["Strength", "Dexterity", "Charisma"],
        "feat": "Musician",
        "skills": ["Acrobatics", "Performance"],
        "tools": ["Musical Instrument (your choice)"],
        "languages": None,
        "feature": "Musician: proficiency with three musical instruments, and you can "
                   "grant Heroic Inspiration to allies who hear you play after a rest.",
        "flavor": "An actor, dancer, musician, or storyteller. You know how to work a crowd. "
                  "Natural fit for Bards, but any social character benefits.",
    },
    "Farmer": {
        "ability_scores": ["Strength", "Constitution", "Wisdom"],
        "feat": "Tough",
        "skills": ["Animal Handling", "Nature"],
        "tools": ["Carpenter's Tools"],
        "languages": None,
        "feature": "Tough: your HP maximum increases by twice your level immediately, "
                   "and by 2 more every level thereafter.",
        "flavor": "You worked the land and tended animals before adventure called. Great "
                  "for characters from small farming communities.",
    },
    "Guard": {
        "ability_scores": ["Strength", "Intelligence", "Wisdom"],
        "feat": "Alert",
        "skills": ["Athletics", "Perception"],
        "tools": ["Gaming Set (your choice)"],
        "languages": None,
        "feature": "Alert: add your Proficiency Bonus to Initiative rolls, and you can "
                   "swap your rolled Initiative with a willing ally's.",
        "flavor": "You stood watch at a gate, wall, or vault. Disciplined, observant, "
                  "and used to a chain of command. Natural for Fighters and Paladins.",
    },
    "Guide": {
        "ability_scores": ["Dexterity", "Constitution", "Wisdom"],
        "feat": "Magic Initiate (Druid)",
        "skills": ["Stealth", "Survival"],
        "tools": ["Cartographer's Tools"],
        "languages": None,
        "feature": "Magic Initiate (Druid): learn 2 Druid cantrips plus one 1st-level "
                   "Druid spell you can cast once per long rest without a slot.",
        "flavor": "Raised in the wild, guiding travelers through dangerous terrain. Great "
                  "for Rangers, Druids, and any wilderness-focused character.",
    },
    "Hermit": {
        "ability_scores": ["Constitution", "Wisdom", "Charisma"],
        "feat": "Healer",
        "skills": ["Medicine", "Religion"],
        "tools": ["Herbalism Kit"],
        "languages": None,
        "feature": "Healer: spend a Healer's Kit use to heal a nearby creature using its "
                   "own Hit Dice, and you reroll 1s on any healing dice.",
        "flavor": "You lived in seclusion — a monk, a hermit sage, or someone who withdrew "
                  "from society. You've had time to think deeply.",
    },
    "Merchant": {
        "ability_scores": ["Constitution", "Intelligence", "Charisma"],
        "feat": "Lucky",
        "skills": ["Animal Handling", "Persuasion"],
        "tools": ["Navigator's Tools"],
        "languages": None,
        "feature": "Lucky: a pool of Luck Points (equal to Proficiency Bonus) you can "
                   "spend for Advantage on your own d20 Tests or Disadvantage on an "
                   "attack roll against you.",
        "flavor": "A trader or caravan apprentice used to fair deals and long roads. "
                  "Good for characters with a trade skill and travel-heavy backstory.",
    },
    "Noble": {
        "ability_scores": ["Strength", "Intelligence", "Charisma"],
        "feat": "Skilled",
        "skills": ["History", "Persuasion"],
        "tools": ["Gaming Set (your choice)"],
        "languages": None,
        "feature": "Skilled: proficiency in any combination of three skills or tools "
                   "of your choice.",
        "flavor": "Born to wealth and status. You know court etiquette, have family connections, "
                  "and are used to getting your way. Creates interesting RP when privilege meets "
                  "the dangers of adventuring.",
    },
    "Sage": {
        "ability_scores": ["Constitution", "Intelligence", "Wisdom"],
        "feat": "Magic Initiate (Wizard)",
        "skills": ["Arcana", "History"],
        "tools": ["Calligrapher's Supplies"],
        "languages": None,
        "feature": "Magic Initiate (Wizard): learn 2 Wizard cantrips plus one 1st-level "
                   "Wizard spell you can cast once per long rest without a slot.",
        "flavor": "A scholar, librarian, or academic. You've spent years studying and know how "
                  "to find obscure information. Perfect for Wizards and any INT-focused character.",
    },
    "Sailor": {
        "ability_scores": ["Strength", "Dexterity", "Wisdom"],
        "feat": "Tavern Brawler",
        "skills": ["Acrobatics", "Perception"],
        "tools": ["Navigator's Tools"],
        "languages": None,
        "feature": "Tavern Brawler: a better Unarmed Strike (1d4 + STR), reroll 1s on "
                   "its damage, proficiency with improvised weapons, and a shove-on-hit "
                   "option once per turn.",
        "flavor": "You served on a ship — merchant, naval, or pirate. You understand the sea "
                  "and the people who live by it. Good for port city campaigns or seafaring stories.",
    },
    "Scribe": {
        "ability_scores": ["Dexterity", "Intelligence", "Wisdom"],
        "feat": "Skilled",
        "skills": ["Investigation", "Perception"],
        "tools": ["Calligrapher's Supplies"],
        "languages": None,
        "feature": "Skilled: proficiency in any combination of three skills or tools "
                   "of your choice.",
        "flavor": "A copyist working in a scriptorium or monastery library, precise and "
                  "detail-obsessed. Pairs well with an INT-focused character.",
    },
    "Soldier": {
        "ability_scores": ["Strength", "Dexterity", "Constitution"],
        "feat": "Savage Attacker",
        "skills": ["Athletics", "Intimidation"],
        "tools": ["Gaming Set (your choice)"],
        "languages": None,
        "feature": "Savage Attacker: once per turn on a weapon hit, roll the weapon's "
                   "damage dice twice and use either result.",
        "flavor": "You served in an army or militia. You know chain of command, tactics, and "
                  "the camaraderie of soldiers. Natural for Fighters and Paladins.",
    },
    "Wayfarer": {
        "ability_scores": ["Dexterity", "Wisdom", "Charisma"],
        "feat": "Lucky",
        "skills": ["Insight", "Stealth"],
        "tools": ["Thieves' Tools"],
        "languages": None,
        "feature": "Lucky: a pool of Luck Points (equal to Proficiency Bonus) you can "
                   "spend for Advantage on your own d20 Tests or Disadvantage on an "
                   "attack roll against you.",
        "flavor": "Raised on the streets, resourceful and quick to adapt. Great for "
                  "Rogues and any city-focused campaign.",
    },
}

# ── Level 1 spell slots by class ─────────────────────────────────────────────
# Paladin and Ranger now get Spellcasting at level 1 (moved up from level 2 in
# 2014), so they get level-1 slots immediately, unlike the old 2014 table.

STARTING_SPELL_SLOTS: dict[str, dict[int, int]] = {
    "Bard":     {1: 2},
    "Cleric":   {1: 2},
    "Druid":    {1: 2},
    "Paladin":  {1: 2},
    "Ranger":   {1: 2},
    "Sorcerer": {1: 2},
    "Warlock":  {1: 1},      # refreshes on short rest
    "Wizard":   {1: 2},
}

# ── Full character-level spell slot progression, for level_up ───────────────
# Standard 5e slot-count-per-character-level tables — unchanged in shape
# between 2014 and 2024 (2024 only moved Paladin/Ranger's start to level 1,
# see below; slot counts once a class is casting were not revised). Authored
# from well-established SRD/PHB knowledge, same basis as Magic Missile/Witch
# Bolt in spells.py, not transcribed line-by-line from the ingested OCR text
# — each dict's level-1 row is cross-checked against STARTING_SPELL_SLOTS
# above for consistency. Keyed by character level -> {spell level: slot count}.

_FULL_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {1: 2},
    2:  {1: 3},
    3:  {1: 4, 2: 2},
    4:  {1: 4, 2: 3},
    5:  {1: 4, 2: 3, 3: 2},
    6:  {1: 4, 2: 3, 3: 3},
    7:  {1: 4, 2: 3, 3: 3, 4: 1},
    8:  {1: 4, 2: 3, 3: 3, 4: 2},
    9:  {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    10: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
    11: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1},
    12: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1},
    13: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1},
    14: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1},
    15: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1},
    16: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1},
    17: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2, 6: 1, 7: 1, 8: 1, 9: 1},
    18: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 1, 7: 1, 8: 1, 9: 1},
    19: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 2, 7: 1, 8: 1, 9: 1},
    20: {1: 4, 2: 3, 3: 3, 4: 3, 5: 3, 6: 2, 7: 2, 8: 1, 9: 1},
}

# Half casters — Paladin, Ranger. 2024 moved their Spellcasting feature to
# level 1 (from 2014's level 2), so this whole table is shifted one
# character level earlier than the 2014 version; level 1 already matches
# STARTING_SPELL_SLOTS ({1: 2}).
_HALF_CASTER_SLOTS: dict[int, dict[int, int]] = {
    1:  {1: 2},
    2:  {1: 2},
    3:  {1: 3},
    4:  {1: 3},
    5:  {1: 4, 2: 2},
    6:  {1: 4, 2: 2},
    7:  {1: 4, 2: 3},
    8:  {1: 4, 2: 3},
    9:  {1: 4, 2: 3, 3: 2},
    10: {1: 4, 2: 3, 3: 2},
    11: {1: 4, 2: 3, 3: 3},
    12: {1: 4, 2: 3, 3: 3},
    13: {1: 4, 2: 3, 3: 3, 4: 1},
    14: {1: 4, 2: 3, 3: 3, 4: 1},
    15: {1: 4, 2: 3, 3: 3, 4: 2},
    16: {1: 4, 2: 3, 3: 3, 4: 2},
    17: {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    18: {1: 4, 2: 3, 3: 3, 4: 3, 5: 1},
    19: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
    20: {1: 4, 2: 3, 3: 3, 4: 3, 5: 2},
}

# Warlock — Pact Magic. Very few slots, but they scale in SLOT LEVEL (not
# just count) with character level, unlike every other caster's table above.
_WARLOCK_SLOTS: dict[int, dict[int, int]] = {
    1:  {1: 1},
    2:  {1: 2},
    3:  {2: 2},
    4:  {2: 2},
    5:  {3: 2},
    6:  {3: 2},
    7:  {4: 2},
    8:  {4: 2},
    9:  {5: 2},
    10: {5: 2},
    11: {5: 3},
    12: {5: 3},
    13: {5: 3},
    14: {5: 3},
    15: {5: 3},
    16: {5: 3},
    17: {5: 4},
    18: {5: 4},
    19: {5: 4},
    20: {5: 4},
}

SPELL_SLOTS_BY_LEVEL: dict[str, dict[int, dict[int, int]]] = {
    "Bard": _FULL_CASTER_SLOTS, "Cleric": _FULL_CASTER_SLOTS, "Druid": _FULL_CASTER_SLOTS,
    "Sorcerer": _FULL_CASTER_SLOTS, "Wizard": _FULL_CASTER_SLOTS,
    "Paladin": _HALF_CASTER_SLOTS, "Ranger": _HALF_CASTER_SLOTS,
    "Warlock": _WARLOCK_SLOTS,
}


def proficiency_bonus_for_level(level: int) -> int:
    """Standard 5e proficiency bonus progression — +2 at levels 1-4, +1 more
    every 4 levels thereafter, capping at +6 for levels 17-20. Unchanged
    between 2014 and 2024."""
    return 2 + (level - 1) // 4

# ── Hit dice (max at level 1) ─────────────────────────────────────────────────
# Unchanged from 2014 — hit die size per class was not touched by the 2024 revision.

HIT_DICE: dict[str, int] = {
    "Barbarian": 12, "Fighter": 10, "Paladin": 10, "Ranger": 10,
    "Bard": 8, "Cleric": 8, "Druid": 8, "Monk": 8, "Rogue": 8, "Warlock": 8,
    "Sorcerer": 6, "Wizard": 6,
}
