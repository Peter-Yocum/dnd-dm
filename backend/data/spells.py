"""Level-1 spell menus for interactive Session 0 spell selection — a curated
subset of real 5e 2024 spells (not the full PHB list), mirroring equipment.py's
"canonical data + class references" shape but for a MENU to choose from, not
an auto-assigned kit.

Content transcribed and mechanically verified directly against the in-repo
2024 PHB text (`docs/source/core/D&D 5.5E - Player's Handbook.md`), correcting
OCR noise found there (e.g. "ldlO" -> "1d10") — every spell below except Magic
Missile and Witch Bolt was read from that source. Those two are authored from
well-established, high-confidence 5e knowledge (both are iconic, essentially
unchanged core spells) because their full stat blocks aren't present in the
ingested text — only index/reference mentions exist, no full write-up.

Level-1 cantrip/spell counts are FLAT counts per the 2024 PHB (not an
ability-modifier formula like 2014's "prepared spells" — every class's
spellcasting section literally states "choose N cantrips" / "choose N level-1
spells"). See SPELL_REQUIREMENTS. Menus are sized larger than the required
count (see design.md for the sizing rationale) so a choice is real, not an
exact-match "take these." Wizard's level-1 menu is deliberately larger than
its 6-spell requirement.

Known simplifications, stated explicitly rather than silently:
- Several spells (Bless, Guidance, Shield of Faith, Hunter's Mark, Hex, Mage
  Armor, Command, Faerie Fire, Sanctuary-adjacent buffs) modify OTHER, future
  rolls rather than resolving their own damage/save — `cast_spell` has no
  buff-tracking system, so these are `AUTOMATIC` with empty `effect_dice`,
  same "no mechanical roll — narrate it" pattern already used for Shield's
  reaction trigger elsewhere in this app. This is an accepted, already-
  established scope boundary, not new.
- Damage-dice modifiers on healing/force spells (Cure Wounds, Healing Word,
  Magic Missile) use "+3" as a representative level-1 spellcasting modifier,
  not a value dynamically computed per-caster — `effect_dice` is a flat
  string on the Spell model, it can't template a specific character's mod.
- Spare the Dying's real effect (auto-stabilize a 0-HP creature) isn't
  mechanically enforced by `cast_spell` — there's no direct "stabilize"
  function separate from rolling 3 successes via `resolve_death_save`. Left
  as AUTOMATIC/narrate for now; a real implementation would set
  `death_save_successes = 3` directly. Not built here — out of this plan's
  stated scope.
- Chromatic Orb lets the caster choose the damage type in real play; fixed to
  FIRE here for a single deterministic `Spell` entry, noted in its description.
- Goodberry's real "10 berries, 1 HP each" effect is simplified to AUTOMATIC/
  no dice (utility/rations flavor) rather than a tiny scripted heal — not a
  combat-relevant spell in practice.

`Spell.ritual` is set True for the 4 menu spells that actually carry the
Ritual tag in the 2024 PHB (Detect Magic, Identify, Comprehend Languages,
Speak with Animals) — none of the rest in this curated menu are rituals.
Per the PHB's general "Casting Without Slots" rule, any prepared ritual spell
can be cast as a Ritual by any class (10 minutes longer, no slot spent); this
app doesn't model Wizard's separate "Ritual Adept" bonus of skipping the
prepared requirement, since Wizard's spellbook-vs-prepared split is already
flattened here (see the Wizard deviation note in design.md) — every known
Wizard spell is already "prepared," so the bonus would be a no-op anyway.
See `cast_spell`'s `as_ritual` parameter in resolution.py.
"""

from backend.models import DamageType, Spell, SpellResolutionType

# ─── Canonical spell objects ───────────────────────────────────────────────────

ALL_SPELLS: dict[str, Spell] = {
    # Cantrips (level 0)
    "Vicious Mockery": Spell(
        name="Vicious Mockery", level=0, school="enchantment", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="wisdom",
        effect_dice="1d6", damage_type=DamageType.PSYCHIC, half_damage_on_success=False,
        description="A string of insults laced with subtle enchantments. On a failed WIS save, "
                    "the target takes psychic damage and has disadvantage on its next attack roll.",
    ),
    "Minor Illusion": Spell(
        name="Minor Illusion", level=0, school="illusion", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Create a sound or an image of an object — no mechanical roll, narrate its use.",
    ),
    "Mage Hand": Spell(
        name="Mage Hand", level=0, school="conjuration", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="A spectral floating hand manipulates objects at range — no mechanical roll.",
    ),
    "Prestidigitation": Spell(
        name="Prestidigitation", level=0, school="transmutation", casting_time="1 action", range="10 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="A minor magical trick (light a candle, clean an object, a sensory effect) — no mechanical roll.",
    ),
    "Guidance": Spell(
        name="Guidance", level=0, school="divination", casting_time="1 action", range="Touch",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Touch a willing creature; it adds 1d4 to one ability check before the spell ends — buff, no self-contained roll.",
    ),
    "Sacred Flame": Spell(
        name="Sacred Flame", level=0, school="evocation", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="dexterity",
        effect_dice="1d8", damage_type=DamageType.RADIANT, half_damage_on_success=False,
        description="Radiant light descends on a creature; a successful DEX save avoids it entirely (ignores cover).",
    ),
    "Thaumaturgy": Spell(
        name="Thaumaturgy", level=0, school="transmutation", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="A minor supernatural manifestation (booming voice, flickering flames, tremors) — no mechanical roll.",
    ),
    "Spare the Dying": Spell(
        name="Spare the Dying", level=0, school="necromancy", casting_time="1 action", range="15 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="A creature at 0 HP becomes stable. Not mechanically auto-enforced by cast_spell today "
                    "(no direct stabilize function) — narrate it and record the character as stable.",
    ),
    "Toll the Dead": Spell(
        name="Toll the Dead", level=0, school="necromancy", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="wisdom",
        effect_dice="1d8", damage_type=DamageType.NECROTIC, half_damage_on_success=False,
        description="A dolorous bell tolls; on a failed WIS save the target takes necrotic damage "
                    "(1d12 instead of 1d8 if it's already missing HP — simplified to 1d8 here).",
    ),
    "Produce Flame": Spell(
        name="Produce Flame", level=0, school="conjuration", casting_time="1 bonus action", range="60 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d8", damage_type=DamageType.FIRE,
        description="A flickering flame in your hand also lights the way; hurl it as a ranged spell attack.",
    ),
    "Thorn Whip": Spell(
        name="Thorn Whip", level=0, school="transmutation", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d6", damage_type=DamageType.PIERCING,
        description="A thorny vine-whip lashes a creature in range with a melee spell attack; can pull a Large-or-smaller target closer on a hit.",
    ),
    "Druidcraft": Spell(
        name="Druidcraft", level=0, school="transmutation", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="A minor nature trick (predict weather, bloom a flower, a sensory effect) — no mechanical roll.",
    ),
    "Fire Bolt": Spell(
        name="Fire Bolt", level=0, school="evocation", casting_time="1 action", range="120 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d10", damage_type=DamageType.FIRE,
        description="A mote of fire streaks toward a creature or object; ranged spell attack.",
    ),
    "Ray of Frost": Spell(
        name="Ray of Frost", level=0, school="evocation", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d8", damage_type=DamageType.COLD,
        description="A frigid beam streaks toward a creature; ranged spell attack, also slows the target's speed by 10 feet on a hit.",
    ),
    "Eldritch Blast": Spell(
        name="Eldritch Blast", level=0, school="evocation", casting_time="1 action", range="120 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d10", damage_type=DamageType.FORCE,
        description="A beam of crackling energy; ranged spell attack. A Warlock's signature cantrip.",
    ),
    "Chill Touch": Spell(
        name="Chill Touch", level=0, school="necromancy", casting_time="1 action", range="Touch",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d10", damage_type=DamageType.NECROTIC,
        description="Channel the chill of the grave; melee spell attack, target can't regain HP until the end of your next turn on a hit.",
    ),

    # Level-1 spells
    "Healing Word": Spell(
        name="Healing Word", level=1, school="abjuration", casting_time="1 bonus action", range="60 feet",
        resolution_type=SpellResolutionType.AUTOMATIC, effect_dice="2d4+3", is_healing=True,
        description="A creature you can see within range regains hit points — no roll needed, always effective.",
    ),
    "Cure Wounds": Spell(
        name="Cure Wounds", level=1, school="abjuration", casting_time="1 action", range="Touch",
        resolution_type=SpellResolutionType.AUTOMATIC, effect_dice="2d8+3", is_healing=True,
        description="A creature you touch regains hit points — no roll needed, always effective.",
    ),
    "Dissonant Whispers": Spell(
        name="Dissonant Whispers", level=1, school="enchantment", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="wisdom",
        effect_dice="3d6", damage_type=DamageType.PSYCHIC, half_damage_on_success=True,
        description="A discordant melody in the target's mind; on a failed WIS save it takes psychic damage and flees.",
    ),
    "Thunderwave": Spell(
        name="Thunderwave", level=1, school="evocation", casting_time="1 action", range="Self",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="constitution",
        effect_dice="2d8", damage_type=DamageType.THUNDER, half_damage_on_success=True,
        description="A wave of thunderous force in a 15-foot cube from you; each creature makes a CON save, pushed back on a failure.",
    ),
    "Faerie Fire": Spell(
        name="Faerie Fire", level=1, school="evocation", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Outlines creatures in glowing light on a failed DEX save, granting advantage on attacks against them — buff, no self-contained roll.",
    ),
    "Charm Person": Spell(
        name="Charm Person", level=1, school="enchantment", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="wisdom",
        condition_on_fail="charmed",
        description="One humanoid makes a WIS save (with advantage if you're fighting it); on a failure it becomes charmed and friendly to you.",
    ),
    "Bless": Spell(
        name="Bless", level=1, school="enchantment", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Up to three creatures add 1d4 to attack rolls and saving throws until the spell ends — buff, no self-contained roll.",
    ),
    "Guiding Bolt": Spell(
        name="Guiding Bolt", level=1, school="evocation", casting_time="1 action", range="120 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="4d6", damage_type=DamageType.RADIANT,
        description="A bolt of light hurled at a creature; ranged spell attack, the next attack against it before your next turn has advantage on a hit.",
    ),
    "Shield of Faith": Spell(
        name="Shield of Faith", level=1, school="abjuration", casting_time="1 bonus action", range="60 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="A shimmering field grants a creature +2 AC for the duration — buff, no self-contained roll.",
    ),
    "Entangle": Spell(
        name="Entangle", level=1, school="conjuration", casting_time="1 action", range="90 feet",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="strength",
        condition_on_fail="restrained",
        description="Grasping plants sprout in a 20-foot square; each creature there makes a STR save or becomes restrained.",
    ),
    "Command": Spell(
        name="Command", level=1, school="enchantment", casting_time="1 action", range="60 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Speak a one-word command (Approach/Drop/Flee/Grovel/Halt); on a failed WIS save the target obeys — no self-contained damage roll.",
    ),
    "Hunter's Mark": Spell(
        name="Hunter's Mark", level=1, school="divination", casting_time="1 bonus action", range="90 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Mark a creature as your quarry; your attacks against it deal an extra 1d6 force damage — buff modifying future attacks, no self-contained roll.",
    ),
    "Goodberry": Spell(
        name="Goodberry", level=1, school="conjuration", casting_time="1 action", range="Self",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Ten magic berries appear; eating one (a bonus action) restores 1 HP and a day's nourishment — utility, simplified to no mechanical roll here.",
    ),
    "Speak with Animals": Spell(
        name="Speak with Animals", level=1, school="divination", casting_time="1 action or ritual", range="Self",
        ritual=True,
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Comprehend and communicate with beasts for the duration — utility, no mechanical roll.",
    ),
    "Magic Missile": Spell(
        name="Magic Missile", level=1, school="evocation", casting_time="1 action", range="120 feet",
        resolution_type=SpellResolutionType.AUTOMATIC, effect_dice="3d4+3", damage_type=DamageType.FORCE,
        description="Three darts of magical force automatically strike — no attack roll, always hits.",
    ),
    "Shield": Spell(
        name="Shield", level=1, school="abjuration", casting_time="1 reaction", range="Self",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="An invisible barrier grants +5 AC (including against the triggering attack) until the start of your next turn — the canonical resolve_pending_action Shield reaction.",
    ),
    "Chromatic Orb": Spell(
        name="Chromatic Orb", level=1, school="evocation", casting_time="1 action", range="90 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="3d8", damage_type=DamageType.FIRE,
        description="Hurl an orb of energy; ranged spell attack. Real play lets you choose the damage type "
                    "(acid/cold/fire/lightning/poison/thunder) — fixed to fire here for a single deterministic entry.",
    ),
    "Burning Hands": Spell(
        name="Burning Hands", level=1, school="evocation", casting_time="1 action", range="Self",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="dexterity",
        effect_dice="3d6", damage_type=DamageType.FIRE, half_damage_on_success=True,
        description="A sheet of flame from your hands in a 15-foot cone; each creature makes a DEX save.",
    ),
    "Mage Armor": Spell(
        name="Mage Armor", level=1, school="abjuration", casting_time="1 action", range="Touch",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="An unarmored willing creature's AC becomes 13 + DEX modifier for 8 hours — buff, no self-contained roll.",
    ),
    "Hex": Spell(
        name="Hex", level=1, school="enchantment", casting_time="1 bonus action", range="90 feet",
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Curse a creature; your attacks against it deal an extra 1d6 necrotic damage — buff modifying future attacks, no self-contained roll.",
    ),
    "Witch Bolt": Spell(
        name="Witch Bolt", level=1, school="evocation", casting_time="1 action", range="30 feet",
        resolution_type=SpellResolutionType.ATTACK_ROLL, effect_dice="1d12", damage_type=DamageType.LIGHTNING,
        description="A sustained arc of lightning; ranged spell attack. Full stat block not present in the "
                    "in-repo PHB text (index-only) — authored from well-established 5e knowledge.",
    ),
    "Arms of Hadar": Spell(
        name="Arms of Hadar", level=1, school="conjuration", casting_time="1 action", range="Self",
        resolution_type=SpellResolutionType.SAVING_THROW, save_ability="strength",
        effect_dice="2d6", damage_type=DamageType.NECROTIC, half_damage_on_success=True,
        description="Tendrils erupt in a 10-foot emanation from you; each creature there makes a STR save.",
    ),
    "Detect Magic": Spell(
        name="Detect Magic", level=1, school="divination", casting_time="1 action or ritual", range="Self",
        ritual=True,
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Sense magical effects within 30 feet for the duration — utility, no mechanical roll.",
    ),
    "Identify": Spell(
        name="Identify", level=1, school="divination", casting_time="1 minute or ritual", range="Touch",
        ritual=True,
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Learn a touched item's magical properties, or a touched creature's active spells — utility, no mechanical roll.",
    ),
    "Comprehend Languages": Spell(
        name="Comprehend Languages", level=1, school="divination", casting_time="1 action or ritual", range="Self",
        ritual=True,
        resolution_type=SpellResolutionType.AUTOMATIC,
        description="Understand any spoken/written language for the duration — utility, no mechanical roll.",
    ),
}

# ─── Per-class curated menus (name references into ALL_SPELLS) ────────────────
# Ordered with PHB-recommended starters first where confirmed (Cleric cantrips:
# Guidance/Sacred Flame/Thaumaturgy; Cleric level-1: Bless/Cure Wounds/Guiding
# Bolt/Shield of Faith — both directly from the source text). Other classes
# ordered sensibly (most broadly useful first) without an explicit sourced
# "recommended" callout for every one.

SPELL_MENUS: dict[str, dict[int, list[str]]] = {
    "Bard": {
        0: ["Vicious Mockery", "Minor Illusion", "Mage Hand", "Prestidigitation"],
        1: ["Healing Word", "Cure Wounds", "Dissonant Whispers", "Thunderwave", "Faerie Fire", "Charm Person"],
    },
    "Cleric": {
        0: ["Guidance", "Sacred Flame", "Thaumaturgy", "Spare the Dying", "Toll the Dead"],
        1: ["Bless", "Cure Wounds", "Guiding Bolt", "Shield of Faith", "Healing Word"],
    },
    "Druid": {
        0: ["Guidance", "Produce Flame", "Thorn Whip", "Druidcraft"],
        1: ["Cure Wounds", "Healing Word", "Entangle", "Thunderwave", "Faerie Fire"],
    },
    "Paladin": {
        1: ["Bless", "Cure Wounds", "Shield of Faith", "Command"],
    },
    "Ranger": {
        1: ["Cure Wounds", "Hunter's Mark", "Goodberry", "Speak with Animals"],
    },
    "Sorcerer": {
        0: ["Fire Bolt", "Ray of Frost", "Mage Hand", "Minor Illusion", "Prestidigitation"],
        1: ["Magic Missile", "Shield", "Chromatic Orb", "Burning Hands"],
    },
    "Warlock": {
        0: ["Eldritch Blast", "Chill Touch", "Minor Illusion", "Prestidigitation"],
        1: ["Hex", "Witch Bolt", "Arms of Hadar", "Charm Person"],
    },
    "Wizard": {
        0: ["Fire Bolt", "Ray of Frost", "Mage Hand", "Minor Illusion", "Prestidigitation"],
        1: ["Magic Missile", "Shield", "Burning Hands", "Mage Armor", "Chromatic Orb", "Detect Magic", "Identify", "Comprehend Languages"],
    },
}

# ─── Required counts at level 1 (flat counts per the 2024 PHB, not a formula) ──
# Every key here must have a matching key in backend/data/fivee_options.py's
# STARTING_SPELL_SLOTS, and vice versa (cross-checked in verification).

SPELLCASTING_ABILITY: dict[str, str] = {
    "Bard": "charisma",
    "Cleric": "wisdom",
    "Druid": "wisdom",
    "Paladin": "charisma",
    "Ranger": "wisdom",
    "Sorcerer": "charisma",
    "Warlock": "charisma",
    "Wizard": "intelligence",
}

SPELL_REQUIREMENTS: dict[str, dict[int, int]] = {
    "Bard": {0: 2, 1: 4},
    "Cleric": {0: 3, 1: 4},
    "Druid": {0: 2, 1: 4},
    "Paladin": {1: 2},
    "Ranger": {1: 2},
    "Sorcerer": {0: 4, 1: 2},
    "Warlock": {0: 2, 1: 2},
    # Wizard: 6 spellbook spells required, but no reselection tool exists (no
    # "prepare 4 of 6 daily" support) — all 6 chosen spells are stored as both
    # spells_known and spells_prepared, i.e. all castable, not gated to 4. See
    # design.md's deferred list for the stated RAW deviation this represents.
    "Wizard": {0: 3, 1: 6},
}
