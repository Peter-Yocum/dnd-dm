from datetime import date
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class ConditionType(str, Enum):
    BLINDED = "blinded"
    CHARMED = "charmed"
    DEAFENED = "deafened"
    FRIGHTENED = "frightened"
    GRAPPLED = "grappled"
    INCAPACITATED = "incapacitated"
    INVISIBLE = "invisible"
    PARALYZED = "paralyzed"
    PETRIFIED = "petrified"
    POISONED = "poisoned"
    PRONE = "prone"
    RESTRAINED = "restrained"
    STUNNED = "stunned"
    UNCONSCIOUS = "unconscious"


class DamageType(str, Enum):
    ACID = "acid"
    BLUDGEONING = "bludgeoning"
    COLD = "cold"
    FIRE = "fire"
    FORCE = "force"
    LIGHTNING = "lightning"
    NECROTIC = "necrotic"
    PIERCING = "piercing"
    POISON = "poison"
    PSYCHIC = "psychic"
    RADIANT = "radiant"
    SLASHING = "slashing"
    THUNDER = "thunder"


class Skill(str, Enum):
    ATHLETICS = "athletics"
    ACROBATICS = "acrobatics"
    SLEIGHT_OF_HAND = "sleight_of_hand"
    STEALTH = "stealth"
    ARCANA = "arcana"
    HISTORY = "history"
    INVESTIGATION = "investigation"
    NATURE = "nature"
    RELIGION = "religion"
    ANIMAL_HANDLING = "animal_handling"
    INSIGHT = "insight"
    MEDICINE = "medicine"
    PERCEPTION = "perception"
    SURVIVAL = "survival"
    DECEPTION = "deception"
    INTIMIDATION = "intimidation"
    PERFORMANCE = "performance"
    PERSUASION = "persuasion"


# Maps each skill to the ability score that governs it.
SKILL_ABILITY: dict[Skill, str] = {
    Skill.ATHLETICS: "strength",
    Skill.ACROBATICS: "dexterity",
    Skill.SLEIGHT_OF_HAND: "dexterity",
    Skill.STEALTH: "dexterity",
    Skill.ARCANA: "intelligence",
    Skill.HISTORY: "intelligence",
    Skill.INVESTIGATION: "intelligence",
    Skill.NATURE: "intelligence",
    Skill.RELIGION: "intelligence",
    Skill.ANIMAL_HANDLING: "wisdom",
    Skill.INSIGHT: "wisdom",
    Skill.MEDICINE: "wisdom",
    Skill.PERCEPTION: "wisdom",
    Skill.SURVIVAL: "wisdom",
    Skill.DECEPTION: "charisma",
    Skill.INTIMIDATION: "charisma",
    Skill.PERFORMANCE: "charisma",
    Skill.PERSUASION: "charisma",
}


class Attitude(str, Enum):
    FRIENDLY = "friendly"
    HELPFUL = "helpful"
    INDIFFERENT = "indifferent"
    CAUTIOUS = "cautious"
    UNFRIENDLY = "unfriendly"
    SUSPICIOUS = "suspicious"
    HOSTILE = "hostile"
    FEARFUL = "fearful"


class ZoneType(str, Enum):
    MELEE = "melee"          # ≤5 ft
    ADJACENT = "adjacent"    # 6–10 ft
    NEAR = "near"            # 10–60 ft
    FAR = "far"              # 60–150 ft
    DISTANT = "distant"      # >150 ft


class CoverType(str, Enum):
    NONE = "none"
    HALF = "half"
    THREE_QUARTERS = "three_quarters"
    TOTAL = "total"


class AreaType(str, Enum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    UNDERGROUND = "underground"
    AQUATIC = "aquatic"
    AERIAL = "aerial"


class LightingType(str, Enum):
    BRIGHT = "bright"
    DIM = "dim"
    DARKNESS = "darkness"


class LocationScale(str, Enum):
    SITE = "site"      # room, chamber, clearing — feet-scale, turn-by-turn
    REGION = "region"  # town, city, dungeon-as-a-whole, landmark — mile/day-scale


class TravelTerrain(str, Enum):
    ROAD = "road"
    TRAIL = "trail"
    WILDERNESS = "wilderness"
    MOUNTAIN = "mountain"
    SWAMP = "swamp"
    WATER = "water"


class WorldPrepStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class CombatantType(str, Enum):
    CHARACTER = "character"
    MONSTER = "monster"
    NPC = "npc"


class QuestStatus(str, Enum):
    UNKNOWN = "unknown"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class QuestType(str, Enum):
    MAIN = "main"
    SIDE = "side"
    PERSONAL = "personal"
    BOUNTY = "bounty"


class HandoutType(str, Enum):
    LETTER = "letter"
    MAP = "map"
    JOURNAL_PAGE = "journal_page"
    INSCRIPTION = "inscription"
    SCROLL = "scroll"
    DRAWING = "drawing"
    OTHER = "other"


class TimeOfDay(str, Enum):
    DAWN = "dawn"
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    DUSK = "dusk"
    EVENING = "evening"
    NIGHT = "night"
    MIDNIGHT = "midnight"


class EncounterDifficulty(str, Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    DEADLY = "deadly"


class SpellSchool(str, Enum):
    ABJURATION = "abjuration"
    CONJURATION = "conjuration"
    DIVINATION = "divination"
    ENCHANTMENT = "enchantment"
    EVOCATION = "evocation"
    ILLUSION = "illusion"
    NECROMANCY = "necromancy"
    TRANSMUTATION = "transmutation"


class MonsterSize(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    HUGE = "huge"
    GARGANTUAN = "gargantuan"


class MonsterType(str, Enum):
    ABERRATION = "aberration"
    BEAST = "beast"
    CELESTIAL = "celestial"
    CONSTRUCT = "construct"
    DRAGON = "dragon"
    ELEMENTAL = "elemental"
    FEY = "fey"
    FIEND = "fiend"
    GIANT = "giant"
    HUMANOID = "humanoid"
    MONSTROSITY = "monstrosity"
    OOZE = "ooze"
    PLANT = "plant"
    UNDEAD = "undead"


# ─── Shared sub-models ────────────────────────────────────────────────────────

class Currency(BaseModel):
    cp: int = 0   # copper
    sp: int = 0   # silver
    ep: int = 0   # electrum
    gp: int = 0   # gold
    pp: int = 0   # platinum

    def to_gp(self) -> float:
        return self.cp / 100 + self.sp / 10 + self.ep / 2 + self.gp + self.pp * 10


class AbilityScores(BaseModel):
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10

    def modifier(self, score: int) -> int:
        return (score - 10) // 2

    @property
    def str_mod(self) -> int: return self.modifier(self.strength)
    @property
    def dex_mod(self) -> int: return self.modifier(self.dexterity)
    @property
    def con_mod(self) -> int: return self.modifier(self.constitution)
    @property
    def int_mod(self) -> int: return self.modifier(self.intelligence)
    @property
    def wis_mod(self) -> int: return self.modifier(self.wisdom)
    @property
    def cha_mod(self) -> int: return self.modifier(self.charisma)


class SpellSlotLevel(BaseModel):
    max: int
    used: int = 0

    @property
    def remaining(self) -> int:
        return self.max - self.used


class LoreLinked(BaseModel):
    """Shared registry-provenance fields for any entity that can be traced
    back to a precomputed canon Lore Registry entry (see
    scripts/extract_entities.py / backend/stores/lore_store.py). A live
    campaign entity's lore_entity_id is set once at creation/backfill time
    and never silently re-synced afterward — the point of these fields is
    provenance, not a live mirror of canon."""
    lore_entity_id: str | None = None
    aliases: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    spoiler_tier: str = "public"   # "public" | "player_discovered" | "dm_only"


class Item(LoreLinked):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    quantity: int = 1
    weight_lbs: float = 0.0
    value_gp: float = 0.0
    description: str = ""
    magical: bool = False
    requires_attunement: bool = False
    attuned_to: str | None = None
    notes: str = ""
    item_type: str = "misc"   # "weapon" | "armor" | "wondrous" | "consumable" | "quest" | "misc"
    rarity: str = ""          # "" (mundane) | "common".."legendary"/"artifact"


class SpellResolutionType(str, Enum):
    ATTACK_ROLL = "attack_roll"     # vs AC — Fire Bolt, Ray of Frost, Guiding Bolt
    SAVING_THROW = "saving_throw"   # vs a DC — Fireball, Hold Person, Command
    AUTOMATIC = "automatic"         # no roll — Shield, Misty Step, Detect Magic, healing


class Spell(BaseModel):
    name: str
    level: int = 0                        # 0 = cantrip
    school: SpellSchool = SpellSchool.EVOCATION
    casting_time: str = "1 action"
    range: str = "Self"
    components: list[str] = Field(default_factory=list)   # ["V", "S", "M (sand)"]
    duration: str = "Instantaneous"
    description: str = ""
    higher_levels: str | None = None      # "At Higher Levels..." text
    ritual: bool = False
    concentration: bool = False
    classes: list[str] = Field(default_factory=list)

    # Mechanical resolution — schema only; population deferred (see design.md's
    # "Deferred from the combat resolution refactor"). Left unset (AUTOMATIC, no
    # dice) for any spell authored before this existed.
    resolution_type: SpellResolutionType = SpellResolutionType.AUTOMATIC
    save_ability: str | None = None       # lowercase ability name, only for SAVING_THROW
    effect_dice: str = ""                 # "" = no dice component
    damage_type: DamageType | None = None # None = not a damage effect
    is_healing: bool = False              # effect_dice applied as healing, not damage
    half_damage_on_success: bool = True   # SAVING_THROW only
    condition_on_fail: str = ""           # e.g. "paralyzed"; "" = none


class Attack(BaseModel):
    name: str
    to_hit_bonus: int = 0
    damage_dice: str = "1d4"              # e.g. "2d6+3"
    damage_type: DamageType = DamageType.BLUDGEONING
    range_ft: str = "5"                   # "5" for melee, "80/320" for ranged
    notes: str = ""


# ─── Combat stat block (for NPCs that might fight but don't warrant full sheet) ─

class CombatStatBlock(BaseModel):
    max_hp: int
    current_hp: int
    ac: int
    speed: int = 30
    ability_scores: AbilityScores = Field(default_factory=AbilityScores)
    attacks: list[Attack] = Field(default_factory=list)
    damage_resistances: list[DamageType] = Field(default_factory=list)
    damage_immunities: list[DamageType] = Field(default_factory=list)
    condition_immunities: list[ConditionType] = Field(default_factory=list)
    cr: str = "0"


# ─── Character (PC and DM-controlled companions) ──────────────────────────────

class Character(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)

    # Identity
    name: str
    race: str = ""
    char_class: str = ""
    subclass: str | None = None
    background: str | None = None
    alignment: str | None = None
    appearance: str = ""   # physical description — mirrors NPC.physical_description
    level: int = 1
    xp: int = 0

    # Ability scores
    ability_scores: AbilityScores = Field(default_factory=AbilityScores)

    # Proficiency bonus is derived from level (floor((level-1)/4) + 2).
    # Stored explicitly so DMs can override for multiclassing edge cases.
    proficiency_bonus: int = 2

    # Proficiencies
    saving_throw_proficiencies: set[str] = Field(default_factory=set)   # e.g. {"strength", "con"}
    skill_proficiencies: set[Skill] = Field(default_factory=set)
    skill_expertise: set[Skill] = Field(default_factory=set)
    armor_proficiencies: list[str] = Field(default_factory=list)
    weapon_proficiencies: list[str] = Field(default_factory=list)
    tool_proficiencies: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)

    # Combat
    max_hp: int = 1
    current_hp: int = 1
    temp_hp: int = 0
    ac: int = 10
    speed: int = 30
    initiative_modifier: int = 0
    hit_dice_total: str = "1d8"
    hit_dice_remaining: int = 1
    passive_perception: int = 10

    # Death saves — reset to 0 on stabilisation or long rest
    death_save_successes: int = 0
    death_save_failures: int = 0

    # Status
    conditions: list[ConditionType] = Field(default_factory=list)
    exhaustion_level: int = 0             # 0–6; each level compounds debuffs
    inspiration: bool = False
    concentration: str | None = None      # name of spell being concentrated on
    # Refreshes at the start of THIS combatant's own turn (see _advance_combatant_turn),
    # not every round — a reaction can be spent reacting to anyone's turn but only
    # comes back on yours. Declining an offered reaction does not spend it.
    reaction_available: bool = True

    # Spellcasting
    spellcasting_ability: str | None = None   # "intelligence" | "wisdom" | "charisma"
    spell_save_dc: int | None = None
    spell_attack_bonus: int | None = None
    # Keys are spell levels 1–9; cantrips don't consume slots.
    spell_slots: dict[int, SpellSlotLevel] = Field(default_factory=dict)
    spells_known: list[Spell] = Field(default_factory=list)
    spells_prepared: list[str] = Field(default_factory=list)   # spell names

    # Equipment
    attacks: list[Attack] = Field(default_factory=list)
    inventory: list[Item] = Field(default_factory=list)
    currency: Currency = Field(default_factory=Currency)

    # Character detail
    features: list[str] = Field(default_factory=list)   # class/racial features, freeform
    personality_traits: list[str] = Field(default_factory=list)
    ideals: list[str] = Field(default_factory=list)
    bonds: list[str] = Field(default_factory=list)
    flaws: list[str] = Field(default_factory=list)
    notes: str = ""

    is_player_controlled: bool = True


# ─── Monster ──────────────────────────────────────────────────────────────────

class Monster(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    size: MonsterSize = MonsterSize.MEDIUM
    monster_type: MonsterType = MonsterType.HUMANOID
    alignment: str | None = None

    # Core stats
    ac: int = 10
    ac_description: str | None = None    # "natural armor", "chain mail + shield"
    max_hp: int = 1
    current_hp: int = 1
    hp_dice: str = "1d8"
    # Keyed by movement type: {"walk": 30, "fly": 60, "swim": 30}
    speed: dict[str, int] = Field(default_factory=lambda: {"walk": 30})

    ability_scores: AbilityScores = Field(default_factory=AbilityScores)

    # Only the saving throws / skills the monster is actually proficient in
    saving_throw_bonuses: dict[str, int] = Field(default_factory=dict)
    skill_bonuses: dict[str, int] = Field(default_factory=dict)

    # Resistances / immunities
    damage_resistances: list[DamageType] = Field(default_factory=list)
    damage_immunities: list[DamageType] = Field(default_factory=list)
    damage_vulnerabilities: list[DamageType] = Field(default_factory=list)
    condition_immunities: list[ConditionType] = Field(default_factory=list)

    # Senses — keyed by sense type: {"darkvision": 60, "tremorsense": 30}
    senses: dict[str, int] = Field(default_factory=dict)
    passive_perception: int = 10
    languages: list[str] = Field(default_factory=list)

    # Tracked symmetrically with Character.reaction_available (a boss monster could
    # get a Parry-style trait later) though only player-controlled Characters are
    # ever gated on this today — see resolution.py's reaction-pause logic.
    reaction_available: bool = True

    cr: str = "0"
    xp: int = 0

    # Actions
    special_abilities: list[str] = Field(default_factory=list)
    attacks: list[Attack] = Field(default_factory=list)
    bonus_actions: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    legendary_resistance_count: int = 0
    legendary_actions_per_round: int = 0
    legendary_actions: list[str] = Field(default_factory=list)
    lair_actions: list[str] = Field(default_factory=list)

    conditions: list[ConditionType] = Field(default_factory=list)
    notes: str = ""


# ─── NPC ──────────────────────────────────────────────────────────────────────

class Relationship(BaseModel):
    npc_name: str
    description: str   # "old rivals", "owes a debt to", "sister of"


class NPC(LoreLinked):
    id: str = Field(default_factory=lambda: uuid4().hex)

    # Identity
    name: str
    race: str = ""
    occupation: str = ""
    physical_description: str = ""
    location: str = ""

    # State
    is_alive: bool = True
    has_met_party: bool = False
    attitude: Attitude = Attitude.INDIFFERENT
    faction_id: str | None = None

    # Personality — mirrors the 5e character background system
    personality_traits: list[str] = Field(default_factory=list)
    ideals: list[str] = Field(default_factory=list)
    bonds: list[str] = Field(default_factory=list)
    flaws: list[str] = Field(default_factory=list)

    # Knowledge the DM tracks
    motivations: list[str] = Field(default_factory=list)
    secrets: list[str] = Field(default_factory=list)   # won't volunteer
    knowledge: list[str] = Field(default_factory=list) # will share if asked right
    relationships: list[Relationship] = Field(default_factory=list)

    # Commerce
    is_merchant: bool = False
    price_modifier: float = 1.0   # 1.0 = standard PHB prices

    # Loot / possessions
    inventory: list[Item] = Field(default_factory=list)
    currency: Currency = Field(default_factory=Currency)

    # Combat — None if this NPC would never fight
    combat_stats: CombatStatBlock | None = None

    traveling_with_party: bool = False

    notes: str = ""


# ─── Faction ──────────────────────────────────────────────────────────────────

class FactionRelationship(BaseModel):
    faction_name: str
    relationship: str   # "allied", "neutral", "hostile", "rival"
    notes: str = ""


class Faction(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str = ""
    alignment: str | None = None
    goals: list[str] = Field(default_factory=list)
    ranks: list[str] = Field(default_factory=list)   # ordered low → high
    npc_members: list[str] = Field(default_factory=list)    # NPC names
    territory: list[str] = Field(default_factory=list)      # location names
    relationships: list[FactionRelationship] = Field(default_factory=list)
    # Numeric score; the DM decides what thresholds mean (e.g. >50 = ally).
    party_reputation: int = 0
    symbol: str | None = None
    notes: str = ""


# ─── Quest ────────────────────────────────────────────────────────────────────

class QuestObjective(BaseModel):
    description: str
    is_completed: bool = False


class Reward(BaseModel):
    xp: int = 0
    gold: int = 0
    items: list[Item] = Field(default_factory=list)
    # Faction name → reputation points awarded on completion
    faction_reputation: dict[str, int] = Field(default_factory=dict)
    notes: str = ""


class Quest(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str = ""
    quest_type: QuestType = QuestType.SIDE
    status: QuestStatus = QuestStatus.UNKNOWN
    giver: str | None = None          # NPC name
    location_id: str | None = None
    objectives: list[QuestObjective] = Field(default_factory=list)
    rewards: Reward = Field(default_factory=Reward)
    prerequisites: list[str] = Field(default_factory=list)   # quest IDs
    notes: str = ""


# ─── Trap ─────────────────────────────────────────────────────────────────────

class Trap(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str = ""
    location_id: str | None = None
    trigger_description: str = ""
    detection_skill: Skill | None = None
    detection_dc: int | None = None
    disarm_skill: Skill | None = None
    disarm_dc: int | None = None
    effect: str = ""    # "2d10 piercing damage + restrained (DC 14 STR to escape)"
    is_detected: bool = False
    is_triggered: bool = False
    notes: str = ""


# ─── Container ────────────────────────────────────────────────────────────────

class Container(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str = ""
    location_id: str | None = None
    is_locked: bool = False
    lock_dc: int | None = None
    is_open: bool = False
    trap_id: str | None = None    # ID of a Trap if this container is trapped
    contents: list[Item] = Field(default_factory=list)
    currency: Currency = Field(default_factory=Currency)
    notes: str = ""


# ─── Handout ──────────────────────────────────────────────────────────────────

class Handout(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    title: str
    content: str
    handout_type: HandoutType = HandoutType.OTHER
    location_found: str | None = None
    relevant_npcs: list[str] = Field(default_factory=list)
    relevant_locations: list[str] = Field(default_factory=list)
    is_revealed_to_party: bool = False
    notes: str = ""


# ─── Session log ──────────────────────────────────────────────────────────────

class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    session_number: int
    real_date: date | None = None
    summary: str = ""
    key_events: list[str] = Field(default_factory=list)
    # Free-text: what part of the adventure module (chapter/section if known,
    # else a plain description of the current story beat) this session's
    # events leave the party at — used to re-ground the next session's
    # kickoff via a fresh search_rules query instead of relying only on the
    # narrative chronicle. Empty for sessions recorded before this field
    # existed; build_session_kickoff_message treats that as "nothing to
    # re-ground on" rather than an error.
    adventure_progress: str = ""
    xp_awarded: int = 0
    loot_gained: list[Item] = Field(default_factory=list)
    quests_started: list[str] = Field(default_factory=list)    # quest IDs
    quests_completed: list[str] = Field(default_factory=list)  # quest IDs
    notes: str = ""
    # LangGraph thread that produced this session — used to fetch the transcript.
    thread_id: str = ""


# ─── Location ─────────────────────────────────────────────────────────────────

class LocationConnection(BaseModel):
    to_location_id: str
    to_location_name: str   # denormalised for display without a lookup
    direction: str = ""      # "north", "up the stairs", "through the iron door"
    distance_ft: int | None = None        # site scale (room-to-room)
    distance_miles: float | None = None   # region scale (town-to-town)
    terrain: TravelTerrain | None = None  # region scale only
    # is_visible/is_passable are reused at region scale: "has the party
    # discovered this route" / "washed-out bridge, closed pass", etc.
    is_visible: bool = True
    is_passable: bool = True
    notes: str = ""


class Location(LoreLinked):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    description: str = ""
    area_type: AreaType = AreaType.INDOOR
    scale: LocationScale = LocationScale.SITE
    size: str = ""
    lighting: LightingType = LightingType.BRIGHT
    terrain_features: list[str] = Field(default_factory=list)
    points_of_interest: list[str] = Field(default_factory=list)
    hidden_elements: list[str] = Field(default_factory=list)   # DM-only
    connections: list[LocationConnection] = Field(default_factory=list)
    current_npcs: list[str] = Field(default_factory=list)      # NPC names
    notes: str = ""


# ─── Encounter (combat state machine) ────────────────────────────────────────

class InitiativeEntry(BaseModel):
    name: str
    combatant_type: CombatantType
    initiative: int
    is_current_turn: bool = False
    is_surprised: bool = False


class CombatantPosition(BaseModel):
    name: str
    zone: ZoneType = ZoneType.NEAR
    cover: CoverType = CoverType.NONE
    # Populated when a grid map is eventually introduced.
    coordinates: tuple[int, int] | None = None
    notes: str = ""


class PendingAction(BaseModel):
    """An attack roll that connected against a player-controlled target who has a
    real reaction option available — paused before damage is applied so the
    player's next turn can declare a reaction (Shield, Parry, ...) or decline. See
    resolution.py's resolve_attack/resolve_pending_action and dm_agent.py's
    stale-pending auto-decline. Single-slot by design: only one pause can be open
    at a time (the mechanics prompt is instructed to stop calling tools the moment
    one goes pending), which is what keeps this from needing a queue."""
    id: str = Field(default_factory=lambda: uuid4().hex)
    # Only "incoming_attack" is implemented. "incoming_save_damage" (an
    # Absorb-Elements-style reaction to a failed/half-damage save) and
    # "spell_cast" (a Counterspell-style interrupt at the moment of casting) are
    # reserved so a schema migration isn't needed if either is built later — see
    # design.md's "Deferred from the combat resolution refactor".
    trigger_type: str = "incoming_attack"
    attacker_name: str
    target_name: str
    attack_name: str
    to_hit_total: int
    was_natural_20: bool
    target_ac_at_time: int
    pending_damage_notation: str          # already crit-doubled if applicable
    damage_type: str
    remaining_swings: int = 0             # unresolved Multiattack swings after this one
    attacker_wanted_end_turn: bool = False
    prompt_note: str = ""                 # pre-built summary for the injected encounter context


class Encounter(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    location_id: str | None = None
    location_description: str = ""
    round: int = 0
    is_active: bool = False
    initiative_order: list[InitiativeEntry] = Field(default_factory=list)
    combatant_positions: list[CombatantPosition] = Field(default_factory=list)
    environmental_effects: list[str] = Field(default_factory=list)
    difficulty: EncounterDifficulty = EncounterDifficulty.MEDIUM
    xp_budget: int = 0
    xp_awarded: int | None = None
    # Hint for the DM agent when dynamically scaling — "add N more minions if needed".
    monster_count_hint: int | None = None
    notes: str = ""
    pending_action: PendingAction | None = None
    # Set by reveal_loot/add_item_to_character/update_character_currency/
    # create_magic_item (backend/tools/party.py) whenever one fires while this
    # encounter is active — end_encounter's automatic loot generation
    # (backend/tools/loot_generator.py) checks this and skips its own roll if
    # loot was already granted manually mid-fight, so a narrated treasure pile
    # and the automatic roll can't both pay out for the same encounter.
    loot_already_granted: bool = False


# ─── Campaign (root object) ───────────────────────────────────────────────────

class Campaign(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    setting: str = ""
    created_at: date = Field(default_factory=date.today)

    # Controls which rulebook chunks are searched during this campaign.
    books_in_play: list[str] = Field(default_factory=list)
    # Selects the system prompt variant passed to the DM agent.
    system_prompt_variant: str = "standard_5e"

    # Automatic world-prep: a background pass reads books_in_play and
    # pre-populates region-scale locations/connections before play begins.
    world_prep_status: WorldPrepStatus = WorldPrepStatus.NOT_STARTED
    world_prep_error: str = ""

    # World entities
    party: list[Character] = Field(default_factory=list)
    monsters: list[Monster] = Field(default_factory=list)   # available stat blocks
    npcs: list[NPC] = Field(default_factory=list)
    factions: list[Faction] = Field(default_factory=list)
    quests: list[Quest] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    containers: list[Container] = Field(default_factory=list)
    traps: list[Trap] = Field(default_factory=list)
    handouts: list[Handout] = Field(default_factory=list)
    sessions: list[Session] = Field(default_factory=list)

    # Active state
    active_encounter: Encounter | None = None
    current_location_id: str | None = None
    party_treasury: Container = Field(
        default_factory=lambda: Container(
            name="Party Treasury",
            description="Shared party loot and funds",
        )
    )

    # In-game calendar / time
    in_game_date: str = ""    # free-form to support any calendar system
    time_of_day: TimeOfDay = TimeOfDay.MORNING
    days_elapsed: int = 0
    # days_elapsed value when the party last took a long rest — used to enforce
    # the once-per-24-hours rule without parsing in_game_date strings.
    last_long_rest_day: int = 0
    current_weather: str = ""

    session_count: int = 0
    notes: str = ""

    # Active safety flags (X-card) — topics the table has asked the DM agent to
    # steer away from. Cleared by the DM once handled; the permanent audit trail
    # lives in `notes`.
    safety_flags: list[str] = Field(default_factory=list)
