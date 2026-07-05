"""
Character generation tools — used only during Session 0.

The DM agent uses these to walk a player through character creation,
update the live draft preview, and commit the finished character to the
campaign when everything is confirmed.
"""

import random

from langchain_core.tools import tool

from backend.data.fivee_options import (
    ABILITY_SCORE_METHODS,
    BACKGROUNDS,
    CLASSES,
    RACES,
    STANDARD_ARRAY,
)
from backend.data.spells import ALL_SPELLS, SPELL_MENUS, SPELL_REQUIREMENTS
from backend.models import AbilityScores, Character
from backend.stores.campaign_store import CampaignStore
from backend.stores.draft_store import DraftStore
from backend.tools._helpers import (
    build_spells_known, derive_level1_stats, derive_saving_throw_proficiencies, derive_spellcasting_stats,
)


def _roll_4d6_drop_lowest() -> tuple[int, list[int]]:
    rolls = [random.randint(1, 6) for _ in range(4)]
    total = sum(sorted(rolls)[1:])
    return total, rolls


def _build_character(draft: dict) -> Character:
    """Convert a completed draft dict into a Character model with derived stats."""
    ab = AbilityScores(
        strength=draft["strength"] or 10,
        dexterity=draft["dexterity"] or 10,
        constitution=draft["constitution"] or 10,
        intelligence=draft["intelligence"] or 10,
        wisdom=draft["wisdom"] or 10,
        charisma=draft["charisma"] or 10,
    )

    cls = draft.get("char_class", "").strip()
    skills = {s.lower() for s in draft.get("skill_proficiencies", [])}
    derived = derive_level1_stats(ab, cls, skills)
    spell_stats = derive_spellcasting_stats(ab, cls, proficiency_bonus=2)
    # finalize_character() already validated the spell selection (right count,
    # all on the class's menu) before this is ever called — an error return
    # here would be a defensive no-op, not a real user-facing path.
    spells_known, spells_prepared, _ = build_spells_known(cls, draft.get("spells_known", []))

    return Character(
        name=draft.get("name") or "Unnamed Adventurer",
        race=draft.get("race", ""),
        char_class=cls,
        subclass=draft.get("subclass") or None,
        background=draft.get("background") or None,
        alignment=draft.get("alignment") or None,
        appearance=draft.get("appearance", ""),
        level=1,
        ability_scores=ab,
        proficiency_bonus=2,
        max_hp=derived["max_hp"],
        current_hp=derived["max_hp"],
        ac=derived["ac"],
        passive_perception=derived["passive_perception"],
        spell_slots=derived["spell_slots"],
        hit_dice_total=derived["hit_dice_total"],
        hit_dice_remaining=1,
        attacks=derived["attacks"],
        inventory=derived["inventory"],
        currency=derived["currency"],
        skill_proficiencies=skills,
        saving_throw_proficiencies=derive_saving_throw_proficiencies(cls),
        spellcasting_ability=spell_stats["spellcasting_ability"],
        spell_save_dc=spell_stats["spell_save_dc"],
        spell_attack_bonus=spell_stats["spell_attack_bonus"],
        spells_known=spells_known,
        spells_prepared=spells_prepared,
        personality_traits=draft.get("personality_traits", []),
        ideals=draft.get("ideals", []),
        bonds=draft.get("bonds", []),
        flaws=draft.get("flaws", []),
        notes=draft.get("backstory", "") + ("\n" + draft.get("notes", "") if draft.get("notes") else ""),
    )


def make_reference_tools() -> list:
    """list_options/get_option_details need no campaign/player/store/draft
    state at all — they only read static reference data (RACES, CLASSES,
    BACKGROUNDS, SPELL_MENUS, ALL_SPELLS). Split out from make_tools() so the
    main in-game agent can bind them too: prompts.py already instructs the
    in-game mechanics model to call list_options('spells <class>') during
    level-up spell selection, but registry.py's get_tools() never included
    chargen.make_tools() (it's Session-0-only, and pulls in draft-mutating
    tools that make no sense in-game) — so that instruction referenced a tool
    the in-game agent structurally could not call. This closes that gap
    without exposing update_character_draft/finalize_character etc. in-game."""

    @tool
    def list_options(category: str) -> str:
        """List available options for a character creation category.

        category must be one of: races, classes, backgrounds,
        ability_score_methods, subraces (include the race name, e.g. 'subraces elf'),
        spells (include the class name, e.g. 'spells wizard' — non-caster classes
        report they have no spellcasting)
        """
        cat = category.lower().strip()

        if cat == "races":
            lines = ["Available races (PHB):"]
            for name, data in RACES.items():
                lines.append(f"  {name}: {data['asi']}, speed {data['speed']}")
            return "\n".join(lines)

        if cat == "classes":
            lines = ["Available classes (PHB):"]
            for name, data in CLASSES.items():
                lines.append(f"  {name} (d{data['hit_die']}): {data['primary_ability']}-based. {data['playstyle'][:80]}…")
            return "\n".join(lines)

        if cat == "backgrounds":
            lines = ["Available backgrounds (PHB):"]
            for name, data in BACKGROUNDS.items():
                lines.append(
                    f"  {name}: {'/'.join(data['ability_scores'])} scores, "
                    f"{data['feat']} feat, {', '.join(data['skills'])}. {data['flavor'][:60]}…"
                )
            return "\n".join(lines)

        if cat == "ability_score_methods":
            lines = ["Ability score generation methods:"]
            for method, desc in ABILITY_SCORE_METHODS.items():
                lines.append(f"\n  {method.upper()}\n  {desc}")
            return "\n".join(lines)

        if cat.startswith("subrace"):
            race_name = cat.replace("subraces", "").replace("subrace", "").strip().title()
            race = RACES.get(race_name)
            if not race:
                return f"No race named '{race_name}' found."
            if "subraces" not in race:
                return f"{race_name} has no subraces."
            lines = [f"Subraces for {race_name}:"]
            for sub, desc in race["subraces"].items():
                lines.append(f"  {sub}: {desc}")
            return "\n".join(lines)

        if cat.startswith("spell"):
            class_name = cat.replace("spells", "").replace("spell", "").strip().title()
            if not class_name:
                return "Usage: list_options('spells <class>'), e.g. 'spells Wizard'."
            menu = SPELL_MENUS.get(class_name)
            if menu is None:
                return f"{class_name} has no spellcasting."
            req = SPELL_REQUIREMENTS.get(class_name, {})
            lines = [f"Spell options for {class_name} (call update_character_draft "
                     f"with field 'spells_known' and a single comma-separated list "
                     f"covering both tiers below):"]
            if 0 in menu:
                lines.append(f"\n  Cantrips — choose {req.get(0, 0)}:")
                for name in menu[0]:
                    lines.append(f"    {name}: {ALL_SPELLS[name].description}")
            if 1 in menu:
                lines.append(f"\n  Level-1 spells — choose {req.get(1, 0)}:")
                for name in menu[1]:
                    lines.append(f"    {name}: {ALL_SPELLS[name].description}")
            return "\n".join(lines)

        return "Unknown category. Use: races, classes, backgrounds, ability_score_methods, 'subraces <race>', or 'spells <class>'."

    @tool
    def get_option_details(category: str, name: str) -> str:
        """Get detailed information about a specific race, class, or background.

        category: 'race', 'class', 'background', or 'spell'
        name: the exact name (e.g. 'Elf', 'Wizard', 'Sage')
        """
        cat = category.lower().strip()
        raw_name = name.strip()
        name = raw_name.title()

        if cat == "spell":
            spell = next((s for s in ALL_SPELLS.values() if s.name.lower() == raw_name.lower()), None)
            if not spell:
                return f"No spell named '{raw_name}' in the curated menus. Use list_options('spells <class>') to see options."
            lines = [
                f"=== {spell.name} ===",
                f"Level {spell.level} {spell.school}, {spell.casting_time}, Range: {spell.range}",
            ]
            if spell.resolution_type.value == "attack_roll":
                lines.append(f"Attack roll vs AC — damage {spell.effect_dice} {spell.damage_type.value if spell.damage_type else ''}".rstrip())
            elif spell.resolution_type.value == "saving_throw":
                save_line = f"{spell.save_ability.title()} saving throw"
                if spell.effect_dice:
                    save_line += f" — damage {spell.effect_dice} {spell.damage_type.value if spell.damage_type else ''}".rstrip()
                    save_line += " (half on success)" if spell.half_damage_on_success else " (no effect on success)"
                if spell.condition_on_fail:
                    save_line += f" — {spell.condition_on_fail} on failure"
                lines.append(save_line)
            else:
                lines.append("No attack roll or saving throw — automatic effect.")
            lines.append(spell.description)
            return "\n".join(lines)

        if cat == "race":
            data = RACES.get(name)
            if not data:
                return f"No race named '{name}'. Use list_options('races') to see available options."
            lines = [
                f"=== {name} ===",
                f"ASI: {data['asi']}",
                f"Speed: {data['speed']} ft",
                f"Flavor: {data['flavor']}",
                "Traits:",
            ]
            lines += [f"  - {t}" for t in data["traits"]]
            if "subraces" in data:
                lines.append("Subraces:")
                for sub, desc in data["subraces"].items():
                    lines.append(f"  {sub}: {desc}")
            if "good_for" in data:
                lines.append(f"Best for: {data['good_for']}")
            return "\n".join(lines)

        if cat == "class":
            data = CLASSES.get(name)
            if not data:
                return f"No class named '{name}'. Use list_options('classes') to see available options."
            lines = [
                f"=== {name} ===",
                f"Hit die: d{data['hit_die']}",
                f"Primary ability: {data['primary_ability']}",
                f"Saving throws: {', '.join(data['saving_throws'])}",
                f"Armor: {data['armor']}",
                f"Weapons: {data['weapons']}",
                f"Skills: Choose {data['skills']['count']} from: {', '.join(data['skills']['from']) if isinstance(data['skills']['from'], list) else data['skills']['from']}",
                "Level 1 features:",
            ]
            lines += [f"  - {f}" for f in data.get("level_1_features", [])]
            if data.get("level_2_features"):
                lines.append("Level 2 features:")
                lines += [f"  - {f}" for f in data["level_2_features"]]
            if data.get("level_3_features"):
                lines.append("Level 3 features:")
                lines += [f"  - {f}" for f in data["level_3_features"]]
            lines.append(f"Playstyle: {data['playstyle']}")
            if "good_for" in data:
                lines.append(f"Best for: {data['good_for']}")
            return "\n".join(lines)

        if cat == "background":
            data = BACKGROUNDS.get(name)
            if not data:
                return f"No background named '{name}'. Use list_options('backgrounds') to see available options."
            lines = [
                f"=== {name} ===",
                f"Ability scores (choose +2/+1 or +1/+1/+1 among these, never above 20): "
                f"{', '.join(data['ability_scores'])}",
                f"Origin feat: {data['feat']}",
                f"Skills: {', '.join(data['skills'])}",
            ]
            if data["tools"]:
                lines.append(f"Tools: {', '.join(data['tools']) if isinstance(data['tools'], list) else data['tools']}")
            if data["languages"]:
                lines.append(f"Languages: {data['languages']}")
            lines += [
                f"Feature: {data['feature']}",
                f"Flavor: {data['flavor']}",
            ]
            return "\n".join(lines)

        return "Unknown category. Use 'race', 'class', or 'background'."

    return [list_options, get_option_details]


def make_tools(
    campaign_id: str,
    player_slug: str,
    store: CampaignStore,
    ds: DraftStore,
) -> list:

    @tool
    def roll_ability_scores() -> str:
        """Roll ability scores using the 4d6-drop-lowest method, six times.
        Returns an array of six values the player can assign to their scores.
        Only call this if the player chose the rolled method."""
        results = []
        for _ in range(6):
            total, rolls = _roll_4d6_drop_lowest()
            results.append((total, sorted(rolls)))

        lines = ["Ability score rolls (4d6, drop lowest):"]
        for i, (total, rolls) in enumerate(results, 1):
            dropped = rolls[0]
            kept = rolls[1:]
            lines.append(f"  Roll {i}: {rolls} → drop {dropped} → {total}")
        scores = sorted([r[0] for r in results], reverse=True)
        lines.append(f"\nFinal scores to assign: {scores}")
        lines.append("Assign these to STR, DEX, CON, INT, WIS, CHA in any order.")
        return "\n".join(lines)

    @tool
    def update_character_draft(field: str, value: str) -> str:
        """Update a field in the character currently being created.

        Supported fields:
          name, race, subrace, char_class, subclass, background, alignment, backstory, notes
          appearance  (freeform physical description — what the player pictures: build, face,
            clothing, notable features. Ask for this explicitly; it doesn't fall out of
            mechanical choices the way race/class does.)
          strength, dexterity, constitution, intelligence, wisdom, charisma  (integers)
          skill_proficiencies  (comma-separated, e.g. "Perception, Stealth, Arcana")
          spells_known  (comma-separated, ONLY for a spellcasting class — check
            list_options('spells <class>') first for the exact menu and how many
            cantrips/level-1 spells are required. Combine both tiers in ONE call,
            e.g. "Fire Bolt, Ray of Frost, Mage Hand, Magic Missile, Shield" —
            don't call this once per spell. Non-caster classes need nothing here.)
          personality_traits, ideals, bonds, flaws  (comma-separated; prefix CLEAR to reset)

        Call this after each confirmed choice so the character sheet preview stays current.
        """
        return ds.update(campaign_id, player_slug, field, value)

    @tool
    def update_ability_scores(
        strength: int | None = None,
        dexterity: int | None = None,
        constitution: int | None = None,
        intelligence: int | None = None,
        wisdom: int | None = None,
        charisma: int | None = None,
    ) -> str:
        """Set some or all six ability scores in ONE call once the player has
        confirmed their final numbers (after rolls/array/point-buy and their
        Background's ability score increase are applied — species grant no
        ability score bonus in this ruleset). Prefer this over six separate
        update_character_draft calls — narrating a score without a matching
        tool call leaves the draft (and the player-visible preview) wrong."""
        scores = {
            "strength": strength, "dexterity": dexterity, "constitution": constitution,
            "intelligence": intelligence, "wisdom": wisdom, "charisma": charisma,
        }
        set_fields = {k: v for k, v in scores.items() if v is not None}
        if not set_fields:
            return "No scores provided — pass at least one of strength/dexterity/constitution/intelligence/wisdom/charisma."
        for field, value in set_fields.items():
            ds.update(campaign_id, player_slug, field, str(value))
        summary = ", ".join(f"{k}={v}" for k, v in set_fields.items())
        return f"Updated ability scores: {summary}."

    @tool
    def get_draft_summary() -> str:
        """Return the current state of the character draft. Call to review before
        finalizing, or to catch anything still missing."""
        draft = ds.get(campaign_id, player_slug)
        ab_scores = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]

        lines = ["=== Current Character Draft ==="]
        for field in ["name", "race", "subrace", "char_class", "subclass", "background", "alignment"]:
            val = draft.get(field, "")
            lines.append(f"  {field:20s}: {val or '(not set)'}")

        appearance = draft.get("appearance", "")
        lines.append(f"  {'appearance':20s}: {appearance or '(not set)'}")

        lines.append("\n  Ability Scores:")
        for stat in ab_scores:
            val = draft.get(stat, 0)
            lines.append(f"    {stat:14s}: {val if val else '(not set)'}")

        for field in ["skill_proficiencies", "spells_known", "personality_traits", "ideals", "bonds", "flaws"]:
            val = draft.get(field, [])
            lines.append(f"  {field:20s}: {', '.join(val) if val else '(not set)'}")

        backstory = draft.get("backstory", "")
        lines.append(f"  {'backstory':20s}: {'(set)' if backstory else '(not set)'}")

        missing = []
        required = ["name", "race", "char_class", "background"]
        required += [s for s in ab_scores if not draft.get(s)]
        for f in required:
            if not draft.get(f):
                missing.append(f)
        if missing:
            lines.append(f"\n  Still needed: {', '.join(missing)}")
        else:
            lines.append("\n  All required fields set. Ready to finalize!")

        return "\n".join(lines)

    @tool
    async def finalize_character() -> str:
        """Commit the character draft as a permanent party member and save to the campaign.

        Call this ONLY when:
        - All required fields are set (name, race, class, background, all six ability scores)
        - The player has confirmed every choice
        - Backstory and personality have been established

        After calling this, the character appears in the campaign party and the draft is cleared.
        """
        draft = ds.get(campaign_id, player_slug)

        required = ["name", "race", "char_class", "background",
                    "strength", "dexterity", "constitution",
                    "intelligence", "wisdom", "charisma"]
        missing = [f for f in required if not draft.get(f)]
        if missing:
            missing_scores = [f for f in missing if f in
                               {"strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"}]
            missing_other = [f for f in missing if f not in missing_scores]
            fix = []
            if missing_scores:
                fix.append(f"call update_ability_scores now with real values for: {', '.join(missing_scores)}")
            if missing_other:
                fix.append(f"call update_character_draft now for: {', '.join(missing_other)}")
            return (
                f"Cannot finalize — the draft is missing: {', '.join(missing)}. "
                f"Do NOT call finalize_character again until you have actually done this: {'; '.join(fix)}. "
                "These must be real tool calls in your next turn, not just described in your reply. "
                "After making those calls, call get_draft_summary to confirm nothing is still '(not set)' "
                "before calling finalize_character again."
            )

        cls = draft.get("char_class", "").strip()
        if cls in SPELL_REQUIREMENTS:
            _, _, spell_err = build_spells_known(cls, draft.get("spells_known", []))
            if spell_err:
                return (
                    f"Cannot finalize — {spell_err} Call list_options('spells {cls}') to see the "
                    f"real menu, then update_character_draft('spells_known', ...) with the corrected "
                    f"list (combine cantrips and level-1 spells in one call). Do NOT call "
                    f"finalize_character again until you have actually done this — these must be real "
                    f"tool calls in your next turn, not just described in your reply."
                )

        character = _build_character(draft)
        campaign = await store.load(campaign_id)
        if not campaign:
            return "Error: campaign not found."

        existing = next((c for c in campaign.party if c.name.lower() == character.name.lower()), None)
        if existing:
            return f"A character named '{character.name}' already exists in the party. Choose a different name."

        campaign.party.append(character)
        await store.save(campaign)
        ds.clear(campaign_id, player_slug)

        return (
            f"✓ {character.name} ({character.race} {character.char_class}) has been added to the party! "
            f"HP: {character.max_hp}, AC: {character.ac}. "
            f"Their character sheet is now saved. Session 0 is complete for this player."
        )

    return [*make_reference_tools(), roll_ability_scores,
            update_character_draft, update_ability_scores, get_draft_summary, finalize_character]
