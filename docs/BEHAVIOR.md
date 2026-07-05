# Agent behavior scenarios

Given/When/Then scenarios specifying how this project's DM agent is meant to
behave. There's no automated test suite behind these (see
[`VERIFICATION.md`](VERIFICATION.md) for why, and how they're checked in
practice instead) — this document is the spec these scenarios are checked
against, not a passing/failing test report.

Each scenario below corresponds to a real behavior this project's agent was
found *not* to reliably exhibit at some point, and was then fixed at either the
prompt level, the guardrail level, or the tool level (often more than one, when
a prompt-only fix wasn't reliably followed by a smaller local model). File
references point to where the behavior is actually implemented.

## Combat formalization

**Scenario: a single decisive blow against a helpless target**
- Given a target that is helpless, unaware, or otherwise unable to act back
- And the blow would end the matter
- When the mechanics model resolves the attack
- Then a bare `resolve_attack` call is acceptable with no active encounter
- And no `start_encounter` call is required

**Scenario: a fight that could continue**
- Given more than one hostile actor, or a target that could plausibly act back
- When an attack is resolved
- Then `create_monster` must register every hostile combatant not already known
- And `start_encounter` must be called before any further resolution
- And the resolution report must state the full initiative order, not just
  confirm an encounter started

*(`backend/agent/prompts.py`'s Combat section bright-line rule;
`backend/agent/dm_agent.py`'s `_detect_missing_encounter_followup` backstops
this when a resolve_attack/resolve_saving_throw call lands on a surviving
Monster with no active encounter.)*

## Turn order integrity

**Scenario: acting out of turn**
- Given an active encounter with a real initiative order
- When a tool call names an attacker/caster who isn't the current-turn combatant
- Then the tool call refuses
- And the refusal names who the current turn actually belongs to

**Scenario: acting in turn**
- Given an active encounter
- When the named attacker/caster matches the current-turn combatant
  (case-insensitive)
- Then the tool call proceeds normally

*(`require_current_turn` in `backend/tools/_helpers.py`, enforced inside
`resolve_attack` and `cast_spell` in `backend/tools/resolution.py` — a hard
refusal, not a soft nudge, since a single soft correction proved insufficient
to stop a local model from repeatedly resolving the same character's turn.)*

## Auto-continuation through non-player turns

**Scenario: the player's turn is resolved, monsters/companions are next**
- Given the player's turn is fully resolved
- And the next combatant(s) in initiative are monsters or DM-controlled
  companions
- When the mechanics model continues
- Then it keeps resolving turns — calling `advance_initiative` between each —
  without waiting for further player input
- Until the initiative order reaches a player-controlled character's turn
- And only then does it stop and hand off to the narrator

**Scenario: narrating a multi-turn reply**
- Given a single reply resolves more than one combatant's turn
- When the narrator writes the response
- Then each combatant's turn is its own distinct narrative beat, in the order
  they acted
- And the reply ends by naming the specific character whose turn it now is,
  not a generic "what do you do?"

*(`backend/agent/prompts.py`'s Combat section and Narration style section.)*

## Loot: reveal before distribution

**Scenario: a find with more than one plausible recipient**
- Given loot found from a defeated enemy or a shared search (a body, a
  container, an area with more than one party member present)
- When it's found
- Then `reveal_loot` records concrete, decided contents and shows them
  unassigned (a 💰-marked block)
- And no character's inventory changes yet

**Scenario: distributing revealed loot**
- Given loot was revealed but not yet assigned
- When the player says who takes what
- Then `add_item_to_character`/`update_character_currency`/
  `remove_item_from_character` are called for the stated allocation
- And the exact character name the player names is the one credited

**Scenario: a solo find**
- Given only one character could plausibly take something found
- When it's found
- Then `add_item_to_character`/`update_character_currency` are called directly,
  same turn, no reveal/distribute split needed

*(`reveal_loot` in `backend/tools/party.py`; the loot guardrail
`_detect_missing_loot_followup` and its `_LOOT_MENTION_RE` pattern in
`backend/agent/dm_agent.py` catch a narrated find with no backing tool call,
including mundane phrasing like "snatches the weapon" that a narrower pattern
previously missed.)*

## NPC persistence

**Scenario: a background character becomes plot-relevant**
- Given an NPC was only ever background scenery or a nameless placeholder
- When they're recruited, give unique information, or otherwise become
  individually important
- Then `create_npc` (or `generate_companion_character` if they'll fight) is
  called immediately
- And they are never left existing only in narration

**Scenario: staying visible while traveling with the party**
- Given an NPC is now traveling with the party across scenes
- When several turns/locations pass
- Then they remain visible in agent context via `set_npc_traveling_with_party`
  and the `[TRAVELING NPCs]` context block
- Rather than silently dropping out once conversation history trims

**Scenario: a scheduled future reappearance**
- Given a departing NPC's narration establishes a specific future reappearance
- When they leave the party
- Then `place_npc_at_location` seeds them at a known destination (if it
  already exists), or a `[SCHEDULED]` note is recorded on the NPC
- So the reappearance survives past the current session, not just this
  conversation's memory

*(`backend/tools/npc.py`, `backend/agent/prompts.py`'s "NPC persistence:"
section.)*

## Grounded weapon attacks

**Scenario: an unequipped mundane weapon**
- Given a character has a mundane weapon in `inventory` but not in `attacks`
- When they try to attack with it
- Then `resolve_attack` does not fabricate to-hit/damage for it — it either
  refuses (no matching `attack_name`) or falls back to an existing single
  attack
- And it never uses the `spell_name` override path to fake a mundane weapon's
  damage under an invented name

**Scenario: equipping a looted mundane weapon**
- Given a mundane weapon is already in a character's `inventory`
- When the character intends to actually fight with it
- Then `add_weapon_attack` grounds a real `Attack` from the weapon's real
  stats and the character's real proficiency bonus/ability modifier
- And `resolve_attack` can then target it by name with a grounded roll

*(`add_weapon_attack` in `backend/tools/party.py`; the tightened
`resolve_attack`/`spell_name` guidance in `backend/agent/prompts.py`.)*
