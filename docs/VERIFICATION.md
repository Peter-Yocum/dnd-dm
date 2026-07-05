# Verification workflow

This project has no automated test suite. That's a deliberate scope choice, not
an oversight — the correctness that actually matters here is behavioral
(*does the agent reliably call the right tool at the right moment, given a
smaller local model that doesn't always follow instructions perfectly?*), and
that's not something a conventional unit test expresses well. A unit test can
assert `apply_damage_to_monster` caps healing at `max_hp` — it can't assert
that the mechanics model reliably calls `start_encounter` the instant an
ambush could turn into a multi-round fight.

## Why manual, scenario-driven verification

Every real bug fixed in this project's development (see
[`docs/engineering-notes/`](engineering-notes/) and the scenarios in
[`BEHAVIOR.md`](BEHAVIOR.md)) was found by actually playing the game through
the running app and reading the transcript — not by code review alone. Examples
from real playtests:

- Combat never formalized into a real encounter (no `start_encounter` call),
  despite attack rolls clearly happening — only visible by watching a real
  ambush scene play out and noticing no `[LIVE ENCOUNTER STATE]` block ever
  appeared.
- The same player character kept acting turn after turn, with the model
  narrating a plausible-looking "Round 1 / Initiative Order" that never
  actually advanced — only caught by playing several turns in a row and
  noticing the round number never changed.
- A looted item was narrated as handed to one companion but silently ended up
  in a different character's inventory — only caught by checking the actual
  character sheet (`get_character`), not by reading the narration, since the
  narration and the real state had quietly diverged.

That last point is the core method: **check real state, not narrated prose.**
The whole reason several of this project's guardrails exist is that an LLM can
narrate an outcome fluently without a backing tool call ever having made it
true. Verification has to look past the narration.

## The workflow

1. Start the app (`make up`) and begin or continue a real campaign session.
2. Play a scripted scenario through the actual chat interface — not a unit
   test harness — driving the specific behavior under check (an ambush, a
   shared loot find, recruiting a background NPC, etc.).
3. Read the transcript as a player would, but also inspect the *actual*
   persisted state behind it:
   - `get_character` / `get_party_status` for HP, inventory, and attacks.
   - The `[LIVE ENCOUNTER STATE]` block (or its absence) for combat/initiative.
   - The campaign record directly (via the store, or `GET /campaigns/{id}/party/{char_id}`)
     when a transcript claim needs independent confirmation.
4. Compare what actually happened against the expected behavior — the
   Given/When/Then scenarios in [`BEHAVIOR.md`](BEHAVIOR.md) are the
   checklist this project uses for "what should happen here."
5. When something doesn't match, the fix is rarely "reword the prompt harder."
   The pattern that's worked here: add a prompt rule for the common case, and
   back it with either a soft one-shot corrective nudge (for things a model
   can reasonably self-correct once told) or a hard tool-level refusal (for
   invariants too important to leave to a nudge — see `require_current_turn`
   in `backend/tools/_helpers.py`).

## A repeatable checklist

Generalized from real fixes made this way — worth replaying after any change
to the agent prompts, tools, or graph structure:

- **Combat formalization** — trigger an ambush/fight that could plausibly
  continue past one blow. Confirm `create_monster`/`start_encounter` actually
  get called and a `[LIVE ENCOUNTER STATE]` block appears, rather than attack
  rolls happening freeform.
- **Turn order integrity** — during an active encounter, try to have a
  character act when it isn't their turn. Confirm the tool refuses and names
  the real current-turn combatant.
- **Auto-continuation** — resolve the player's turn when the next several
  combatants in initiative are monsters/companions. Confirm the reply resolves
  all of them in sequence (each as its own narrated beat) before stopping to
  ask the player anything, and that the round counter actually increments.
- **Loot reveal vs. distribution** — have the party find loot with more than
  one plausible recipient. Confirm it's revealed unassigned (a 💰 block, no
  inventory change) and only lands in a specific character's inventory once
  the player says who takes what.
- **NPC persistence** — recruit a previously-background character into the
  story. Confirm `create_npc`/`generate_companion_character` actually gets
  called (check `get_npc`), and that they're still known about — not silently
  forgotten — several scenes later.
- **Grounded weapon attacks** — give a character a mundane weapon that isn't
  yet a real `Attack`. Confirm `resolve_attack` doesn't fabricate stats for it,
  and that `add_weapon_attack` is the correct path to make it usable.

A project-specific `verify` skill (matching this environment's own `/verify`
convention for exercising real app behavior after a change) is a natural next
step here, but hasn't been built yet — this document is the manual equivalent
in the meantime.
