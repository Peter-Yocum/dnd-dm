# Working-Tree Changes Summary

Code review and simplification pass on top of `cb199a9`. Bundles a refactor (centralized
LLM client construction, lock extraction), a multi-combatant resolution redesign, several
live-bug fixes, and minor infra cleanup.

## 1. LLM client construction consolidation

**New: `backend/llm.py`**
Single factory for every Ollama client in the app (`ollama_chat`, `ollama_embeddings`),
replacing duplicated `ChatOllama(...)`/`OllamaEmbeddings(...)` construction across the
codebase.

- Always sets `reasoning=False` on chat models — otherwise `gemma4:26b-mlx` leaks
  `<|channel|>thought` tags into `.content`.
- Applies `keep_alive=settings.ollama_keep_alive` and a default timeout
  (`CHAT_TIMEOUT_S=120s`, `EMBED_TIMEOUT_S=60s`) to prevent an unbounded httpx client from
  leaking a thread-pool slot forever on a hung Ollama request (an unkillable blocking
  socket read).
- Exists because a prior manual fix for that leak (2026-07-08) missed a call site
  (`rag/contextualizer.py`, which previously had *no* client-level timeout at all).
- Offline ingest scripts (`scripts/build_index.py`, `merge_chroma.py`, `extract_entities.py`,
  `add_headers.py`, `clean_source.py`) pass `timeout=None` intentionally — long
  model-swap thrash during overnight runs can legitimately take minutes.

Callers migrated to the factory (pure consolidation, no behavior change beyond the
factory's own fixes): `backend/agent/dm_agent.py`, `backend/rag/contextualizer.py`,
`backend/rag/grading.py`, `backend/rag/reranker.py`, `backend/stores/history_store.py`,
`backend/stores/rules_store.py`.

**`backend/config.py`**
- New `embed_model: str = "nomic-embed-text"` setting (was hardcoded at five call sites).
- New `ollama_keep_alive: str | None = None` setting. Comment documents a 2026-07-10
  experiment: `keep_alive=-1` fixed an embed↔chat model-swap freeze but caused unbounded
  KV-cache growth (18GB → 26GB resident over hours), which segfaulted a concurrent ingest
  run — reverted to `None` (server's own idle-timeout eviction).

## 2. Lock extraction

**New: `backend/locks.py`**
Extracts the per-campaign `asyncio.Lock` registry (`campaign_write_lock(campaign_id)`)
out of `dm_agent.py`'s private `_get_tool_lock`, since `main.py` had started reaching into
that private symbol. The invariant it protects (serializing load→mutate→save cycles
against the campaign store, which persists via delete-all/reinsert, so concurrent cycles
can silently drop mutations) belongs to campaign persistence, not the agent. Not
reentrant; plain dict with no eviction (fine for a single-user app).

`backend/main.py`'s `end_session` load→mutate→save cycle is now wrapped in this lock
(previously unlocked) — fixes a **confirmed live bug (2026-07-10)**: a slow
`summarize_session` call (30s–2min) let a concurrent tool-call save land and get silently
clobbered by `end_session`'s stale in-memory write, causing a just-looted item to vanish.

## 3. Multi-combatant turn resolution redesign

`backend/agent/dm_agent.py` — the largest change, several intertwined pieces:

- **Per-combatant turn boundaries**: `_messages_since_last_human` renamed to
  `_messages_since_last_turn_boundary`, now also stops at a tool-call-free `AIMessage`
  (a completed narrator reply), because `narrator_node` can loop back to `mechanics`
  multiple times per player message — once per combatant.
- **`stream_response` restructured** from one `astream_events` call covering the whole
  auto-resolved sequence to a loop of separate calls per combatant (new `TURN_BOUNDARY`
  sentinel between them, capped at `_MAX_COMBATANT_TURNS=10`). Fixes two problems: (a) a
  fixed `recursion_limit` per call was being exhausted by one hard combatant, starving
  later easy ones into a hard `GraphRecursionError`; (b) guardrail budgets need to reset
  per-combatant, not once per player message. `_FRESH_GUARDRAIL_BUDGETS` centralizes the
  four per-combatant counters that must reset each turn.
- **`prompts.py`** combat instructions rewritten to match: the model now resolves only the
  *current* combatant's turn and stops; the harness brings it back for the next one. Also
  fixes a bug where the first turn after `start_encounter` (when a non-player combatant
  wins initiative) wasn't being resolved.
- **`main.py` `/stream`** forwards the new `TURN_BOUNDARY` sentinel as an SSE
  `turn_boundary` event; **`templates/game.html`** adds a handler that closes the current
  chat bubble and opens a fresh one (re-showing "thinking") on that event, instead of
  silently concatenating multiple combatants' narration into one bubble. Bubble lifecycle
  also centralized into `openBubble`/`clearThinking`/`settleBubble` helpers shared by the
  `token`, `turn_boundary`, `done`, and `error` handlers.
- **Detection helpers converted async → sync**: `_detect_missing_followup`,
  `_detect_missing_encounter_followup`, `_detect_stalled_non_player_turn_followup`,
  `_next_turn_ground_truth_note` now take a shared `campaign` snapshot loaded once per
  mechanics pass, instead of each independently reloading Postgres (was 3–5 round trips
  per combatant, multiplied by the new per-combatant loop).
- **New `_detect_missing_end_encounter_followup`**: catches the model narrating "combat has
  ended" without calling `end_encounter`, which otherwise leaves `active_encounter.is_active`
  stuck `True` forever and skips the automatic post-combat loot roll (live-observed,
  2026-07-11).
- **New orphaned-turn recovery** (`_orphaned_interrupted_turn`/`_recover_orphaned_turn`):
  detects a turn interrupted mid-execution by a process crash/restart (an `AIMessage` with
  unresolved tool_calls and no following `ToolMessage`s/narrator reply), which would
  otherwise hand Ollama a malformed message sequence and error outright.
- **New verified-roll relay** (`_verified_rolls_note`, `VERIFIED_ROLLS_MARKER`,
  `_VERIFIED_ROLL_TOOLS`, `_CHARGEN_VERIFIED_ROLL_TOOLS`): captures verbatim tool-output
  dice breakdowns before scratch-message purge and appends them to the resolution report
  so the narrator can't silently paraphrase or invent a damage total. `prompts.py` and
  `session_zero_prompt.py` updated to require copying these numbers character-for-character
  rather than "smoothing out" an unlucky result.
- **`_COMBAT_ROLL_MENTION_RE`** extended with a regex arm for bare-prose strike verbs
  (stab/slash/pierce, plus ambiguous strike/hit/swing gated on a following preposition) to
  catch narration of a physical hit with no backing roll (accepted false-positive rate,
  documented). `_detect_missing_combat_roll_followup` now exempts text starting with the
  new `OOC_MARKER` ("[OOC]"), since rules-quote answers legitimately use combat vocabulary
  with no roll behind them.

## 4. Session/thread continuity fix

- **`backend/models.py`**: new `Campaign.active_thread_id: str = ""` — durable
  server-side pointer to the active chat thread.
- **`backend/main.py`**: `GET /campaigns/{campaign_id}` now reuses
  `campaign.active_thread_id` instead of unconditionally minting a new thread on every
  page load — fixes a bug where reopening a campaign mid-session (closed tab, browser
  restart) silently forked into a brand-new empty thread, orphaning the in-progress
  session. **Confirmed live**: one evening's play got split across three threads this way.
  New `_mint_active_thread(campaign, store)` helper atomically mints/persists a fresh
  thread id. The "already summarized" retry-echo path now returns the *current*
  `active_thread_id` instead of minting a new one, avoiding orphaning on a stale duplicate
  close from a second tab.
- **`templates/game.html`**: comment updated — sessionStorage is now just a same-tab
  convenience, not the source of truth for thread id.

## 5. Combat/damage correctness fixes

- **`backend/tools/_helpers.py`**: new `with_ability_mod(dice, modifier)` helper. Fixes a
  real accuracy bug — `Attack.damage_dice` never had the character's ability modifier
  baked in at any of its three construction sites (chargen starting weapon,
  `unarmed_strike_attack`, `add_weapon_attack`/`create_magic_item` in `party.py`), even
  though `to_hit_bonus` did include it. Cited example: a DEX 18 shortbow user rolling 6
  piercing damage instead of 7–10. `unarmed_strike_attack` now also includes STR mod
  (RAW: "1 + STR mod"), previously a flat 1d4.
- **`backend/tools/party.py`**: both weapon-attack construction sites switched to
  `with_ability_mod` instead of ad hoc `f"{dice}+{bonus}"` string building.
- **`backend/tools/resolution.py`**: new `_roll_effect(notation)` wrapper around
  `roll_notation` that floors any HP-affecting roll total at 0. Load-bearing now that
  `damage_dice` can carry a negative modifier (via `with_ability_mod`) — an unfloored
  negative total would sign-flip through `apply_damage_to_character/monster`'s
  signed-amount convention, turning a hit into a heal (or a failed heal into damage). All
  damage/healing roll sites switched from raw `roll_notation` to `_roll_effect`.
- **`backend/tools/campaign.py`**: `add_session_note` rewritten — no longer appends
  directly into `campaign.sessions` (could fabricate a phantom "Session 1" with no
  thread_id/summary if called before any session closed, or silently write into an
  already-closed/summarized session's `key_events`). Now purely a transcript-visible
  marker; the real chronicle is regenerated from the full transcript by
  `summarize_session` at session end.
- **`backend/tools/combat.py`**: new `_encounter_monster_names(enc)` helper, deduplicating
  logic independently re-derived at three call sites. `end_encounter` now warns (but
  doesn't refuse) when a still-alive/conscious monster remains on the roster — since
  fled/surrendered isn't modeled as a condition, the tool can't distinguish a legitimate
  close from a mistake, so it surfaces a warning rather than silently dropping a live
  threat from tracking.

## 6. Infra / scripts cleanup

- **`docker-compose.yml`**: blanket `.:/app` bind mount replaced with scoped mounts
  (`backend/`, `static/`, `templates/`, `alembic/`, `alembic.ini`, `data/`,
  `docs/source/adventures/`), excluding `docs/raw`, `docs/source/core`, and `scripts/`
  (only read by offline ingest scripts, which now run natively, not in-container).
  `--reload` narrowed to `--reload-dir backend`.
- **`scripts/add_headers.py`, `clean_source.py`**: added `sys.path.insert(0, ...)` shim
  so `backend` is importable when run as a bare script (needed to import `backend.llm`).
- **`scripts/overnight_queue_native.sh`**: removed three books already completed in a
  prior overnight run (Xanathar's Guide to Everything, Mordenkainen's Tome of Foes,
  Volo's Guide to Monsters), leaving Tasha's Cauldron of Everything and Sword Coast
  Adventurer's Guide.

## Not part of this review (untracked, ignored)

- `scripts/spikes/` — build artifacts only (stale `.pyc`, a Chroma snapshot dump) from a
  vLLM-embedding experiment; no source.
- `overnight-ingest-*.log`, `.dockerignore` — logs/generated, not reviewed changes.
