# D&D Dungeon Master — Design Document

## Goal

A local web app that acts as an AI Dungeon Master for D&D 5e. The AI narrates, rules, and rolls dice via a LangGraph agent backed by local models (vLLM-metal for chat, Ollama for embeddings — see Tech Stack). A player visits a localhost page to interact; no cloud services required. Designed to eventually be hosted for friends via Railway.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM (in-game, Session 0, world-prep — all roles) | vLLM-metal, one model (`mlx-community/Qwen3-30B-A3B-4bit`, an MoE checkpoint — ~3B active params of 30B total) in three roles: mechanics/tool-calling (temp 0.1), narrator/prose (temp 0.8, no tools), and the single-pass world-prep/party-fill/summarization role (temp 0.7) | Was Ollama (`gemma4:26b-mlx` for mechanics/narrator, `qwen2.5:14b` then `gemma4:26b-mlx` for the single-pass role) until the 2026-07-13 vLLM-metal migration (`vllm-migration-plan.md`) — root cause: Ollama has no real `tool_choice`/forced-tool-calling mechanism at all (confirmed against `langchain_ollama`'s own source), which is the entire reason the guardrail chain in "Agent Architecture" below exists. Qwen3-30B-A3B-4bit was chosen over the plan's originally-spiked Gemma4 checkpoint on the strength of outside research suggesting better tool-calling quality — re-verified empirically before adopting it (see the plan doc's "Step 0"): real `tool_choice="required"` compliance (14/15 on a fresh battery of real tool schemas), throughput beating the original Gemma4 spike. The historical Ollama two-model-vs-one-model benchmarking narrative that used to live in this row (`gemma4:26b-mlx` vs `gemma4:12b-mlx` narrator, `qwen3:8b`'s evict-and-reload behavior) is Ollama-specific and no longer describes the current setup — see git history for that record if needed. |
| Embeddings | Ollama (`nomic-embed-text`) | Paired with Postgres/pgvector for local RAG (was Chroma — see Vector store row). Not yet migrated to vLLM-metal — `nomic-embed-text`'s BERT-family architecture can't run on vllm-metal's MLX backend at all (confirmed live); a verified alternative (`mlx-community/Qwen3-Embedding-0.6B-8bit` via vLLM's `--convert embed`) exists and is planned (`vllm-migration-plan.md` §7.7) but not implemented yet. |
| Vector store | PostgreSQL 16 + `pgvector` (same instance as campaign data) | Was ChromaDB (two collections: `rules` + `session_chronicles`) until the 2026-07-12 migration — root cause: Chroma's own local vector index needed ~2.6GB resident just to query the 441k-vector `rules` collection, independent of any keyword-index approach, and this app is meant to be hosted on Railway (this repo's own "one DB everywhere" principle — see Database row below — argued directly against a second, Chroma-shaped storage engine). Now two tables (`rule_chunks`, `session_chronicle_chunks`) with `Vector` columns + native `tsvector`/GIN full-text search, fused via the same `reciprocal_rank_fusion()` as before — see "Datastores" below. |
| Agent framework | LangGraph | Tool-calling loop with checkpointed memory |
| Session memory | PostgreSQL (`langgraph-checkpoint-postgres`) | Same DB as campaign data; persists by `thread_id` |
| Database | PostgreSQL 16 (Docker locally, Railway in prod) | One DB everywhere; no SQLite/Postgres divergence |
| DB access | SQLAlchemy Core + `psycopg` (async) | Async queries, Alembic migrations, no ORM |
| Migrations | Alembic | Standard, Docker-friendly, Railway-compatible |
| Backend | FastAPI + Jinja2 | Python-native, async, good SSE support |
| Frontend | HTMX + Jinja2 templates | No npm, no build step, all Python |
| Streaming | Server-Sent Events (SSE via `sse-starlette`) | Token-by-token DM narration stream to browser |
| Data models | Pydantic v2 | Validation, serialisation, tool input/output typing |
| Config | `pydantic-settings` | Typed env var loading; `.env` per environment |

---

## Directory Layout

```
dnd-dm/
├── design.md
├── Dockerfile
├── docker-compose.yml           # local dev: postgres + app; chroma bind-mounted to ./data/chroma_db
├── Makefile                     # up, down, migrate, shell, fresh, index, setup, index-if-empty
├── .env                         # local overrides — gitignored (can be empty locally)
├── requirements.txt
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
│       └── 0001_initial_schema.py
│
├── backend/
│   ├── config.py                # Settings: DATABASE_URL, OLLAMA_BASE_URL (embeddings), VLLM_BASE_URL (chat)
│   ├── models.py                # all Pydantic v2 domain models
│   ├── main.py                  # FastAPI app, all routes, lifespan
│   ├── agent/
│   │   ├── dm_agent.py          # in-game DM agent: lifespan, trimmer, streaming, summarizer
│   │   ├── prompts.py           # get_system_prompt(campaign) — in-game prompt
│   │   ├── session_zero_agent.py  # (wired into dm_agent.py) Session 0 agent factory
│   │   └── session_zero_prompt.py # get_session_zero_prompt(campaign) — char creation prompt
│   ├── data/
│   │   └── fivee_options.py     # hardcoded PHB races, classes, backgrounds, ability score methods
│   ├── stores/
│   │   ├── tables.py            # SQLAlchemy Core Table definitions (13 tables)
│   │   ├── campaign_store.py    # CampaignStore: CRUD, dice log, parallel entity load
│   │   ├── rules_store.py       # RulesStore: Postgres/pgvector "rule_chunks" table, book-filtered search
│   │   ├── history_store.py     # HistoryStore: Postgres/pgvector "session_chronicle_chunks" table, RAG
│   │   └── draft_store.py       # DraftStore: in-memory character drafts during Session 0
│   └── tools/
│       ├── _helpers.py          # find_char, find_npc, find_monster, char_summary, etc.
│       ├── registry.py          # get_tools(campaign_id, store, rules_store, history_store, books_in_play)
│       ├── dice.py              # roll_dice
│       ├── rules.py             # search_rules (with book filter)
│       ├── memory.py            # search_campaign_history (RAG on session chronicles)
│       ├── party.py             # 8 tools: party status, HP, conditions, spell slots, items
│       ├── npc.py               # 4 tools: get, attitude, knowledge, create
│       ├── combat.py            # 5 tools: encounter start/end, initiative, monster HP, position
│       ├── world.py             # 4 tools: location, move, reveal, container
│       ├── quest.py             # 3 tools: active quests, objectives, status
│       ├── campaign.py          # 2 tools: summary, session note
│       └── chargen.py           # 6 tools: list/detail options, roll scores, draft update, finalize
│
├── templates/
│   ├── base.html                # HTMX CDN, CSS link, header
│   ├── index.html               # campaign selector + create form + adventure picker
│   ├── game.html                # chat + sidebar + SSE + session end overlay
│   ├── sessions.html            # session browser (list + chronicle + transcript)
│   ├── session_zero_index.html  # Session 0 lobby: party roster + start form
│   └── session_zero.html        # char creation: DM chat + live character sheet preview
│
├── static/
│   └── style.css                # dark fantasy theme, streaming cursor, all page layouts
│
├── docs/
│   ├── raw/                     # PDFs to process (moved to raw/done/ after extraction)
│   │   └── done/
│   └── source/
│       ├── core/                # Core rulebooks — always searched, every campaign
│       │   ├── D&D 5E - Dungeon Master's Guide.md
│       │   ├── D&D 5E - Monster Manual.md
│       │   ├── D&D 5E - Player's Handbook.md
│       │   ├── D&D 5E - Mordenkainen's Tome of Foes.md
│       │   ├── D&D 5E - Sword Coast Adventurer's Guide.md
│       │   ├── D&D 5E - Tasha's Cauldron of Everything.md
│       │   ├── D&D 5E - Volo's Guide to Monsters.md
│       │   └── D&D 5E - Xanathar's Guide to Everything.md
│       └── adventures/          # Adventure modules — searched only when in campaign.books_in_play
│           ├── Tyranny of Dragons/
│           │   ├── _meta.json   # {"name": "...", "description": "...", "levels": "1-15", "recommended_players": "4-6"}
│           │   ├── D&D 5E - Tyranny of Dragons - Hoard of the Dragon Queen.md
│           │   └── D&D 5E - Tyranny of Dragons - The Rise of Tiamat.md
│           ├── Tomb of Annihilation/
│           ├── Storm King's Thunder/
│           └── Waterdeep - Dragon Heist/
│
├── data/                        # was also home to chroma_db/ (bind-mounted, gitignored) before the
│                                 # 2026-07-12 Postgres/pgvector migration — rule/session-chronicle
│                                 # vectors now live in Postgres itself, no local data directory for them
│
├── scripts/                      # moved here from repo root 2026-07-05 — see "Prep Scripts" below
│   ├── ocr_ingest.py             # PDF → Markdown (MinerU vlm-engine, MLX-accelerated, macOS only)
│   ├── clean_source.py           # LLM cleanup of garbled extraction artifacts
│   ├── validate_source.py       # heuristic QA: repeated lines, HP math, ability scores
│   └── build_index.py           # docs/source/ → Postgres/pgvector rule_chunks (core + adventures, with metadata)
```

---

## API Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Campaign selector page |
| `POST` | `/campaigns` | Create campaign (with adventure selection) → redirect |
| `GET` | `/campaigns/{id}` | Game page |
| `DELETE` | `/campaigns/{id}` | Delete campaign |
| `POST` | `/campaigns/{id}/message` | Enqueue player message (HTMX form, returns 204) |
| `POST` | `/campaigns/{id}/session/begin` | "Begin/Continue the Adventure" button: builds a session-opening message server-side (first-session intro or recap of the last chronicle) and enqueues it exactly like a player message |
| `GET` | `/campaigns/{id}/stream?thread_id=` | SSE: stream DM tokens, `done` event when finished |
| `GET` | `/campaigns/{id}/thread-info?thread_id=` | JSON message count vs the mechanics trim window, for the context-length warning banner |
| `POST` | `/campaigns/{id}/books` | Add adventure slug to `books_in_play` mid-campaign |
| `POST` | `/campaigns/{id}/session/end` | Summarize session, save chronicle, return new `thread_id` |
| `GET` | `/campaigns/{id}/sessions` | Session list page |
| `GET` | `/campaigns/{id}/sessions/{sid}` | Session detail: chronicle + transcript |
| `GET` | `/campaigns/{id}/session-zero` | Session 0 lobby: party roster + start form + "fill the party" button |
| `POST` | `/campaigns/{id}/session-zero` | Start char creation, redirect to player slug URL |
| `POST` | `/campaigns/{id}/session-zero/fill-party` | DM-triggered one-shot: generate a companion, optionally of an explicit class (`char_class` form field — empty means "DM/model decides"). Synchronous JSON response, not SSE. |
| `GET` | `/campaigns/{id}/session-zero/{slug}` | Char creation: DM chat + live sheet preview |
| `GET` | `/campaigns/{id}/session-zero/{slug}/draft` | JSON of current character draft |
| `POST` | `/campaigns/{id}/session-zero/{slug}/message` | Enqueue char creation message |
| `GET` | `/campaigns/{id}/session-zero/{slug}/stream?thread_id=` | SSE for char creation |
| `DELETE` | `/campaigns/{id}/party/{character_id}` | DM-only: remove a character (PC or companion) from the party, redirect to Session 0 lobby |
| `GET` | `/campaigns/{id}/rolls` | JSON list of recent dice rolls |
| `POST` | `/campaigns/{id}/safety-flag` | Player X-card: append a topic to `safety_flags` + log to `notes` |
| `POST` | `/campaigns/{id}/safety-flag/clear` | DM-only: clear active `safety_flags` (note log stays intact) |
| `POST` | `/campaigns/{id}/rest/long` | Whole-party long rest — deterministic, no LLM call (`apply_long_rest`, `_helpers.py`). Full HP, spell slots, hit dice (half regained, min 1), exhaustion -1, death saves cleared; advances the clock 8h and sets `last_long_rest_day`. |
| `POST` | `/campaigns/{id}/rest/short` | Whole-party short rest — deterministic, no LLM call (`apply_short_rest`, `_helpers.py`). Each character spends just enough hit dice (average value per die) to reach full HP; Warlock spell slots also restored (Pact Magic's short-rest recharge). Advances the clock 1h. |

**Streaming flow:** HTMX form POSTs to `/message`, enqueues text in an in-memory `asyncio.Queue` per campaign (or per chargen session), then JS calls `startStream()`. The SSE endpoint drains the queue, calls the appropriate agent, and yields `token` events until `done`. The browser appends tokens in real time with a blinking cursor.

**Thinking indicator:** the mechanics tool-calling loop (in-game) and the Session 0 ReAct loop's intermediate tool calls both produce no streamed content of their own — only narrator tokens (in-game) or a model's final reply (Session 0) actually stream. Without something to show, the player watches a blank bubble for however long that phase takes (multi-tool-call turns can run minutes). `startThinking()` in both `game.html` and `session_zero.html` fills the empty bubble with a rotating DM-flavored verb ("Pondering…", "Weaving the tale…", "Consulting the dice…") every 1.8s until the first real token arrives, then clears itself.

**Thread ID management:** Generated server-side in the game/chargen `GET` routes, embedded in the page as a JS variable, and persisted in `sessionStorage` so page refreshes within the same tab reuse the same LangGraph thread. The client passes it as a query param to the `/stream` endpoint.

---

## Data Models (`backend/models.py`)

`Campaign` is the root object and owns everything. All models use Pydantic v2 with `uuid4().hex` IDs.

### Entities

| Model | Description |
|---|---|
| `Campaign` | Root object; owns all entities and active state. `books_in_play: list[str]` controls adventure RAG scope. |
| `Character` | PC or DM-controlled companion. Built by `finalize_character` during Session 0. |
| `Monster` | Full combat stat block with legendary/lair actions. |
| `NPC` | Interactable character with personality, knowledge, motivations, optional `CombatStatBlock`. |
| `Faction` | Organisation with reputation score, ranks, territory, inter-faction relationships. |
| `Quest` | State machine (Unknown → Active → Completed/Failed) with objectives and rewards. |
| `Session` | Per-session record: summary, key_events, XP, loot, `thread_id` (for transcript lookup). |
| `Location` | World graph node: connections, lighting, terrain, hidden elements, present NPCs. |
| `Encounter` | Combat state machine: initiative order, combatant positions, difficulty, XP budget. |
| `Trap` | Location-bound hazard with detection/disarm DCs and triggered state. |
| `Container` | Chest/bag with lockable, trappable contents and currency. |
| `Handout` | Document, map, or letter the party has found. |

### Key Campaign fields

| Field | Purpose |
|---|---|
| `books_in_play: list[str]` | Adventure slugs (folder names) included in RAG searches. Core always implicit. |
| `sessions: list[Session]` | All past session records with summaries and `thread_id` references. |
| `session_count: int` | Running count, incremented on each session end. |
| `party_treasury: Container` | Shared loot container — same model as any other chest. |
| `last_long_rest_day: int` | `days_elapsed` at last long rest; enforces once-per-24h rule. |
| `safety_flags: list[str]` | Active X-card topics injected into the system prompt; DM clears once handled. Permanent audit trail lives in `notes`. |

### `Session` model fields

`id`, `session_number`, `real_date`, `summary` (LLM-generated chronicle), `key_events: list[str]` (bullet points), `xp_awarded`, `loot_gained`, `quests_started`, `quests_completed`, `notes`, `thread_id` (LangGraph thread key for transcript retrieval).

---

## Tools (`backend/tools/`) — 45 tools total (+1 for `levelup.py`, added 2026-07-03)

All tools use `async def` with the **closure factory pattern**: `make_tools(campaign_id, store, ...)` returns async tool functions bound to the current session. Two tool sets exist: the **in-game set** (all except chargen) and the **Session 0 set** (dice + rules + chargen only).

### `dice.py` (1 tool)
| Tool | Description |
|---|---|
| `roll_dice(notation)` | Roll dice in standard notation: "2d6+3", "4d6kh3". Never invent a number. |

### `rules.py` (1 tool)
| Tool | Description |
|---|---|
| `search_rules(query)` | Semantic search of indexed rulebooks, filtered to `books_in_play`. Cites book and section. |

### `memory.py` (1 tool)
| Tool | Description |
|---|---|
| `search_campaign_history(query)` | Semantic search of past session chronicles for this campaign. Used when referencing past events, returning NPCs, or following up plot hooks. Not a substitute for `search_rules`. |

### `party.py` (8 tools)
| Tool | Description |
|---|---|
| `get_party_status()` | HP, conditions, spell slots, exhaustion for all party members. |
| `get_character(name)` | Full character sheet: ability scores, proficiencies, spells, attacks, inventory. |
| `update_character_hp(name, delta)` | Freeform damage (negative) or healing (positive) not tied to a specific `Attack` — prefer `resolve_attack` for an actual attack roll. Temp HP absorbed first. |
| `add_condition(name, condition)` | Apply a condition. |
| `remove_condition(name, condition)` | Remove a condition. |
| `use_spell_slot(name, level)` | Expend one spell slot. |
| `restore_spell_slots(name)` | Long rest: restore all slots, reduce exhaustion by 1, reset death saves. |
| `add_item_to_character(name, item, qty)` | Add item to inventory, stacking if already present. |

### `npc.py` (4 tools)
| Tool | Description |
|---|---|
| `get_npc(name)` | Attitude, motivations, knowledge, secrets, merchant status, inventory. |
| `update_npc_attitude(name, attitude)` | Change attitude after a significant interaction. |
| `reveal_npc_knowledge(name, index)` | Mark knowledge as shared; moves it to notes. |
| `create_npc(name, race, occupation, …)` | Add an improvised NPC to the campaign. |

### `combat.py` (6 tools) — plus `build_encounter_context(campaign)`, a plain (non-tool) helper
| Tool | Description |
|---|---|
| `create_monster(name, ac, hp, attacks, count=1, …)` | Add a monster stat block, grounded via `search_rules`. `count>1` creates several identical copies (e.g. 3 goblins) in one call, auto-suffixed `"{name} 1".."{name} N"`. |
| `start_encounter(location, combatants)` | Begin combat: rolls initiative internally (DEX/`initiative_modifier`, or `initiative_override` per-combatant) and builds the order, mark encounter active. Refreshes every combatant's `reaction_available`. |
| `advance_initiative()` | Move to the next turn; increments round counter on wrap; refreshes the new current combatant's `reaction_available`. |
| `end_encounter(xp_awarded)` | Close combat, record XP, clear active encounter. Also rolls and reveals post-combat loot automatically — see "Combat loot generation" below. |
| `update_monster_hp(name, delta)` | Freeform damage/healing to a monster not tied to a specific `Attack` (falling, traps, poison) — prefer `resolve_attack` for an actual attack roll. |
| `set_combatant_position(name, zone, cover)` | Update spatial zone and cover. |

`get_active_encounter` was removed as a callable tool (2026-07-03) — its content (round, initiative, monster stats, positions, any pending reaction) is now auto-injected into the mechanics model's context on every invocation during an active encounter via `build_encounter_context()` + `dm_agent.py`'s `_make_mechanics_modifier`, rather than a tool the model had to remember to call every turn. See "Deferred from the combat resolution refactor" below and the architecture section for why.

### Combat loot generation (2026-07-09)

Before this, all loot was 100% LLM-invented, then recorded via `party.py`'s bookkeeping tools (`reveal_loot`/`add_item_to_character`/`update_character_currency`/`create_magic_item`) — nothing rolled against a real table. `end_encounter` now rolls real loot automatically for every monster the party just defeated, so this no longer depends on the model remembering (or choosing) to invent something reasonable.

**Data (`backend/data/treasure_tables.py`, generated, not hand-written):** the DMG's Individual Treasure tables, Treasure Hoard tables, gem/art object tables, and Magic Item Tables A-I, parsed directly out of the ingested `docs/source/core/D&D 5E - Dungeon Master's Guide.md`'s embedded HTML tables by `scripts/gen_treasure_tables.py` rather than hand-transcribed — at ~250 rows across 9 magic item tables plus four treasure tiers, retyping by hand was both slow and a real transposition risk. Re-run that script if the source file ever changes. Two source-scan defects were caught and fixed during generation (both noted in the generated file's own docstring): the Challenge 11-16 hoard table was missing row 10 entirely (a dropped leading digit, patched to a contiguous range) and "Ioun stone" was OCR'd as "loun stone" throughout Table I. A full contiguity sweep (every d100 range across every table, including the two nested sub-tables — Table G's Figurine of wondrous power, Table I's magic armor — sums to exactly 1-100 with no gaps/overlaps) confirmed clean after both fixes.

**Roll engine (`backend/tools/loot_generator.py`):** `generate_encounter_loot()` rolls Individual Treasure per defeated monster at its own CR tier (DMG's own per-monster convention), then separately gates a single Treasure Hoard roll behind `hoard_drop_chance()` — a homebrew scaling (15%/40%/70%/95% across the four CR tiers), not a DMG rule, chosen specifically so a beefier monster has a real, visible chance at more valuable/magical loot rather than every fight paying out identically. Verified live with a sweep: CR 1/4 hit a hoard ~17% of the time (avg 1.4 items when it did), CR 20 hit ~97% of the time (avg 8.2 items) — matches the intended shape.

Since the DMG's own roll tables don't carry rarity/attunement data (those live in each item's own writeup, not the table), `RARITY_BY_TABLE` assigns one rarity per table letter (A=common..I=legendary — the DMG's own rough correspondence between table letter and hoard tier) and a name-keyword heuristic (`_looks_attuned`) flags likely-attunement items (rings, rods, staves, named weapons, etc.). Both are documented best-effort approximations, not a per-item transcription of all ~250 items' actual individual rarity/attunement.

**Adventure-specific enrichment:** `enrich_with_adventure_loot()` queries the canon Lore Registry (`LoreStore.all_for_book`) for any item whose extracted `owned_by`/`found_at` profile fields (see `scripts/extract_entities.py`'s `ItemExtractor`) match one of the defeated monsters' names or the encounter's location (name or alias) — a guaranteed addition, not chance-gated, since a published adventure ties a specific item to a specific monster/location for a real story reason (a key, a letter, a plot-relevant trinket). This is a best-effort substring match, not a hard link — a hiding spot phrased obliquely in the source text ("tucked beneath the throne") can still miss a location simply named "Great Hall." `prompts.py`'s Loot section carries the explicit backstop: for a named boss or plainly significant fight, if the automatic result seems thin, the DM agent is instructed to still call `search_adventure_literal`/`search_rules` once after `end_encounter` resolves and `reveal_loot` anything that turns up — a genuinely new find at that point, not a duplicate of the automatic roll.

**Double-dip guard:** a narrated treasure pile mid-fight (the model calling `reveal_loot` before combat ends) and the automatic roll are two independent code paths with no natural correlation — without a guard, both could pay out for the same fight. `Encounter.loot_already_granted` (new field, `backend/models.py`) is set by any of `party.py`'s four loot tools whenever they fire while an encounter is active (only on a *gain* for `update_character_currency`, not a spend, so paying off a hostile creature mid-fight doesn't suppress the automatic roll); `end_encounter` checks it first and skips its own roll entirely if it's already `True`. `prompts.py`'s Loot section carries the matching instruction: post-combat loot is `end_encounter`'s job now, narrate its result, don't invent or grant anything for a defeated monster before it resolves.

**Narration hook (`[[Item Name]]` markers):** with items now regularly appearing via an automatic roll rather than model-authored text, `_NARRATOR_BASE` (`prompts.py`) instructs the narrator to wrap any concrete item name in double square brackets wherever it's mentioned — loot lines, inventory mentions, a weapon named mid-fight — mirroring the existing 🎲/💰 line-marker convention. This backs the item-detail popup (see below): the frontend parses `[[...]]` into a clickable span, invisible to the reader otherwise.

### Stalled non-player-turn guardrail (2026-07-09)

Reported live: mid-combat, the DM asked the player to take "Elara's" turn — a DM-controlled companion, not the player's own character (Tarvokk). The Combat prompt section already correctly instructs auto-continuing through every non-player turn in one response ("if the next combatant(s) in initiative order are monsters or DM-controlled companions... keep going in this same response... until the initiative order comes back around to a turn belonging to a player-controlled character") — this was a compliance miss, not a missing instruction, but nothing deterministically caught it, unlike the loot/encounter guardrails above.

`_detect_stalled_non_player_turn_followup` (`dm_agent.py`) closes that gap: after each mechanics response, if there's an active encounter with no `pending_action` (a real reaction prompt legitimately awaiting the player — exempted), it checks live `initiative_order` for whichever combatant is currently `is_current_turn`. If that combatant is a monster, an NPC, or a `Character` with `is_player_controlled=False` (a DM companion), it fires a correction telling the model to call `advance_initiative` and resolve that turn itself rather than waiting on the player. Verified with four cases before wiring in: DM companion's turn (fires), player's turn (doesn't fire), monster's turn (fires), monster's turn with a pending reaction prompt (doesn't fire — the exemption).

**Recurred within the hour — a budget-starvation bug, not a detection bug.** Reported live again: the DM stopped on Kaelen Swiftstep's turn (also a real DM companion, confirmed `is_player_controlled=false` in the DB) and asked the player to act. The logs showed why: `_detect_missing_combat_roll_followup` had already fired earlier in the *same* response (auto-continuing through several combatants per the Combat section's own rule can cover many turns in one reply) and spent the turn's one `correction_count` retry before the mechanics model ever reached Kaelen — exactly the starvation `lore_guardrail_count`'s own doc comment already predicted for a different pair of guardrails, just not yet triggered for this one. Fix: pulled `_detect_stalled_non_player_turn_followup` out of the shared `correction_count` chain into its own budget, `stalled_turn_guardrail_count` (`DMState`, reset alongside the others in `stream_response`) — same shape as `lore_guardrail_count`'s own split from `correction_count`. Re-verified the detection logic still fires correctly for the exact reported case (Kaelen, DM companion, `is_current_turn`) after the refactor.

**Recurred a third time — the guardrail's one retry got spent on a different problem, again.** Reported live once more, this time on Thrainna Stoneheart's turn (also confirmed `is_player_controlled=false`) — asked directly, mid-round, skipping what should have been Elara's and Tarvokk's next turns. Three separate DM companions (Elara, Kaelen, Thrainna) each independently triggered this same class of mistake over one fight, which reframed the problem: the model's own free-text tracking of "whose turn is next" is fundamentally unreliable across a long, multi-combatant auto-continued response, not just occasionally wrong. A per-guardrail retry budget helps but doesn't fully close it, since a single response can still contain more distinct stalls than it has retries for.

The fix moves from "detect and correct after the fact" to "state the fact so there's nothing to get wrong": `_live_current_turn(campaign)` (new, factored out of the guardrail) is the one shared live-state lookup, and `_next_turn_ground_truth_note()` calls it unconditionally at the very end of `mechanics_node` — no retry budget, because it never loops, it just appends a `[GROUND TRUTH — ...]` line to the resolution report stating exactly whose turn it live-is and whether they're player-controlled. `_NARRATOR_BASE` (`prompts.py`) now instructs the narrator that this line, when present, overrides its own read of the scene entirely: name exactly the stated character if player-controlled, or don't prompt the player at all if not. This runs *in addition to* `_detect_stalled_non_player_turn_followup` (still valuable — it's the one that can actually force a real re-resolution via a mechanics retry), not instead of it: the guardrail tries to fix the underlying resolution when it has budget left, and the ground-truth note guarantees the player is never shown a wrong or premature turn prompt even in the worst case where it doesn't. Verified against the exact reported shape (a DM companion mid-round, and a genuine player turn) after fixing one bug caught in testing — the player-controlled branch's `[GROUND TRUTH — ...]` string was missing its closing bracket.

### The actual root cause behind all of tonight's combat bugs: `create_monster` crashing on a bare `KeyError` (2026-07-09)

Traced all the way back after the turn-order fixes above kept not helping a specific live session: `campaign.active_encounter` didn't exist at all — zero rows in the `monsters`/`encounters` tables for the whole rest of the fight. Every guardrail and ground-truth fix built earlier tonight depends on a real `Encounter` existing to check against; with none there, they were all silent no-ops, not failures — there was nothing for them to catch.

Inspected the actual LangGraph checkpoint directly (`AsyncPostgresSaver` pointed at the live thread, same technique as the Session 0 investigation and the world-prep freeze diagnosis) to see the real tool-call history, not guess from the narrated transcript. Found it precisely: the model correctly called `create_monster` for the goblins — for real, four times in a row — and every single call crashed. `attacks: list[dict]` was processed with `Attack(name=a["name"], ...)`, a bare dict index with no validation; the model's payload never included a `"name"` key on the attack dict, so this raised `KeyError('name')`, which the generic tool-error handler (`_handle_any_tool_error`) stringified via `str(e)` into literally `"Error: 'name'"` — a message carrying zero information about what was wrong or how to fix it. The model retried the identical mistake three more times against that same useless message (it tried `search_rules` for a goblin stat block twice in between, looking for grounding that wouldn't have helped), then gave up and fabricated the entire rest of the encounter in narration — no monster, no encounter, ever actually created — which is exactly why the turn-order ground-truth/guardrail fixes above had nothing to engage with for the rest of that fight.

Fix: `create_monster` (`backend/tools/combat.py`) now validates every `attacks[i]` has a non-empty `name` up front and returns a clear, actionable message (naming the exact index and showing a correct example) instead of letting it crash into an opaque `KeyError`. Verified against the exact failing payload from the real transcript — now returns the clear message instead of crashing — and against a corrected payload, which creates the monsters successfully.

Broader lesson, worth stating plainly: a generic `except Exception: return str(e)` tool-error handler is only as good as the exceptions it's converting — a bare `KeyError`/`AttributeError` from an unguarded dict/attribute access makes a fine crash but a terrible corrective message, and the model has no way to self-correct from one. Every other tool's error paths in this codebase already return hand-written, specific messages (`"No character named '{name}'..."`, `"'{base_item}' isn't a recognized weapon..."`) precisely for this reason; `create_monster`'s `attacks` handling was the one spot that fell through to the generic path instead. Worth a pass over `backend/tools/` for other raw dict/list indexing inside a `@tool` function that could hit the same failure mode, not yet done tonight.

### Narrator inventing a character's weapon (2026-07-09)

Reported live: the narrator described Elara "her Shortsword drawn," but her actual (and only) attack is a Rapier — confirmed on her own character sheet. The roll numbers shown (+5 to hit, 1d8 piercing) exactly matched Rapier, so the mechanics layer resolved this correctly (`resolve_attack` was almost certainly called with the real, grounded `attack_name`); the hallucination was purely in the narrator's prose.

Root cause: the mechanics resolution-report instruction (`_MECHANICS_BASE`'s "Resolution report" section) required naming the roll *type* ("attack roll," "damage roll") but never required naming the specific weapon/spell used — so that fact could be, and was, silently dropped between the tool call (which knew "Rapier") and the report the narrator actually reads. Compounding it: `_campaign_block()` (shared context for both mechanics and narrator prompts) listed each character's race/class/level/flavor but never their actual equipped attacks — so the narrator had no independent ground truth to fall back on either, the same gap that let it invent a plausible-sounding but wrong weapon.

Two-layer fix, matching this session's other guardrail-plus-grounding fixes: (1) the mechanics report instruction now explicitly requires naming the exact `attack_name`/`spell_name` alongside every attack/damage roll, never paraphrased; (2) `_campaign_block()` now lists each character's `Real attacks:` (their actual `Character.attacks` names) as a standing reference, so the narrator has a ground-truth list to check against even if a future resolution report is incomplete for some other reason; (3) `_NARRATOR_BASE` gained an explicit instruction to name only the report-stated (or "Real attacks"-listed) weapon/spell, calling out that inventing one is the same class of error as inventing an unbacked loot line. Verified `_campaign_block()` renders the new line correctly (`Real attacks: Rapier` for a test character with one Attack).

### `resolution.py` (6 tools, new 2026-07-03) — atomic dice-resolution, not combat.py-scoped
| Tool | Description |
|---|---|
| `resolve_attack(attacker, target, attack_name/spell_name, attack_count=1, end_turn=False, …)` | Roll-to-hit + crit/fumble + damage roll + HP application in one call. `attack_count>1` resolves a same-target Multiattack in one call. Pauses (returns PENDING) instead of applying damage if the target is a reaction-eligible, conscious, player-controlled character. Refuses if the attacker is at 0 HP (unconscious — see `resolve_death_save`). |
| `resolve_saving_throw(target_names, ability, dc, …)` | Save roll + optional damage/condition application for one or many targets (AoE) in one call. |
| `resolve_check(character_name, ability_or_skill, dc=None, …)` | Ability/skill check with the character's real modifier (proficiency/expertise) looked up automatically — no separate `get_character` call needed. Refuses if the target is at 0 HP (incapacitated). |
| `resolve_death_save(character_name)` | Rolls a death saving throw — the only legal action for a combatant at 0 HP on their turn. Tracks `Character.death_save_successes`/`death_save_failures` (both real fields, previously dead code — wired up 2026-07-03): nat 20 revives with 1 HP, nat 1 = 2 failures, 3 successes stabilizes, 3 failures kills. Damage taken while already at 0 HP is a separate path — handled automatically by `apply_damage_to_character` (1 failure, 2 on a crit, or instant death if a single hit's damage ≥ max HP) whenever `resolve_attack`/`resolve_saving_throw`/`resolve_pending_action` applies it, not through this tool. |
| `resolve_pending_action(reaction_declared="", ac_bonus=0, damage_reduction=0, damage_multiplier=1.0)` | Finishes an attack that paused for a reaction (Shield/Parry/Uncanny-Dodge-style). |
| `cast_spell(caster_name, spell_name, target_names=[], slot_level=None, as_ritual=False, …)` | Consumes a spell slot and resolves a known spell's stored `resolution_type` (`Spell.resolution_type`: `attack_roll` / `saving_throw` / `automatic`, plus `effect_dice`/`save_ability`/`damage_type`/`is_healing`/`half_damage_on_success`/`condition_on_fail`) atomically. Spell data now populated (see spell-selection feature below and `backend/data/spells.py`) — characters created or backfilled after 2026-07-03 have real `spells_known`. Also refuses if the caster is at 0 HP. Checks Material ("M") components as of 2026-07-04 — refuses (no slot consumed) if the spell needs a focus/pouch, or a specific costly named component, and the caster's inventory doesn't have one; Verbal/Somatic requirements are NOT checked (see "Deferred from the combat resolution refactor" below). `as_ritual=True` (added 2026-07-03) casts a `Spell.ritual=True` spell per the 2024 PHB's general ritual rule — no slot spent, 10 fictional minutes longer — refusing for a cantrip, a non-ritual spell, or one not currently prepared; incompatible with `slot_level` upcasting. See "Ritual casting" below. |

### `world.py` (4 tools)
| Tool | Description |
|---|---|
| `get_current_location()` | Description, terrain, lighting, exits, present NPCs. |
| `move_party(location_name)` | Move party to a named location. |
| `reveal_hidden_element(location, index)` | Move a hidden element to visible points of interest. |
| `open_container(name)` | Open a container; list items and currency. |

### `quest.py` (3 tools)
| Tool | Description |
|---|---|
| `get_active_quests()` | All active quests with objectives and reward summary. |
| `complete_quest_objective(quest, index)` | Mark one objective done. |
| `change_quest_status(quest, status)` | Set status to active / completed / failed. |

### `campaign.py` (2 tools)
| Tool | Description |
|---|---|
| `get_campaign_summary()` | Party, location, active quests, time, weather, combat state. |
| `add_session_note(note)` | Append a key event note (legacy; session chronicles now handled by `session/end`). |

### `chargen.py` (6 tools — Session 0 only)
| Tool | Description |
|---|---|
| `list_options(category)` | List available races, classes, backgrounds, ability score methods, or (since 2026-07-03) `"spells <class>"` — the curated cantrip/level-1 menu from `backend/data/spells.py`, headed by the exact required counts; non-casters get "has no spellcasting." |
| `get_option_details(category, name)` | Full detail for a specific race, class, background, or (since 2026-07-03) `"spell"` — resolution type, damage/save info, description. |
| `roll_ability_scores()` | Roll 4d6-drop-lowest × 6 with full breakdown. |
| `update_character_draft(field, value)` | Write a field to the in-progress character draft, including (since 2026-07-03) `spells_known` — comma-separated, combine cantrips and level-1 choices in one call. Called after each confirmed choice to keep the live preview current. |
| `get_draft_summary()` | Return current draft state with missing-field report, including chosen spells. |
| `finalize_character()` | Validate draft completeness (since 2026-07-03, including spell selection for a caster: every name must be on that class's curated menu, and the per-tier count must exactly match `SPELL_REQUIREMENTS` — rejected with a corrective message otherwise, same pattern as the existing missing-ability-scores check), build a `Character` with derived stats (HP, AC, passive perception, spell slots, `spellcasting_ability`/`spell_save_dc`/`spell_attack_bonus`, `spells_known`/`spells_prepared`), append to `campaign.party`, save to Postgres, clear the draft. |

**Interactive spell selection (2026-07-03).** `cast_spell` (`resolution.py`, built earlier the same
session) had nothing to work with — no character ever got `spells_known`/`spellcasting_ability`
populated. A first proposal (auto-assign each class a small fixed default, mirroring
`equipment.py`'s `STARTING_KITS`) was rejected: 5e splits casters into those who
permanently pick specific spells and those who can reselect some/all on a long rest,
and a fixed default represents neither. Real interactive selection was built instead:
`backend/data/spells.py` (new) — `ALL_SPELLS` (41 spells, transcribed and mechanically
verified directly against the in-repo 2024 PHB text at
`docs/source/core/D&D 5.5E - Player's Handbook.md`, correcting OCR noise like "ldlO" ->
"1d10"; two spells, Magic Missile and Witch Bolt, weren't present in that text — full
stat blocks missing, index-only — and were authored from well-established 5e knowledge
instead), `SPELL_MENUS` (curated per-class subsets, sized larger than required so the
choice is real — Wizard's level-1 menu is 8 options against a 6-spell requirement),
`SPELL_REQUIREMENTS` (flat counts per the 2024 PHB — confirmed the rules do NOT use an
ability-modifier formula for spell counts the way 2014 did; every class's spellcasting
section states a literal "choose N"), `SPELLCASTING_ABILITY`. Two new functions in
`_helpers.py`: `derive_spellcasting_stats` (pure arithmetic: `8+prof+mod`/`prof+mod`)
and `build_spells_known` (validates chosen names against the menu, returns a
corrective error string on a bad name or wrong count). Wired into `_build_character()`,
`generate_companion_character()` (`companion.py`, gained an optional `spells_known`
param with the same validation — built now rather than deferred, since skipping it
would leave every DM-generated companion permanently spell-less), and a new
`backfill_character_spells.py` (mirrors `backfill_character_equipment.py`'s exact
idempotent-untouched-check pattern; auto-picks the first N names per tier from
`SPELL_MENUS`, defensible specifically because those menus are ordered with each
class's PHB-recommended starters first — confirmed against the source text for
Cleric, where the recommended set matches exactly). Run against the real "Yawning
Portal" campaign: Xander (Ranger), Lana (Cleric), Sir Valiant (Paladin), and Eldrin
(Wizard) backfilled correctly; Mira (Rogue, non-caster) correctly untouched; a second
run confirmed idempotency (0 backfilled, all skipped).

**Wizard's spellbook-vs-prepared distinction is a stated RAW deviation, not an
oversight**: 2024 rules give Wizards a 6-spell spellbook but only 4 castable at a time
(reselectable each long rest). No reselection tool exists, so gating to 4 would make
the other 2 permanently inaccessible — worse than not distinguishing at all. All 6
chosen spells are stored in both `spells_known` and `spells_prepared` (all castable).
See "Deferred" below for what a real fix would need.

**Verified live, and a real, more serious pre-existing bug found along the way
(2026-07-03):** a live Session 0 test (Wizard, the structural spellbook-outlier case)
first hit a `GraphRecursionError` — `session_zero_stream` (`main.py`) had never set
`recursion_limit` on its LangGraph config, silently running at the framework default
(25) while the main game's `stream_response` had long since raised this to 60 for the
same reason (a turn walking through several tool calls in a row). Fixed by adding the
same override. Re-running past that, a second, much more serious issue surfaced: the
model used by `_get_model()` — hardcoded to `"qwen2.5:14b"`, a smaller, separate model
from `settings.mechanics_model`, shared by Session 0/world-prep/party-fill/session
summarization — walked through an entire multi-turn character-creation conversation
writing convincing prose and fake fenced ` ```json` blocks that *looked* like tool
calls, then confidently declared the character "successfully created," while
`DraftStore`'s actual live state (checked via the real `/draft` HTTP endpoint, since
`DraftStore` is in-memory and per-server-process) stayed completely empty the entire
time — `finalize_character` was never genuinely called. This is the same quirk already
documented at `run_fill_party`'s comment about `qwen2.5:14b` appending a decorative,
never-executed fenced json block after its real summary — this session's test just
hit a far more severe instance of it (the *entire* conversation, not a decorative
extra). Root-caused and fixed by switching `_get_model()` to `settings.mechanics_model`
(`gemma4:26b-mlx`) — the same model already extensively live-tested elsewhere in this
app for reliable, genuine tool-calling discipline, including self-correcting after
guardrail rejections across multi-turn combat. Standardizing on one validated model
closes this failure class at the root rather than patching around it per call site
(as `run_fill_party`'s existing fenced-block-stripping code had to). Re-testing the
same repro after the model switch surfaced a third, distinct bug: past ~9-10 turns the
model's output degraded into literal garbled tokens (`<channel|>thought` fragments,
stray `</div>` repeats) leaking into the visible reply, with `DraftStore` never
actually updated despite fluent on-topic prose. Root-caused to Session 0 having no
equivalent of the main game's `mechanics_node` intra-turn scratch purge (see line 443)
— unbounded tool-call scratch accumulating in the persisted checkpoint across a long
chargen conversation, `_MAX_MESSAGES=100` only trimming what's sent to the model per
call, never what's retained. Fixed by adding `_purge_session_zero_turn_scratch`
(`dm_agent.py`), wired in as `create_react_agent(..., post_model_hook=...)`: after each
model call, if the response has no pending tool calls (i.e. it's the turn's final
reply), `RemoveMessage`s every scratch message since the last human turn *except* that
final reply itself — the one deliberate difference from `mechanics_node`'s purge, which
discards its own final message too because a separate narrator node replaces it; Session
0 has no narrator, so its own reply is the one thing that must survive. Verified on a
fresh campaign/thread with the same ~10-turn Wizard repro that previously garbled:
checkpoint message count stayed flat (~2-4/turn, never compounding) across the whole
conversation, no garbled tokens at any point, and `finalize_character` produced a
genuine success — real party member with correct `spellcasting_ability`/`spell_save_dc`
and all 9 selected spells recorded in `spells_known`. Confirms the unpurged-scratch
hypothesis was the actual root cause; the residual "compiled Ollama parser bug" hedge
never needed to fire.

**Superseded 2026-07-04:** `create_react_agent(..., post_model_hook=_purge_session_zero_turn_scratch)`
described above no longer exists — `get_session_zero_agent` was restructured into its
own two-node mechanics/narrator `StateGraph` (see Agent Architecture's "Session 0
agent" section), whose `chargen_mechanics_node` does the same inline scratch-purge
`mechanics_node` always has. Separately, the raw `<channel|>thought` tag leakage
itself (as opposed to the unbounded-scratch problem the purge fixed) has since had
its actual mechanism identified — see Agent Architecture's "Reasoning-tag leak fix."

**Another real, longstanding gap found and fixed (2026-07-03), unrelated to the above:**
a player noticed the real Yawning Portal campaign's character sheet showed no Skills
section for Xander specifically — his `skill_proficiencies` was genuinely empty (the
other four party members had theirs set correctly), a one-off miss from his creation.
Fixed directly in the DB with the player's chosen skills (Athletics, Investigation,
Perception), recomputing `passive_perception` accordingly (11 → 13, now Perception-
proficient). Investigating turned up a deeper, campaign-wide bug: `Character.
saving_throw_proficiencies` (used by `resolve_saving_throw`'s `_save_bonus` to add
proficiency bonus) had existed on the model since the combat-resolution refactor but
was **never populated by chargen.py or companion.py for any character, ever** — every
saving throw in every campaign had been rolling with no proficiency bonus regardless of
class. Fixed by adding `derive_saving_throw_proficiencies(char_class)` (`_helpers.py`)
— a pure lookup against each class's two listed saves in `fivee_options.CLASSES`, no
per-character choice involved (unlike skills/spells) — and wiring it into both
`_build_character` (`chargen.py`) and `generate_companion_character` (`companion.py`).
Backfilled all existing characters via `backfill_character_saving_throws.py` (same
idempotent-untouched-check pattern as the equipment/spell backfills); ran against the
real Yawning Portal campaign — all 5 party members backfilled correctly (e.g. Ranger →
{strength, dexterity}, Wizard → {intelligence, wisdom}), re-run confirmed idempotent
(0 backfilled, all skipped).

**Loot tool-calling gaps found live, both fixed 2026-07-03:** watching real play in the
Yawning Portal campaign surfaced two distinct ways narrated loot could go missing from
actual character state. (1) A real Investigation/Sleight of Hand-backed find (a coin)
never reached `add_item_to_character` — the mechanics prompt had explicit tool-call
instructions for magic items and for purchases, but nothing for mundane loot found via
exploration/search/looting; fixed with a new "## Loot" section in `_MECHANICS_BASE`
(`prompts.py`) modeled on the existing magic-item bullet. (2) Worse, a follow-up
question ("did anyone else find anything?") produced an entirely fabricated find with
**no roll or tool call backing it at all** — confirmed by inspecting the live LangGraph
checkpoint directly (a fresh `AsyncPostgresSaver` pointed at the same Postgres backend,
same technique established during the Session 0 investigation) and cross-checking the
character's actual DB currency/inventory, both unchanged. Fixed with a second guardrail
bullet in `_MECHANICS_BASE`'s "ALWAYS use tools — never invent" list: a follow-up
question about an unresolved character's action must trigger a real `resolve_check`
before reporting any outcome, never an invented one. Also added, per the reporting
player's own suggestion: a 💰 "loot line" narrator convention (`_NARRATOR_BASE`,
`renderDmParagraphs` in `game.html`, `.loot-line` in `style.css`) mirroring the existing
🎲 roll-line marker — the narrator may only emit one when the mechanics resolution
report explicitly states an item/currency change, making a real grant visible (and a
fabricated one conspicuously absent) directly in the transcript.

**Ritual casting, added 2026-07-03:** a player asked whether ritual-castable spells
(cast without spending a slot, 10 minutes longer) were modeled at all — they weren't,
across all three layers. `Spell.ritual: bool` existed on the model since the
combat-resolution refactor but was never set `True` for any spell, `cast_spell` had no
ritual parameter, and the mechanics prompt never mentioned the option. Grounded in the
2024 PHB's general "Casting Without Slots" rule (any class can ritual-cast a *prepared*
Ritual-tagged spell — no per-class "Ritual Casting" feature gate the way 2014 had one;
Wizard's separate "Ritual Adept" bonus of skipping the prepared requirement is a no-op
here since this app already flattens every known Wizard spell into `spells_prepared`,
see the Wizard deviation note above). Of this app's 41-spell curated menu, exactly 4
spells actually carry the Ritual tag per the source PHB text: Detect Magic, Identify,
Comprehend Languages, and Speak with Animals (whose `casting_time` was also wrong —
"1 action" instead of "1 action or ritual" — until now). Fixed by setting
`ritual=True` on those four (`spells.py`), adding `cast_spell(..., as_ritual=False)`
(`resolution.py`) — refuses for a cantrip, a non-`ritual` spell, an unprepared spell, or
combined with `slot_level` upcasting; skips slot consumption entirely when it succeeds
— and a new mechanics-prompt bullet instructing the model to surface the choice (never
silently assume it) when a prepared ritual spell is being cast outside combat/urgency.
Verified directly against `cast_spell` (not yet re-tested live in-game): ritual cast of
Detect Magic succeeds with slots unchanged; the same call with Magic Missile (known,
not ritual-tagged) and Fire Bolt (a cantrip) both correctly refuse; a normal
(non-ritual) cast of Detect Magic still consumes a slot as before; an unprepared
ritual-tagged spell still correctly refuses via the pre-existing prepared-check.

**Follow-up — three issues reported live, investigated 2026-07-13. Items 1-2
resolved; item 3 confirmed to be model non-compliance, not a code bug (nothing to fix
there beyond the guardrail idea noted at the end). Regression coverage:
`tests/test_chargen_and_character_details.py`.**

1. **Starting equipment missing daggers (Ranger, Rogue) — fixed 2026-07-13.**
   `STARTING_KITS` (`backend/data/equipment.py`) is a flat per-class dict —
   `"weapon"` becomes the character's one mechanical `Attack`, and any second weapon
   rides along in `"gear"` (`_helpers.py`'s `_starting_equipment` turns each `gear`
   string into an `Item`). Rogue's kit already listed `"Shortbow"` in `gear` (so that
   half worked), but **neither kit listed a Dagger**, despite both classes' real 2024
   PHB "Option A" kit including two (confirmed directly against
   `docs/source/core/D&D 5.5E - Player's Handbook.md`'s own Starting Equipment table
   rows for Ranger/Rogue). Added `"Dagger (x2)"` to both kits' `gear` lists.
2. **Starting gold correction — fixed 2026-07-13, and a correction to this doc's own
   earlier note.** Investigating "gold should be rolled, not flat" surfaced something
   that changes the fix: the **2024 PHB does not roll dice for starting gold at all**
   — every class's Starting Equipment row is "Choose A or B," where A is a fixed kit
   + a small flat GP remainder and B is a larger flat GP total with no items. There is
   no "5d4 × 10 gp"-style roll anywhere in this ruleset's actual source text (that's a
   2014-edition habit, not what this app is grounded in). So the real bug wasn't
   "flat instead of rolled" — it was that `STARTING_KITS`' flat GP values don't match
   the real Option A remainders for every class (checked against the source table:
   Ranger's real Option A leftover is 7 gp, Rogue's is 8 gp; this app had both at a
   generic 10). Corrected Ranger → 7, Rogue → 8 gp alongside the dagger fix above.
   **Full-table pass done, 2026-07-13:** all 12 classes' `STARTING_KITS` entries
   rewritten to match their real Option A row exactly (weapon/armor/shield/gear/gold),
   transcribed directly from the PHB's per-class Starting Equipment table cells —
   same discipline as `spells.py`/this file's own WEAPONS/ARMOR tables. Added five
   WEAPONS entries that didn't exist yet but are needed by real Option A kits
   (Sickle, Spear, Greatsword, Flail, Javelin — stats transcribed from the PHB's own
   weapon table, `docs/source/core/D&D 5.5E - Player's Handbook.md`). Several classes'
   mechanical mismatches went well beyond the two originally reported (Ranger/Rogue
   daggers): Bard and Rogue's `"weapon"` (Rapier) wasn't in either real kit at all;
   Fighter had Longsword + a Shield neither real option grants; Cleric/Druid/Ranger's
   armor didn't match (Scale Mail vs. real Chain Shirt, Leather vs. real Studded
   Leather); several classes' gold was a generic round number instead of the real
   per-class Option A remainder. Verified live: built a level-1 character of every
   class through `_starting_equipment()` directly, confirmed each one's attack/armor/
   gear/gold now matches its real PHB row.
3. **Ability-check die/modifier breakdown sometimes not narrated.** Checked whether
   this is a data gap — it isn't. `resolve_check` (`resolution.py`) genuinely computes
   and returns the full breakdown string (`"{name} — {label} check: d20 {roll} + {mod}
   = {total}"`, proficiency/expertise already folded into `mod`), `resolve_check` is in
   `_VERIFIED_ROLL_TOOLS` (`dm_agent.py`), and the narrator prompt (`prompts.py`)
   explicitly instructs copying that exact string into a 🎲 line, "never recompute,
   round, or 'correct' them." So when a check gets narrated as a bare result with no
   roll shown, that's the narrator model failing to follow an already-correctly-plumbed
   instruction, not a missing-data bug — same shape of problem the loot-line and
   verified-roll guardrails elsewhere in this doc exist to catch for combat rolls. Not
   yet checked: whether the existing guardrail-detection functions (e.g.
   `_detect_missing_combat_roll_followup`) actually cover `resolve_check` output the
   same way they cover combat/damage rolls, or whether there's a narrower enforcement
   gap for plain ability checks specifically that a similar regex guardrail could close.

**Character/companion pronouns — added 2026-07-13.** Reported live: a player couldn't
tell how the DM should refer to a party member (Kaelen) because nothing ever asked.
`Character.pronouns: str` (freeform, `models.py`) is now collected the same way
`appearance` already was — `update_character_draft`'s `pronouns` field
(`chargen.py`), a matching `DraftStore._empty()` key, a new Session 0 prompt step
right next to the existing appearance-ask instruction (never inferred from name/race),
and a `pronouns` param on `generate_companion_character` (`companion.py`) for
DM-generated companions. Surfaced in the character sheet preview
(`static/character-sheet.js`'s `renderCharacterSheet()`, next to the name) — shared by
both `game.html` and the Session 0 lobby's sheet preview (see the clickable-cards item
above). `NPC.pronouns` was also added to the model for symmetry but **is not yet wired
into `create_npc`** — narrator-facing NPCs still have nowhere to record this.
**Fixed 2026-07-13:** added `update_character_detail(character_name, field, value)`
(`party.py`) — the in-game counterpart to `update_character_draft`, for a party member
who was already finalized before this fix (or whose player skipped the question).
Deliberately narrow: only `pronouns`/`appearance`/`alignment`/`notes` are editable —
mechanical fields (race, class, ability scores, equipment, ...) aren't, since those
cascade into derived stats (HP, AC, spell slots, ...) this tool doesn't recompute, and
changing them mid-campaign is a DM judgment call beyond "fix a forgotten detail."
`value="CLEAR"` blanks a field (`None` for `alignment`, matching its `str | None`
type; `""` for the others). Regression coverage for both the draft→finalize
pronouns path and `update_character_detail` (including its mechanical-field
refusal and the alignment `CLEAR` → `None` behavior):
`tests/test_chargen_and_character_details.py`.
changing them mid-campaign is a bigger DM decision than "fix a forgotten cosmetic
detail." `value="CLEAR"` blanks a field (`None` for `alignment`, matching its
`str | None` type; `""` for the others).

### `companion.py` (1 tool — Session 0 and in-game)
| Tool | Description |
|---|---|
| `generate_companion_character(...)` | DM-steered (not random) creation of a level-1 `is_player_controlled=False` companion, added directly to `campaign.party` in one call — no per-player draft. The DM/agent picks every field deliberately after checking `get_campaign_summary` for current party composition and the adventure's recommended party size (see `_meta.json`), choosing a build that complements rather than duplicates the existing party. Shares derived-stat math (`derive_level1_stats` in `_helpers.py`) with `finalize_character`. Reachable two ways: conversationally (mechanics prompt / Session 0 prompt both instruct the agent to offer this when the party's short) or via a dedicated "Ask DM to add a member" button on the Session 0 lobby page (one companion per click, by design), which runs a one-shot non-conversational agent (`get_party_fill_agent` / `run_fill_party` in `dm_agent.py`, same one-shot pattern as `get_world_prep_agent` — no checkpointer, not a resumable thread). The button has a class dropdown next to it, defaulting to "Random (DM decides)"; picking a specific class bypasses the model's own class judgment entirely (see `_fill_party_prompt` in `dm_agent.py`) and also skips the recommended-size gate, since picking a class is itself an explicit DM decision to add someone. **Verified bug, fixed 2026-07-02:** with only a general "complement, don't duplicate" instruction, `qwen2.5:14b` anchored on generating Cleric regardless of actual composition — reproduced by seeding a party of three Clerics and asking it to fill a fourth slot; it added a *fourth* Cleric while its own summary said the party "lacks variety beyond Clerics." Root cause was likely the prompt's own illustrative example ("don't add a fourth Rogue to a party with no healer") anchoring the model on "healer." Fix: compute the party's overrepresented classes in Python and state them as a direct fact in the prompt, rather than relying on the model to infer duplication from `get_campaign_summary`'s prose. Re-verified against the same seeded scenario — correctly added a Rogue instead. |

### `levelup.py` (1 tool — in-game only)
| Tool | Description |
|---|---|
| `level_up(character_name, new_level, new_spells_known="", subclass="")` | Advances a character to a higher level in one call, recomputing HP, proficiency bonus, spell slots, stored weapon `to_hit_bonus` values, and hit dice from real data — the same "never let the model invent a number" discipline chargen already applies to character creation. `new_level` is an absolute target (supports multi-level jumps for a big milestone), rejected if not strictly higher than the character's current level or above 20. Validates any `new_spells_known` (against the class's `SPELL_MENUS`, rejecting duplicates already known) *before* mutating anything else, so a rejected call leaves the character completely untouched — verified directly (a bad spell name left level/HP/attacks unchanged; a valid retry then succeeded). |

**Why this was built (2026-07-03):** a real live session had the DM narrate "The party has reached Level 2" as a dramatic beat with a full paragraph of flavor text — but no leveling mechanism existed anywhere in this codebase before this tool, so nothing backing it had actually happened. Confirmed by checking the real party's DB state directly: every character was still level 1, unchanged HP/proficiency/slots. Same failure class as the loot/Sir-Valiant bugs from earlier the same day (a model narrating a mechanical outcome with zero tool call behind it), just for an entirely unbuilt feature rather than a missing prompt instruction.

**Design choices** (via `AskUserQuestion`, since these are genuine rules choices, not obvious defaults):
- **HP gain uses the fixed average** (`hit_die // 2 + 1 + CON mod` per level gained), not a rolled hit die — matches how level-1 HP is already computed deterministically (max, no rolling), avoids introducing a new "does everyone agree to roll or take average" table rule mid-campaign.
- **New known spells at level-up reuse the same interactive pattern as Session 0** — `list_options('spells <class>')` / `get_option_details`, then pass the chosen name(s) to `new_spells_known`, validated against the class's menu exactly like chargen. No auto-pick shortcut, consistent with the original interactive-spell-selection decision earlier this session.
- **The real Yawning Portal party was patched to Level 2 immediately**, via the actual `level_up` tool (not a raw DB edit) to both fix the live campaign and prove the tool end-to-end: Xander (Ranger) and Sir Valiant (Paladin) → HP 20, 2 level-1 slots (half-caster table, no growth yet at level 2); Lana (Cleric) → HP 13, 3 slots; Eldrin (Wizard) → HP 12, 3 slots, plus 2 new spellbook spells (Identify, Comprehend Languages — the 2024 PHB's "Wizard adds 2 spells to spellbook every level after 1" rule, filled from his level-1 menu's only 2 remaining unpicked entries); Mira (Rogue, non-caster) → HP 17, no spell changes. All verified against the live `/party/{id}` endpoint post-patch.

**Known gaps, stated explicitly:**
- **No per-level "spells known" count table** — `SPELL_REQUIREMENTS` (spells.py) is level-1-only. The tool relies on the calling model checking `search_rules`/class text for whether a class gains a new known spell at a given level, rather than enforcing an exact count itself. Only Wizard's "+2 spellbook spells every level after 1" is unambiguous enough to have been applied confidently above; Ranger/Sorcerer/Bard/Warlock's exact per-level known-spell growth isn't hardcoded anywhere in this app yet.
- **No level-2+ spell content exists at all** — `ALL_SPELLS`/`SPELL_MENUS` only cover cantrips and level-1 spells (see spells.py's scope). A full caster reaching character level 3 (first access to 2nd-level spell slots) has no real 2nd-level spell to select — `level_up` will grant the slot correctly but there's nothing in the menu to fill it with yet. Not addressed here — flagged for whenever spell data gets extended beyond level 1.
- **No Ability Score Improvement / feat selection** — real 5e grants these at specific levels (4, 8, ...); `level_up` doesn't model or prompt for them at all.
- **Subclass isn't validated** against the class's real subclass list (`level_3_features` text names them, e.g. "Battle Master, Champion, Eldritch Knight, or Psi Warrior" for Fighter) — `subclass` is a free-text passthrough, same as it already was in `chargen.py`.

**Level-up timing:** decided (via `AskUserQuestion`) to leave `level_up` ungated — real 5e milestone leveling happens whenever the DM decides the story earned it, not only after a rest. No prompt change needed; tonight's mid-dungeon level-up was already correct as narrated once the tool existed to back it.

### Rest buttons (2026-07-03) — `POST /rest/long`, `POST /rest/short`

**A second real bug found while investigating the level-up gap:** the sidebar's "Long rest taken today" status line (`campaign.last_long_rest_day == campaign.days_elapsed`) was permanently stuck on "No long rest today" — `last_long_rest_day` is declared on `Campaign` but was never assigned anywhere in the codebase. There was also no long-rest *button* at all, despite the user's impression there was one — only a conversational path (`restore_spell_slots`, called per-character, easy for the model to forget or only apply to some of the party) and no short-rest mechanism whatsoever.

Fixed with two new deterministic, whole-party endpoints — **deliberately not routed through any LLM**, since this is pure arithmetic with a single correct answer, not a narrative choice, matching this session's broader lesson about not trusting a model for state changes it can just as easily get wrong:
- `apply_long_rest`/`apply_short_rest` (`_helpers.py`), full 5e-rules bulk effects: long rest = full HP, all spell slots restored, exhaustion -1, death saves cleared, hit dice regained (half of total, rounded down, min 1), clock +8h, and (finally) `last_long_rest_day` actually set. Short rest = clock +1h, Warlock Pact Magic slots restored (the one class whose slots recharge on a short rest), and HP via hit dice.
- **Short rest HP simplification**: real 5e lets each player choose how many Hit Dice to spend; with no per-character interactive UI for that, every character automatically spends *just enough* hit dice (average value per die, same fixed-average approach as `level_up`'s HP gain) to reach full HP, capped at whatever they have remaining — not "spend everything." **Caught and fixed during testing**: an early version spent every remaining hit die regardless of how little healing was actually needed, and separately reported the *uncapped* healed amount in its summary text even when the max-HP cap reduced the real gain — both fixed and reverified (a lightly-hurt character now saves unneeded dice; the reported number always matches the real HP change).
- Both buttons live in the sidebar's World section (`game.html`), call their endpoint directly (no LLM round trip, so effectively instant), show the plain-text summary, then reload the page to reflect new HP/slots/clock everywhere.
- Verified live end-to-end against a disposable test campaign (not the real Yawning Portal game): damaged a companion to 3 HP with 0 hit dice remaining, short rest correctly reported "no healing needed/no hit dice left," long rest fully healed them, restored slots, and flipped the rest-status line to "Long rest taken today."

### Out-of-character (OOC) input (2026-07-03)

Surfaced by the same session's confusion: every player message goes through the same in-fiction mechanics→narrator pipeline, so a genuine question about real game state ("are we actually level 2?") gets treated as a character action and wrapped in narrative prose — there was no way to just ask the DM directly. Built a lightweight, prompt-level solution rather than a separate pipeline/architecture:

- **Frontend** (`game.html`): a checkbox ("OOC (ask the DM directly)") next to the message input. On submit, `htmx:configRequest` prepends a fixed marker (`"[OOC] "`) to the outgoing message text; the checkbox auto-unchecks after sending so it can't silently stick to the next real in-fiction action. The player's own chat bubble shows a small "OOC" badge instead of the raw marker text, and gets a distinct dashed-border style (`.ooc-message`).
- **Mechanics prompt**: a message starting with `[OOC]` is answered as a direct, tool-grounded question (`get_character`/`get_party_status`/`search_rules`/`get_campaign_summary`, same standards as ever — never guess) rather than an in-fiction action; its resolution report is prefixed `[OOC]` too, signaling the narrator.
- **Narrator prompt**: on a `[OOC]`-prefixed resolution report, replies in plain DM-to-player voice — no second-person scene prose, no "what do you do?" — prefixed with 🛈, which the frontend detects (`rawText.trimStart().startsWith("🛈")`) to style the *entire* reply bubble distinctly (`.dm-message.ooc-message`), not just one line, unlike the roll-line/loot-line markers which only tag individual paragraphs.
- Verified live end-to-end (disposable test campaign): `[OOC] What level is Testerbot, and how many spell slots do they have right now?` → `🛈 Testerbot is a level 1 Wizard and currently has two 1st-level spell slots available.` — correct, tool-grounded, correctly marked.
- **Known gap, stated explicitly**: OOC turns are NOT excluded from session summarization or campaign-history search (`summarize_session`, `search_campaign_history`) — they still persist in the checkpoint like any other turn and could theoretically bleed into a session chronicle or be retrieved as "campaign history." Not addressed here; low real-world impact expected (an OOC turn is a factual Q&A, unlikely to read as a fabricated story event even if summarized), but worth a real fix if it ever causes a visible problem.

---

## Datastores

### Overview

| Store | Technology | Purpose |
|---|---|---|
| `CampaignStore` | PostgreSQL via SQLAlchemy Core | Campaign entities, dice roll log |
| Checkpoint store | PostgreSQL via `AsyncPostgresSaver` | LangGraph conversation memory per `thread_id` |
| `RulesStore` | Postgres/pgvector table `rule_chunks` | Rulebook embeddings for RAG, filtered by `books_in_play` |
| `HistoryStore` | Postgres/pgvector table `session_chronicle_chunks` | Session chronicle embeddings for RAG, filtered by `campaign_id` |
| `DraftStore` | In-memory dict (module-level singleton) | Character draft state during Session 0 |

One PostgreSQL instance holds campaign data, LangGraph checkpoints, AND (since the 2026-07-12 migration off ChromaDB) the rules/session-chronicle vector+keyword search tables — one DB for everything, matching this repo's own "no SQLite/Postgres divergence" principle (Tech Stack table above).

### Database schema (13 tables)

Semi-normalised: top-level entities get their own tables with flat queryable columns; nested data lives in `JSONB`. Every entity table has `campaign_id UUID FK ON DELETE CASCADE`.

Tables: `campaigns`, `characters`, `monsters`, `npcs`, `factions`, `quests`, `locations`, `containers`, `traps`, `handouts`, `sessions`, `encounters`, `rolls`

### Adventure groups and RAG scoping

Source documents are split into two tiers:

- `docs/source/core/` — always embedded with `source_type: "core"`, searched in every campaign
- `docs/source/adventures/{slug}/` — embedded with `source_type: "adventure"`, `adventure: "{slug}"`

Each adventure folder has `_meta.json`: `{"name": "...", "description": "...", "levels": "1-15", "recommended_players": "4-6"}`. `recommended_players` is a free-text range (e.g. `"3-7 (optimized for 4)"`) surfaced to the DM agent via `get_campaign_summary` / the campaign context block, alongside the current party count, so it can decide whether to offer a DM-controlled companion (see `companion.py`).

`campaign.books_in_play` stores the active adventure slugs. `RulesStore.search()` builds the equivalent predicate as real SQL now (was a ChromaDB `$or` filter dict before the 2026-07-12 migration):

```python
or_(t.rule_chunks.c.source_type == "core", t.rule_chunks.c.adventure.in_(books_in_play))
```

Core books are always included; adventure books are opt-in per campaign. Adventures can be added mid-campaign via `POST /campaigns/{id}/books`.

### Session memory — two-tier

**Tier 1 — Structured campaign state (Postgres):**
NPCs, quests, party, location, and session records are always current and injected into the system prompt. This is the "always relevant" memory.

**Tier 2 — Session chronicles (Postgres/pgvector `session_chronicle_chunks`, was ChromaDB `session_chronicles` before the 2026-07-12 migration):**
When a session ends, the full thread is summarized by the LLM into a narrative chronicle + key events list. The chronicle is embedded into `session_chronicle_chunks` with a `campaign_id` column. The `search_campaign_history` tool retrieves relevant past events on demand — the goblin fight from session 1 is never injected into session 10 unless someone asks about that road.

---

## Agent Architecture (`backend/agent/`)

### In-game DM agent (`dm_agent.py`) — two-model mechanics/narrator split

`get_agent()` builds a custom LangGraph `StateGraph` (not `create_react_agent`) with three nodes:

```python
class DMState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    mechanics_notes: str  # this turn's resolved outcome, handed to the narrator
    correction_note: str  # one-shot retry nudge text, set by whichever guardrail fired
    correction_count: int  # caps missing-followup/fake-tool-call guardrail retries per player turn
    narrator_correction_note: str  # Session 0 only — narrator's own self-check retry nudge
    narrator_correction_count: int  # separate budget from correction_count (starvation risk otherwise)
    tool_error_count: int  # caps retries for a bad/hallucinated tool call specifically (2026-07-04, see below)

def get_agent(campaign, store, rules_store, history_store):
    # mechanics: vllm_chat(temperature=0.1).bind_tools(tools) — was
    #   ChatOllama(settings.mechanics_model, temp=0.1) before the 2026-07-13
    #   vLLM-metal migration (see "Backend swap" below).
    #   Loops against the tools node until it returns a message with no tool_calls.
    #   Routes via Command(goto=...) rather than a conditional edge, since the
    #   no-tool-calls branch appends nothing to `messages` — a plain conditional
    #   edge would see stale state. That final message's text becomes
    #   `mechanics_notes` and is NEVER appended to `messages` (never shown to the
    #   player, never a "dm" transcript turn).
    # tools: reuses the existing per-campaign-locked ToolNode unchanged.
    # narrator: vllm_chat(temperature=0.8), no tools — same backend swap as mechanics.
    #   Sees narrative-only history (_narrative_messages — same predicate
    #   format_transcript uses) + mechanics_notes as a trailing directive.
    #   Its output is the only message ever appended as the turn's DM turn.
    ...

async def stream_response(campaign, store, rules_store, history_store, message, thread_id):
    # yields tokens via astream_events, filtered to
    # event["metadata"]["langgraph_node"] == "narrator" — the mechanics node's
    # tool-call JSON and reasoning never reach the SSE stream.
```

**Message trimmer:** `_make_mechanics_modifier` (mechanics — async, replaced the plain synchronous `_make_state_modifier` for this one node on 2026-07-03) prepends the system prompt, keeps only the last `_MAX_MESSAGES` raw (non-system) messages — 100, raised from the original single-model design's 30 once it became clear that figure was sized for a single-loop agent, not the two-node graph: every mechanics tool call burns 2 raw messages (the tool-call AIMessage + its ToolMessage), well within `gemma4:26b-mlx`'s 32k-token headroom — and, during an active encounter, live-fetches and appends a `[LIVE ENCOUNTER STATE]` block (`combat.py`'s `build_encounter_context()`) on **every** invocation, not once per turn, so it reflects tool calls made earlier in the same turn too. This replaced `get_active_encounter` as a callable tool entirely — correctness no longer depends on the model remembering to call it. `_make_narrator_modifier` (narrator) instead filters to narrative-only turns (last 20) plus the current turn's `mechanics_notes`, unaffected by this change.

**Intra-turn scratch purge:** once the mechanics loop ends (no more tool calls), `mechanics_node` returns `RemoveMessage`s for every tool-call `AIMessage`/`ToolMessage` since the player's last message — this scratch already did its job driving the turn's resolution and is never displayed, but left in place it would keep counting toward every future turn's context/KV cache. Only the player's `HumanMessage` and the narrator's own final `AIMessage` remain as the permanent record of the turn. This fixed *cross-turn* context bloat; the 2026-07-03 consolidated resolution tools (`resolve_attack`/`resolve_saving_throw`/`resolve_check`/`cast_spell`/`resolve_pending_action` in `resolution.py`, replacing several sequential `roll_dice`/`update_*_hp`/`advance_initiative` calls with one atomic call each) target the *intra-turn* cost instead — see "Deferred from the combat resolution refactor" below for what's still open.

**Reasoning-tag leak fix (2026-07-04):** every `ChatOllama` instance in this file (mechanics, narrator, and the shared `_get_model()` used by world-prep/party-fill/summarization) now explicitly passes `reasoning=False`, rather than leaving `langchain-ollama`'s default (`None`). Root cause, confirmed against `langchain_ollama`'s own source and Ollama's `gemma4:26b-mlx` model page: every non-E2B/E4B Gemma 4 variant unconditionally wraps output in a `<|channel>thought...<channel|>` block (empty or not) unless reasoning is explicitly disabled — with `reasoning=None`, any such tags land directly in `.content` instead of being split into `additional_kwargs`. This is almost certainly the actual mechanism behind the "literal garbled tokens" bug documented in the chargen.py Tools section (2026-07-03) — that fix (scratch purge) addressed a real, separate problem (unbounded context growth) but never explained *why* raw tags could leak into `.content` in the first place. None of the three model roles ever read `reasoning_content`, so `False` (skip reasoning entirely) was chosen over `True` (perform it, capture it separately) — no product value in paying for reasoning nothing uses. A defensive `strip_reasoning_leakage()` regex guard (belt-and-suspenders, not the primary fix) is also applied to narrator-facing text at both emission points — per-chunk (best-effort, not chunk-boundary-safe) in `stream_response`, and on the full buffered reply in `main.py`'s `session_zero_stream`. Verified live: a real streamed turn came back with zero leaked tags.

**Tool-call validation/repair harness (2026-07-04):** LangGraph's `ToolNode` already turns an unknown/hallucinated tool name and a Pydantic arg-validation failure into a corrective `ToolMessage` fed back to the mechanics node via the existing `tools -> mechanics` edge — verified directly against the installed `langgraph.prebuilt.tool_node` source rather than assumed. Two real gaps closed on top of that: (1) `_handle_any_tool_error` broadens `ToolNode`'s `handle_tool_errors` to catch *any* exception raised from inside a tool's own body — previously only Pydantic validation failures were caught, so a raw `ValueError`/`KeyError` from a tool that doesn't guard its own errors crashed the whole turn instead of giving the model a chance to self-correct; (2) a new `tool_error_count` field on `DMState`, checked via `_last_tool_batch_had_error` at the top of both `mechanics_node` and `chargen_mechanics_node`, bounds the mechanics↔tools retry loop at `_MAX_TOOL_ERROR_RETRIES` (2) — a separate budget from `correction_count`/`narrator_correction_count`, same starvation reasoning as those. Without this, a model stuck emitting bad tool calls had no backstop but `recursion_limit` (60) — up to 60 real LLM calls wasted on one stuck turn before failing with a generic, misleading "Lost connection to the model backend" error. Now it bails out after 2 retries with an in-character "had trouble resolving that, try rephrasing" message instead. Verified directly against a compiled test graph exercising the exact `_make_tool_node` configuration: a hallucinated tool name, malformed arguments, and a raised non-validation exception (`KeyError`) all now correctly degrade to a corrective `ToolMessage` rather than crashing, while a well-formed call still succeeds normally.

**Backend swap: Ollama → vLLM-metal, and the planned `conclude_turn`/forced-tool-calling control-flow change (2026-07-13, full plan: `vllm-migration-plan.md`).** Root cause underlying the entire guardrail chain above (`_detect_missing_followup`, `_detect_missing_combat_roll_followup`, `_detect_missing_loot_followup`, `_detect_missing_encounter_followup`, `_detect_stalled_non_player_turn_followup`, the Stage-2 lore guardrails): Ollama has no real `tool_choice`/forced-tool-calling mechanism at all — confirmed directly against the installed `langchain_ollama==1.1.0` source, whose own docstring says `tool_choice` "is currently ignored as it is not supported by Ollama." Every guardrail above exists to catch the model narrating an outcome without ever making the backing tool call, because Ollama genuinely cannot be forced not to do that. Mainline vLLM *does* implement real `tool_choice="required"` enforcement (structured-output/structural-tag constraints via `llguidance`), and a live spike (see the plan doc's §1) confirmed this empirically on real 5e-shaped tool schemas.

*Landed so far*: the chat backend itself is swapped — every `ChatOllama` construction in this file (`_get_model`, `_get_mechanics_model`, `_get_narrator_model`) now goes through `backend/llm.py`'s `vllm_chat()` instead of `ollama_chat()`, pointed at a `vllm-metal` server running `mlx-community/Qwen3-30B-A3B-4bit` (an MoE model, ~3B active params of 30B total — chosen over the plan's originally-spiked Gemma4 checkpoint on the strength of outside research suggesting better tool-calling quality; re-verified empirically before adopting it, not taken on faith — see the plan doc's "Step 0" section for the full re-verification: real `tool_choice="required"` compliance, 14/15 on a fresh battery of real tool schemas, throughput beating the original Gemma4 spike). This is a pure client/model swap — `mechanics_node`'s control flow, the guardrail chain, and `DMState` are all unchanged so far; nothing here yet takes advantage of real forced tool-calling. One new real finding from this swap, unrelated to tool-calling: Qwen3-30B-A3B-4bit reasons by default and leaks a `<think>...</think>` block straight into `.content`, the same class of problem the reasoning-tag leak fix above solved for Gemma4 on Ollama — fixed the same way, in principle (skip reasoning entirely rather than pay for content nothing reads), via a different mechanism: `vllm_chat()` sends `extra_body={"chat_template_kwargs": {"enable_thinking": False}}` on every call, with `--reasoning-parser qwen3` on the server as a defense-in-depth backstop.

*Planned, not yet implemented*: `tool_choice="required"` can't simply be bound onto every mechanics call as-is, because `mechanics_node`'s loop currently terminates precisely when the model responds with *no* tool call — that response's plain text becomes `mechanics_notes`. Force every call and the model could never produce that terminal response; the loop would never end. The fix (full design in the plan doc's §3.2): add a new tool, `conclude_turn(resolution_notes: str)`, whose "execution" is trivial (just carries the notes through, no state mutation) — with it always available, `tool_choice="required"` can bind on *every* mechanics call, since the model always has either a real mechanics tool or `conclude_turn` as a way to signal "done." `mechanics_node` checks for `conclude_turn` among `response.tool_calls` before the existing `if response.tool_calls: goto="tools"` branch, extracting `resolution_notes` in place of today's `_extract_text(response.content)` when present. This should make the "zero tool calls at all" branch of `_detect_missing_followup` (and the reason `_detect_missing_combat_roll_followup` exists as a not-gated-to-combat backstop) structurally impossible rather than just statistically rare — but per the plan's §5, none of the existing guardrail *functions* get deleted on this change alone, even ones whose trigger condition should now be unreachable; that has to be proven live first. Real risks flagged in the plan and not yet resolved: `conclude_turn` can still be called with fabricated `resolution_notes` (forcing guarantees *a* tool call happened, never that it was the semantically correct one — the existing content-based guardrail checks still matter, just re-pointed at `conclude_turn`'s argument instead of raw content); and universal forcing could surface a new failure mode (the model spamming a real tool needlessly, e.g. `roll_dice`, on a turn with no game-mechanics need, instead of just calling `conclude_turn`) that has no precedent under the old architecture and needs real live-play observation, not assumption.

**Potential future direction, not committed (2026-07-13): dedicated CUDA GPU host.** Tonight's vLLM-metal work surfaced a real, measured limitation of Apple Silicon for this workload — aggregate generation throughput stayed roughly flat (~20-25 tok/s) whether serving 1 or 7-8 concurrent requests, because everything shares one GPU's fixed compute pool with no dedicated-VRAM headroom the way a discrete GPU has; vLLM's continuous-batching design assumes the latter. A 5060 Ti (16GB dedicated VRAM) plus a Linux host is a real candidate for offloading the chat/mechanics model (and possibly the bulk-ingest contextualization pass, see "Prep Scripts" below) later, on the strength of that measured gap — not because anything here is broken, just because the current hardware's ceiling was found empirically, not assumed. Real considerations if this gets picked up, none verified live yet:
- `mlx-community/*` checkpoints are Apple/MLX-format only — a CUDA host needs a different quantized checkpoint (AWQ/GPTQ/GGUF), and mainline vLLM's CUDA backend directly (no `vllm-metal` plugin involved at all, a much more mature code path than what tonight fought with).
- `Qwen3-30B-A3B` is MoE — every expert must stay resident in VRAM regardless of the ~3B active per token, so the real footprint is set by the full ~30.5B params, not the active count. At 4-bit (~15.25GB) that's too tight against 16GB once KV cache + any co-resident model are counted; 3-bit (~11.4GB, already confirmed tonight to have equivalent tool-calling compliance to 4-bit) leaves real headroom instead.
- Running the tiny embedding model on CPU while the chat model uses the GPU is a plausible way to avoid VRAM contention entirely (CPU and GPU are separate resource pools) — mainline vLLM does have a genuine CPU backend (confirmed indirectly tonight: a non-CUDA `pip install vllm` resolved to a `+cpu`-tagged build), and an embedding call is one forward pass per input on a ~0.6B model, a far lighter CPU workload than autoregressive chat generation would be. The exact CLI mechanics for pinning one vLLM instance to CPU while another uses the GPU on the same machine haven't been tested.
- Same portability path already used for native-desktop ingestion applies here too: point `DATABASE_URL` at the canonical Postgres over the network (already exposed on port 5432 per `docker-compose.yml`), no Docker needed on the GPU host, same `chunk_id`-based upsert semantics.

**`thread_id`** is `{campaign_id}:{uuid}` — generated in the game `GET` route, stored in browser `sessionStorage`, passed as a query param to `/stream`. Stable across refreshes within the same tab.

**Out of scope:** the automatic world-prep pass (`get_world_prep_agent`) and one-shot party-fill (`get_party_fill_agent`/`run_fill_party`) stay on the original single-model `create_react_agent(_get_model())` — both are structured/tool-driven passes with no dedicated narration step. Session 0 chargen used to be in this list too — see "Session 0 agent" below for why it moved off it (2026-07-04).

### Session 0 agent

`get_session_zero_agent()` moved off the single-model `create_react_agent` pattern to its own two-node mechanics/narrator `StateGraph` (2026-07-04) — same split and same reason as `get_agent()` above (see the chargen.py Tools section's "Verified live..." narrative: the combined single-role agent was caught narrating fake tool calls and declaring character creation "successful" while `DraftStore` stayed empty, and switching only the underlying model didn't fully close that gap). Only the narrator's output is ever appended to `messages` as the turn's reply.

```python
def get_session_zero_agent(campaign, player_slug, store, rules_store, ds):
    # tools: dice + rules + chargen + companion (excludes in-game tools —
    #   combat, NPC management, quests — the job here is character creation,
    #   not running the game)
    # chargen_mechanics_node: same tool-calling loop shape as mechanics_node,
    #   plus the two Session 0-specific guardrails below and the same
    #   tool_error_count bounded retry (see "Tool-call validation/repair harness")
    # chargen_narrator_node: UNIQUE to Session 0 — validates its OWN output
    #   before it ever reaches the player (see below), using a separate
    #   narrator_correction_count budget from the mechanics-side one
    # thread_id: {campaign_id}:chargen:{player_slug}:{uuid}
```

Session 0-specific guardrails:
- `_detect_fake_tool_call` (`_FAKE_TOOL_CALL_RE`) — catches the mechanics node narrating a fenced ` ```json` block that looks like a tool call but isn't one (the original qwen2.5:14b-era failure mode; still checked regardless of which model is currently standardized on — `gemma4:26b-mlx` at the time this guardrail was written, `mlx-community/Qwen3-30B-A3B-4bit` via vLLM-metal as of the 2026-07-13 migration).
- `_detect_invented_spells` — checked on the mechanics node's notes AND (uniquely) re-checked on the narrator's own output in `chargen_narrator_node` before it streams, catching a spell name not on the character's class's real menu (`SPELL_MENUS`). The narrator-side check works because `main.py`'s `session_zero_stream` buffers narrator tokens per-invocation and only forwards the invocation that reaches `END`, so a self-caught retry here never leaks a discarded first draft to the player — the deliberate tradeoff is that Session 0 replies arrive as one block instead of typing out live, unlike the main game's narrator.
- `chargen_mechanics_node` also appends any `list_options` tool output verbatim to its notes regardless of whether the model's own text quoted it, so the real menu is structurally present for the narrator rather than depending on the model having followed the "quote it verbatim" prompt instruction.

### Session summarization

```python
async def summarize_session(thread_id, campaign_name) -> tuple[str, list[str]]:
    # Loads messages from LangGraph checkpoint
    # Filters to HumanMessage + AIMessage (drops ToolMessage / tool-call-only AIMessages)
    # Sends structured prompt: ---CHRONICLE--- / ---KEY EVENTS--- format
    # Returns (narrative_summary, key_events_list)
```

Called by `POST /campaigns/{id}/session/end`. The chronicle is saved to `Campaign.sessions` and embedded in `HistoryStore` for future RAG retrieval.

### Transcript retrieval

```python
async def get_thread_messages(thread_id) -> list[BaseMessage]:
    # Reads from AsyncPostgresSaver checkpoint, returns raw LangChain message objects

def format_transcript(messages) -> list[dict]:
    # Filters to [{role: "player"|"dm", content: str}]
    # Drops tool calls and tool responses — shows only the narrative layer
```

Used by `GET /campaigns/{id}/sessions/{sid}` to render the session transcript page.

### Decision record (2026-07-09): this is a workflow orchestrating agents, not a multi-agent system — and that's deliberate

Worth stating precisely, since "agentic" gets used loosely: this app is not
one agent, but it's also not the LLM-orchestrated multi-agent system the
term sometimes implies. Precisely what exists:

**Five distinct agent constructors in `dm_agent.py`**, two architectural
shapes:

| Agent | Shape | Tools (scoped) |
|---|---|---|
| `get_agent()` — main gameplay | Custom 2-node `StateGraph` (mechanics → narrator) | Full 45+ tool set |
| `get_session_zero_agent()` — chargen | Same 2-node shape, different prompts | dice + rules + chargen + companion — no combat/quest/travel |
| `get_world_prep_agent()` — region seeding | Single-loop `create_react_agent` | `create_location`/`connect_locations` + rules search only |
| `get_npc_prep_agent()` — opening-scene NPCs/site detail | Single-loop `create_react_agent` | `create_npc` + `set_opening_location_detail` + rules search — no party/combat/quest/movement/travel |
| `get_party_fill_agent()` — DM companion generation | Single-loop `create_react_agent` | Character-generation tools only |

Each has its own prompt, its own scoped tool set (`search_rules` is shared
across most; `create_location`, `create_npc`, `resolve_attack` etc. are each
scoped to exactly one role), and its own execution context — genuinely
distinct agents, not one flat toolset.

**But nothing routes between them with a model.** Checked directly in
`main.py`: which agent gets built is decided by plain deterministic Python,
keyed on which HTTP route fired — `POST /campaigns` always calls
`get_world_prep_agent()` then `get_npc_prep_agent()` twice, in a hardcoded
sequence inside `run_world_prep()`. The session-zero stream route always
calls `get_session_zero_agent()`. The main game stream always calls
`get_agent()`. No model ever reasons about which agent should handle a
request — the game's own state machine (Session 0 vs. world-prep vs. live
play) already fully determines it, so there's nothing ambiguous left for an
LLM to resolve. And the mechanics→narrator split within `get_agent()` isn't
multi-agent orchestration either — it's one `StateGraph` with two
LLM-calling nodes connected by explicit `Command(goto=...)` control flow,
closer to Anthropic's **prompt-chaining/sequential-handoff** pattern than to
agent-to-agent delegation.

This maps onto Anthropic's own published distinction: **workflows**
(predefined code paths orchestrating LLM calls: routing, chaining,
orchestrator-workers) versus **agents** (a model dynamically directs its own
process, including the control flow itself). What's built here is a
workflow that orchestrates multiple genuinely-agentic sub-processes — each
node the workflow dispatches to is a real agent (an LLM in a loop, deciding
which tools to call and when, over multiple steps), but the *dispatch
itself* is deterministic. That's not a lesser version of "true multi-agent"
— per Anthropic's own guidance, it's the better choice whenever the routing
decision is already knowable ahead of time, which it is here: the game's
phase isn't ambiguous, so paying for a model to figure out what a
`if`-statement already knows would be pure latency and cost with no
corresponding benefit.

**Considered, deliberately rejected: true multi-agent orchestration for
turn-based gameplay.** Reasons, not just intuition:
- **State-consistency risk.** The whole architecture exists to guarantee
  HP/inventory/initiative mutate through exactly one validated path — the
  2026-06-30 bug audit's finding #3 ("parallel tool calls silently drop
  mutations") needed a per-campaign lock to fix even *within a single
  agent's own response*. Multiple independent agents, each with their own
  read of world state and their own authority to call
  `update_character_hp`, reintroduces that race at a worse scale — not just
  "did two tool calls in one response collide" but "did two *agents* act on
  mutually stale state."
- **No real decomposition hides in a turn.** "Attack the goblin" is
  roll → apply → narrate, tightly sequential — the mechanics/narrator split
  already captures the one genuine distinct-skill boundary (rules-
  correctness vs. prose quality). A third agent wouldn't decompose anything
  real, just add latency.
- **Combat only looks multi-agent-shaped.** Many combatants, each
  superficially "deciding" their own action, could look like a fit — but
  initiative order is strictly sequential and centrally arbitrated by rules,
  which is exactly why the mechanics node already resolves every
  non-player combatant's turn in one response rather than spinning up a
  per-monster agent. Monster AI needs speed and rule-consistency, not
  distinct personalities — one model resolving all of them sequentially
  gets both, faster than N agent calls would.
- **Latency is directly felt here**, unlike background work — a player is
  watching a spinner for a reply. Every extra orchestration round-trip taxes
  the exact thing that matters most for a live game.

**Considered, deliberately deferred (not rejected): true multi-agent
orchestration for world-prep.** This one's genuinely closer to justified,
worth recording the real argument on both sides rather than a flat no:
- **For:** world-prep already runs as a fire-and-forget background task
  (`asyncio.create_task`, never blocking a response — see the "world-prep
  freeze" incident above for how much care went into making that safe), so
  extra orchestration latency is nearly free — nobody's watching a spinner
  synchronously. And the fixed pipeline has *already* hit a real
  decomposition limit once: `run_world_prep()`'s own comment records that
  asking one agent to create a whole NPC roster *and* the opening location
  in one pass let the location call starve after the roster ran long,
  fixed by manually splitting into two separate calls. That's evidence a
  single fixed shape doesn't scale cleanly across adventures of very
  different size (Lost Mine of Phandelver's dozen-ish locations vs. Curse
  of Strahd's sprawling geography) — a model reading the adventure and
  deciding "this splits into 3 distinct regions, dispatch one sub-agent per
  region" is a genuine content-understanding judgment call a size-based
  heuristic couldn't make as well.
- **Against:** running region sub-agents in parallel creates real
  duplicate/conflict risk — `entity_resolution.py`'s fuzzy-match guard
  exists precisely to catch near-duplicate entities for a *single* agent's
  sequential creates; independent parallel sub-agents creating locations
  without seeing each other's in-flight work makes that collision more
  likely, not less. And world-prep is explicitly best-effort already (a
  failure just means the DM improvises — see the world-prep freeze
  writeup's UI-gate fix) — there's no demonstrated pain forcing this, only
  a plausible hypothesis.
- **The actual trigger for revisiting this, stated in advance so it isn't
  built speculatively:** play a genuinely sprawling adventure (Curse of
  Strahd, Storm King's Thunder) through world-prep and check whether the
  seeded content comes out measurably thinner or less coherent than it does
  for a small adventure like Lost Mine of Phandelver. If it does, that's a
  real, measured decomposition problem — same discipline as the rest of
  this project (measure, don't assume) — and that's when a chief-
  worldbuilder-plus-regional-subagents pattern would earn its added
  complexity. Not before.

---

## Session 0 — Campaign Pitch + Character Creation

### Flow

1. DM navigates to `/campaigns/{id}/session-zero` — sees current party roster and a "Start Character Creation" form
2. Enters a player name → redirected to `/campaigns/{id}/session-zero/{player-slug}`
3. Split-pane UI: DM chat on left, live character sheet preview on right
4. DM agent opens with a **campaign pitch** (setting, premise, adventure hooks drawn from `campaign.name`, `campaign.setting`, `campaign.books_in_play`, `campaign.notes`)
5. DM walks through character creation in order: concept → race → class → background → ability scores → skills → backstory → appearance → party ties
6. After each confirmed choice, agent calls `update_character_draft` → preview panel refreshes via fetch on each `done` SSE event
7. When complete, agent calls `finalize_character` → character added to `campaign.party`, draft cleared
8. Player can return to `/session-zero` to add another character
9. Once satisfied, the DM can click "Ask DM to add a member" on the lobby page — a one-shot request (`POST .../fill-party`) that checks party composition against the adventure's recommended size and, if short, generates ONE DM-controlled companion via `generate_companion_character` (see `companion.py`) — one click, one companion, so the DM can review each addition before asking for another. Only shown when the campaign has an adventure with a `recommended_players` value.

**Known issue to investigate:** observed once during testing (2026-07-01) — after a long multi-tool-call turn, the Session 0 model (`qwen2.5:14b`) lost track of a `finalize_character` call that had already succeeded, believed it had failed, and re-finalized under a different name — leaving two near-identical characters in the party (`finalize_character`'s duplicate-name guard only catches an exact name match, so the rename slipped past it). Likely a tool-call-tracking degradation over long turns rather than a bug in `finalize_character` itself. Not yet root-caused or fixed.

**TODO (found 2026-07-04):** hitting the browser Back button after `finalize_character` can briefly show a stale bfcache snapshot of the lobby page (from before the character existed) instead of the current party roster — the DB write itself is fine (confirmed: character re-appeared once the page re-synced), this is a client-side caching gap, not data loss. Session 0 has no `Cache-Control: no-store` on its routes and no `pageshow`/`event.persisted` handling to force a reload on bfcache restore, and finalize never does a `history.pushState`/redirect, so Back has nothing correct to return to. Fix: add `Cache-Control: no-store` to the session-zero routes in `backend/main.py` and/or a `pageshow` listener in `session_zero.html`/`session_zero_index.html` that reloads when `event.persisted` is true.

### Ability score methods

All three PHB methods offered:
- **Rolled**: 4d6 drop lowest × 6, via `roll_ability_scores` tool
- **Standard array**: 15/14/13/12/10/8, assigned in any order
- **Point buy**: 27 points, cap 15 before racial bonus

### Character sheet preview

Rendered client-side from the draft JSON (`GET /session-zero/{slug}/draft`) on each SSE `done` event. Shows:
- Identity block (name, race, class, background, alignment)
- 3×2 ability score grid with modifiers
- Skill proficiencies
- Personality traits / ideals / bonds / flaws
- Backstory excerpt (first 300 chars)

### Derived stats at finalization (`finalize_character`)

| Stat | Calculation |
|---|---|
| `max_hp` | Hit die (by class) + CON modifier, minimum 1 |
| `current_hp` | = `max_hp` |
| `ac` | 10 + DEX modifier (unarmored default; DM adjusts for armor) |
| `passive_perception` | 10 + WIS modifier + 2 if Perception proficient |
| `proficiency_bonus` | +2 at level 1 |
| `spell_slots` | From `STARTING_SPELL_SLOTS` map by class |
| `hit_dice_total` | `1d{hit_die}` by class |

Also captured but historically **silently discarded**: `personality_traits`/`ideals`/`bonds`/`flaws` were passed into `Character(...)` by `chargen.py` since Session 0 was built, but the `Character` model never actually defined those fields (only `NPC` did) — Pydantic's default `extra="ignore"` behavior swallowed them with no error. **Fixed 2026-07-02** by adding the fields to `Character` in `models.py`. Any character finalized before this fix has no personality data in its stored record; there's nothing to backfill, it was never persisted.

### Hardcoded 5e data (`backend/data/fivee_options.py`)

2024 PHB options for fast DM reference without RAG latency (see the 5.5E migration note further down) — `search_rules` remains the fallback for edge cases or optional sourcebook content.

| Category | Contents |
|---|---|
| Species | Aasimar, Dragonborn, Dwarf, Elf, Gnome, Goliath, Halfling, Human, Orc, Tiefling — no ability score bonuses (2024 rule; those come from Background) |
| Classes | All 12 classes with hit die, primary ability, saves, armor/weapon profs, level 1–3 features (subclass choice unified to level 3), playstyle description |
| Backgrounds | All 16 backgrounds with ability score triple, Origin feat, skills, tools, flavor |
| Score methods | Rolled, standard array, point buy (with full cost table) |

---

## Session Continuity & Narration

Added 2026-07-02, closing a real gap: Session 0 already gathered rich character detail (backstory, personality, and now appearance), but none of it ever reached the in-game DM — the main game's system prompt only ever showed name/race/class/level. Every session also opened cold: `game.html` just showed a static "The DM awaits your command" placeholder and did nothing until the player typed first, identically whether it was the party's very first session or their tenth.

**Character appearance** — `Character.appearance: str` (mirrors `NPC.physical_description`), captured as an explicit Session 0 step (`update_character_draft` field `"appearance"`) and by `generate_companion_character`. Shown in the live character sheet preview during Session 0.

**Curated flavor excerpts** — `_char_flavor_excerpt()` in `prompts.py` builds a short (~100 chars/field) excerpt of appearance + first personality trait + backstory per party member, injected into both the mechanics and narrator system prompts' party listing. Deliberately short and per-turn, not the full text — a "short curated excerpt," not a context-budget-eating full dump.

**Narrator vividness** — the narration-style instructions now explicitly call out that this app has no illustrations or generated art, so prose is the only visual the player gets; reach for concrete sensory detail rather than a flat one-liner. Combat is the stated exception (kept fast/kinetic) — the richer description is for exploration, arrivals, and quiet moments.

**Session kickoff** (`build_session_kickoff_message()` in `prompts.py`, `POST /campaigns/{id}/session/begin`) — a "Begin the Adventure" / "Continue the Adventure" button (label depends on `campaign.session_count`) replaces the old static placeholder in `game.html`, both on first load and after `closeOverlay()` following `/session/end`. Clicking it builds a server-side message and enqueues it exactly like a real player message, so it flows through the normal mechanics → narrator pipeline over the existing `/stream` SSE endpoint — no new streaming path needed. Two cases:
- **First-ever session** (`session_count == 0`): ask if the player's ready, then introduce the opening scene and every party member (PCs and companions) using their curated appearance/personality/backstory, ending on "what do you do?"
- **Later sessions**: recap the most recent chronicle (`campaign.sessions[-1].summary` + `key_events`, shown in full — this is a one-time insertion at a session boundary, not repeated every turn) before re-establishing current state and asking what they do next.

Verified end-to-end, both first-session and later-session paths:
- First session: seeded a PC with appearance/backstory, triggered the kickoff, and confirmed the narrator wove the seeded appearance directly into the opening prose, *and* the mechanics model autonomously noticed the party was short of Icewind Dale's recommended size and called `generate_companion_character` mid-turn — the resulting companion's own generated `appearance` also showed up correctly in the same narration. No tool-call leakage.
- Later session: seeded a `Session` chronicle (summary + key_events) and `session_count=1`, triggered the kickoff, and confirmed a clean "Previously on..." recap faithfully reflecting the seeded chronicle (no invented events), transitioning into a fresh vivid description of current state before asking what they do next.

Also fixed along the way, discovered by this testing:
- **`Character.personality_traits`/`ideals`/`bonds`/`flaws` were silently discarded** — `chargen.py` had been passing them into `Character(...)` since Session 0 was built, but the model never defined those fields (Pydantic's default `extra="ignore"` swallowed them with no error). Confirmed via a real character's stored DB record having no `personality_traits` key at all. Fixed in `models.py`; nothing to backfill for characters created before the fix, the data was never persisted.
- **`run_fill_party`'s summary text occasionally leaked a fenced ```json tool-call block** — observed once: the model's real `generate_companion_character` call succeeded (with no `appearance` set), then its closing prose *narrated* what looks like a corrected second call, complete with a nicer appearance description, but never actually executed it — just typed it out. Fixed by truncating the returned summary at the first fenced code block, so the DM-facing UI never shows raw tool-call JSON. Re-verified clean afterward.

**Context-length warning** — `get_context_status()` in `dm_agent.py` compares a thread's raw message count against `_MAX_MESSAGES` (the mechanics trim window); `GET /campaigns/{id}/thread-info` exposes it. `game.html` polls this after every `done` SSE event and shows a sidebar banner near "End Session" once within 5 messages of the trim limit. Deliberately a UI-level check computed in Python, not left for the model to notice and mention in-character — unreliable by nature, and this app already has a concrete example of an LLM not reliably tracking a fact it should have (the fill-party Cleric-anchoring bug, see `companion.py` above).

---

## Evolution: From Naive RAG to a Hybrid Retrieval Pipeline

This section exists because the re-architecture described below happened in
one large commit and was never written up anywhere — the code changed, this
doc didn't. Written retroactively (2026-07-08) as a record of what the first
version got right/wrong and what specifically replaced it, for anyone
(including future-me) who wants the reasoning, not just the current state.

**Superseded note (2026-07-12):** every ChromaDB reference below (`Chroma.similarity_search`,
the `rules`/`session_chronicles` collections, `data/chroma_db/`, the BM25 pickle sidecar
file) describes the architecture as it existed through 2026-07-12 — it is **not current**.
ChromaDB was removed entirely and replaced with Postgres/pgvector (`rule_chunks`/
`session_chronicle_chunks` tables, native `tsvector`/GIN full-text search instead of the
BM25 pickle) — see the Tech Stack table and "Datastores" section above for the current
architecture. The *design decisions* below (parent/child chunking, contextual retrieval,
Reciprocal Rank Fusion, the CRAG-style grading/reformulation loop) are all unchanged and
still accurate — only the storage engine underneath them changed. Similarly, every
`gemma4:26b-mlx`/Ollama reference below describing the in-game chat model is superseded
by the 2026-07-13 vLLM-metal migration (see "Agent Architecture" above) — left as
historical record of the reasoning at the time, not current state.

### Phase 0 — the naive baseline (through 2026-07-05)

The first working version's RAG was about as simple as it gets:
`RulesStore.search()` was one call — `Chroma.similarity_search(query, k=4,
filter=where)` — plain dense-vector top-k, nothing else. Ingestion
(`build_index.py`) chunked each `.md` file on `##`/`###` headers, capped at
1500 chars, embedded with `nomic-embed-text`, done. No BM25, no reranking,
no contextualization, no parent/child structure, no entity extraction, no
relationship graph. It was enough to stand up Session 0, combat, and
exploration end-to-end — the point of a first pass — but it inherited every
well-known weakness of pure dense retrieval: exact-name lookups (a spell,
a monster, a specific magic item) compete on semantic similarity against
paraphrases instead of just matching the term, and nothing caught a chunk
that read fine in isolation but lost its subject once split out of its
section ("she agreed" — who?).

It's worth noting the *agent* side of the project was already further along
than the *retrieval* side at this point — the two-model mechanics/narrator
split (`dm_agent.py`) and tool-level guardrails already existed before any of
the RAG work below, built on the same underlying instinct (don't trust one
LLM pass to get retrieval-grounded correctness and free-form prose right in
the same shot) that the RAG re-architecture later applied to search itself.
The lesson landed in one part of the codebase before it generalized to
another.

### The self-review that forced a stop (2026-06-30 bug audit)

Before any RAG redesign, an internal review of `backend/` surfaced 10 bugs
(`docs/engineering-notes/2026-06-30-bug-audit.md`) — general correctness
issues, not RAG-specific, but the very first one was foundational: **the
rules RAG pipeline was silently dead**. `RulesStore` was constructed but
`.load()` was never called — the only method that actually opens the Chroma
collection — so `is_ready()` was always `False` and `search_rules` returned
"index not ready" on *every single call*, regardless of whether indexing had
even run. The entire grounding story the README leads with ("RAG-grounded
rules, not invented ones") had never actually worked in a live session until
this was fixed. That's the kind of finding that justifies stepping back
instead of patching forward — if the foundational plumbing was broken, no
amount of tuning the retrieval algorithm on top of it would have mattered.

### The re-architecture (`b7d6c73`, 2026-07-07) — pulling in named industry techniques

One large commit (49 files, +5378/-479) rebuilt retrieval from that naive
baseline into a multi-stage pipeline. The code itself groups the change into
stages (see docstrings in `reranker.py`/`grading.py`/`lore_store.py`), which
is a useful way to read it:

**Stage 0 — ingest & retrieval core** (`backend/rag/hybrid.py`,
`contextualizer.py`, `reranker.py`; `backend/stores/rules_store.py`):

- **Parent/child chunking.** Each `##`/`###` section becomes one PARENT
  (≤1500 chars, oversized ones split with 50-word overlap between parts —
  fixing a real bug in the old size-split, which had *zero* overlap and lost
  continuity right at the seam), further split into ~350-char CHILDREN. Only
  children are embedded/BM25-indexed; a retrieved child expands back to its
  full parent via `parent_chunk_id` before reaching the agent — small chunks
  for precise matching, full sections for actual context.
- **Hybrid search via Reciprocal Rank Fusion.** Dense (Chroma) and sparse
  (`rank_bm25.BM25Okapi`, built from the same stored text, pickled to
  `data/bm25_rules.pkl`) each return a top-`wide_k` (30) ranked list;
  `reciprocal_rank_fusion()` merges them by the standard RRF formula
  (`score = Σ 1/(k + rank + 1)`, `k=60`) — no learned weighting, no tuning
  knob, just the well-known formula. This is the direct fix for the
  naive baseline's exact-name-lookup weakness: BM25 catches "Aboleth" as a
  literal token match even when the dense embedding ranks something
  semantically-similar-but-wrong higher.
- **Contextual Retrieval** (the module docstring names it explicitly as
  Anthropic's published technique): before embedding, each child chunk gets
  a one-sentence LLM-generated blurb situating it ("who/what/where, using
  proper names") prepended — but **only to the text that gets embedded**;
  the stored/citable text stays the raw, unmodified original. This is the
  fix for the "she agreed — who?" problem. Uses the main 26B model rather
  than a cheaper one, on the strength of an earlier documented reliability
  incident with a smaller model (`qwen2.5:14b`) producing garbled/fake
  tool-call output under sustained use — not worth reintroducing that risk
  to save a few seconds per chunk.
- **Reranking**, over the RRF-fused 30 candidates down to the caller-facing
  `k=6`. Two implementations exist behind a `Reranker` protocol —
  **LLM-as-judge is the default** (batched Ollama call, ranks candidates by
  relevance), with a cross-encoder (`sentence-transformers`
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) implemented as an opt-in
  alternative but explicitly *not* wired in by default. Why: the module's
  own docstring records that loading the cross-encoder OOM-killed under
  Docker Desktop's constrained memory allocation — **the same
  small-Docker-VM failure mode documented below in the 2026-07-07/08
  ingestion incidents, just discovered earlier and in a different subsystem.**
  This project has now hit that ceiling twice in two different parts of the
  pipeline before actually fixing the ceiling itself.

**Stage 1/1.5 — canon lore, once per book** (`scripts/extract_entities.py`,
`backend/stores/lore_store.py`, `backend/stores/graph_store.py`,
`backend/rag/entity_resolution.py`):

- `extract_entities.py` exists because of a measured, specific failure: a
  single live-agent pass asked to both *discover* and *profile* every named
  entity in a chapter in one shot missed 4 of 5 expected NPCs on Curse of
  Strahd — everyone whose only evidence was scattered mentions rather than
  one concentrated scene. The fix, same principle as the two-model agent
  split again: separate the jobs. A five-stage offline pipeline (windowed
  discovery → canonicalize/alias-merge → reference collation → type-specific
  profile generation → checkpointed JSON write) runs once per book, not live
  per campaign, and writes into `LoreStore` (Postgres), a canon registry the
  live agent reads from but never mutates.
- `graph_store.py` adds a lightweight relationship graph on top —
  self-described in its own docstring as **LightRAG-style, set-merging,
  campaign-scoped** (NPC↔faction, NPC↔location, item↔location). No full
  rebuilds: a Postgres unique constraint on `(campaign_id, source_id,
  target_id, relation)` with `ON CONFLICT DO NOTHING` *is* the merge/dedup
  mechanism, and `networkx` graphs are rebuilt fresh from Postgres on demand
  — cheap at the actual scale involved (tens to low-hundreds of edges per
  campaign).
- `entity_resolution.py` is the live-play guard that keeps campaign-specific
  entities from silently duplicating canon — RapidFuzz's `WRatio` (chosen
  over `token_sort_ratio` after checking real examples: "Toblen" vs. "Toblen
  Stonehill" scored ~55 on the token-sort metric, too low to flag, vs. ~90 on
  `WRatio`) flags likely duplicates at insert time; it never auto-merges,
  only warns, leaving the actual merge decision to the calling agent.

**Stage 2 — query-time self-correction** (`backend/rag/grading.py`,
wired into `search_lore` in `backend/tools/lore.py`):

- A CRAG/Self-RAG-style pattern: retrieve → an LLM grades whether the
  top-5 results plausibly answer the query → if not, reformulate the query
  once and re-retrieve at a wider `wide_k=50` → re-grade. Exactly **one**
  bounded retry, matching this codebase's established no-unbounded-retry
  discipline (the same pattern as the agent's `tool_error_count`/
  `correction_count` retry caps). If still insufficient after the retry, the
  tool doesn't silently drop the results or fail — it returns what it found
  with an explicit disclaimer appended (`"[Note: retrieval may be incomplete
  for this query — consider saying so rather than filling gaps with
  invention.]"`), i.e. abstention-signaling over false confidence.

### What didn't need to change

The campaign-scoping logic (`books_in_play` → a `$or` filter of
`source_type: core` OR `adventure: {$in: books_in_play}`) predates this
re-architecture and was left untouched — it was already correct, and the
new hybrid pipeline just inherited the same `where` filter on both its dense
and BM25 legs. Not every naive-v1 decision was wrong; the re-architecture
targeted retrieval quality specifically, not the scoping model around it.

### Measured, not assumed (2026-07-08): running `eval_retrieval.py` for real

First pass, 17 hand-labeled questions: baseline (plain dense) 64.7%, hybrid
58.8% — hybrid nominally worse. Investigating *why* surfaced a real
methodology bug in the eval itself, not just a quirk of small-n: **5 of the
17 questions targeted books (*Xanathar's Guide to Everything*, *Volo's Guide
to Monsters*) that are indexed under the pre-Stage-0 schema — no
`granularity`/`chunk_id`/`parent_chunk_id` metadata at all**, because they
were ingested before the RAG re-architecture and never migrated. That
metadata gap isn't cosmetic: `RulesStore.search()` hard-filters on
`granularity: {"$eq": "child"}` (`rules_store.py:162`) for *both* its dense
and BM25 legs, so books without that field are **structurally invisible to
the hybrid pipeline** — not deprioritized, not scored lower, literally never
in the candidate set. Confirmed directly: `docs/source/adventures/` has 10
adventures, and only Lost Mine of Phandelver has real parent-granularity
chunks; the other 9 are in the same un-migrated state as Volo's/Xanathar's.
This is a known, already-documented gap (see the docstring on
`RulesStore.is_ready()`) — a full `make reindex-full` across the whole
library, not yet done, is the actual fix — but it means the original
eval's baseline-vs-hybrid comparison on those 5 questions wasn't measuring
retrieval quality at all, just "does this book happen to still be
findable by unfiltered dense search." Baseline "won" those by accident, not
by being smarter, and hybrid "lost" them by construction, not by being
worse — which is the precise mechanism behind the intuition that prompted
rechecking this in the first place.

**Fix: rebuilt `retrieval_questions.json` from 17 to 62 questions, scoped
exclusively to the four fully Stage-0-migrated books** (PHB, Monster Manual,
DMG, Lost Mine of Phandelver) — every `expected_book`/`expected_section`
pulled from a real `# `-header confirmed present in the actual indexed
markdown, not guessed. The 4 class questions and the Hill Giants question
were retargeted to their real current home (PHB, Monster Manual) instead of
dropped. Re-ran both modes on the clean set:

| Mode | Recall@6 |
|---|---|
| `--baseline` (plain dense `similarity_search`) | **58/62 = 93.5%** |
| current hybrid pipeline (RRF + LLM rerank + parent expansion) | **55/62 = 88.7%** |

With the confound removed and the sample nearly 4x larger, hybrid *still*
trails baseline — by enough now (4.8 points, on a clean same-corpus
comparison) to treat as signal, not noise. Category breakdown:

| Book | Baseline | Hybrid |
|---|---|---|
| Monster Manual (16 qs) | 16/16 = 100% | 16/16 = 100% |
| DMG (17 qs) | 16/17 = 94.1% | 15/17 = 88.2% |
| PHB (13 qs) | 11/13 = 84.6% | 8/13 = 61.5% |
| Lost Mine of Phandelver (9 qs) | 8/9 = 88.9% | **9/9 = 100%** |

Lost Mine of Phandelver is hybrid's clean win (BM25 caught "Emerald Enclave"
where dense search alone missed it — the exact class of fix hybrid search
was built for). **PHB's 11-of-13-to-8-of-13 drop is the whole story**, and
it clusters almost entirely on one query template: "What is the `{class}`
class like?" Of 11 PHB classes tested, hybrid missed 5 (Barbarian, Bard,
Fighter, Ranger, Wizard) that baseline got right.

Traced the actual mechanism with a direct diagnostic (Fighter, which failed,
vs. Cleric, which passed) rather than guessing:

```
Fighter — hybrid's final results (only 2, not 6):
  PLAYER'S HANDBOOK | FIGHTER (2)
  PLAYER'S HANDBOOK | FIGHTER (1)
Cleric — hybrid's final results (4):
  PLAYER'S HANDBOOK | CLERIC (2)
  PLAYER'S HANDBOOK | CLERIC (1)
  PLAYER'S HANDBOOK | CLERIC CLASS FEATURES   <- the one the eval wants
  PLAYER'S HANDBOOK | LEVEL 1: SPELLCASTING (3)
```

Each class has two competing sections: a short flavor-text intro titled just
`"{CLASS}"`, and the actually-substantive `"{CLASS} CLASS FEATURES"` section
the eval questions target. Two things compound against the latter for
Fighter specifically: (1) hybrid's post-rerank, post-parent-expansion result
set can come back **shorter than the requested k** once duplicate parents
collapse (2 final results for Fighter vs. 6 requested) — a much narrower
funnel than baseline's raw, undeduplicated top-6; (2) BM25's exact-token
scoring rewards a *short* section whose entire heading is just the matched
word ("FIGHTER") at least as strongly as a longer, differently-titled
section that merely contains it ("FIGHTER CLASS FEATURES") — so the flavor
text can crowd out the more useful content within that narrower funnel.
Baseline's plain dense search doesn't have either problem: no BM25 exact-
token bias, and no post-expansion dedup shrinking its result count. This is
a real, reproducible weakness in the current hybrid pipeline for
short-title-vs-long-title section pairs, not an artifact and not something
the LLM reranker is likely responsible for (it operates on whatever survived
RRF fusion — the fusion/dedup stage is what's actually narrowing the field).
**Concrete follow-up, not yet done:** widen `wide_k` and/or fix
`_expand_to_parents()` to backfill toward the requested `k` when
deduplication shrinks the candidate set below it, so classes like Fighter
get the same headroom Cleric happened to get.

**Answering the original question directly:** no, this isn't the naive
baseline "knowing" something via hallucination — `eval_retrieval.py`
measures pure retrieval (which indexed chunk came back), no generation
happens in the eval at all. What actually happened was two unrelated
effects layered together: an eval-methodology bug (comparing against
un-migrated, structurally-invisible-to-hybrid books) that made hybrid look
worse than it is, *and*, once that was fixed, a real, narrower, reproducible
hybrid weakness on one specific query shape that makes it genuinely worse
in that case — both true at once, and only separable by actually tracing
individual queries rather than trusting the aggregate number either time.

### The fix (2026-07-08): backfill past the dedup collapse, retrieval-time only

Implemented the "concrete follow-up" above: `RulesStore.search()` now
reranks the **full** candidate set (`top_n=len(candidates)`, not `top_n=k`)
— free, since both `Reranker` implementations already compute the whole
ordering internally before slicing — and hands that fuller order to
`_expand_to_parents(reranked, k)`, which now walks it and stops once **k
distinct parents** are collected, instead of truncating to k children and
deduping after (where dedup collapse could silently shrink the result count
below k with no way to recover). Purely a query-time fix — `search()` and
`_expand_to_parents()` are the only two touched, both in
`backend/stores/rules_store.py`. No change to chunking, embedding,
contextualization, or BM25 indexing, so nothing needed re-running against
the already-built index — `make reindex-full` was not required, and the fix
took effect on the next `RulesStore.search()` call.

Verified directly: re-ran the Fighter query that motivated the fix.
Before: 2 results (`FIGHTER (2)`, `FIGHTER (1)` — the flavor-text intro,
twice, no `FIGHTER CLASS FEATURES`). After: 6 results, with
`FIGHTER CLASS FEATURES` restored at position 3. Re-ran the full 62-question
suite:

| Mode | Recall@6 |
|---|---|
| `--baseline` (plain dense) | 58/62 = 93.5% |
| hybrid, pre-fix | 55/62 = 88.7% |
| **hybrid, post-fix** | **60/62 = 96.8%** |

Hybrid now beats baseline outright — 5 of the 6 pre-fix regressions
(Barbarian, Bard, Cleric already passed, Druid already passed, Fighter,
Ranger — all now correct) are fixed; only "What is the Wizard class like?"
still misses, and "What are a villain's methods?" remains a miss in *both*
modes (not a hybrid-specific issue — likely a genuinely weak semantic/lexical
match for that particular phrasing against the source section, a separate
question from anything fixed here). This is the first point in tonight's
work where hybrid search has an actual measured, apples-to-apples advantage
over the naive baseline on a same-corpus, non-trivial (n=62) benchmark —
worth having, given the whole point of building Stage 0 was that it should
outperform plain dense search, not just cost more per query.

### Chasing the last miss (2026-07-08): two more bugs, one retrieval-time, one needing a cheap rebuild

Asked "why does Wizard still miss" rather than accepting 96.8% as good
enough. Found two separate, unrelated bugs stacked on top of each other:

**Bug 1 — OCR drop-cap artifacts.** The PHB's Wizard section opens with a
decorative oversized first letter ("**W**IZARDS ARE DEFINED BY THEIR
exhaustive study of magic..."), and the OCR/markdown pipeline parsed that
lone "W" as its own one-character section header — splitting the real
opening paragraph into a bogus `"# W"` section. `grep -c '^# [A-Z]$'`
against the indexed markdown found **3 in PHB, 26 in Monster Manual, 0 in
DMG** — not a one-off. A short, keyword-dense bogus section like this can
out-rank a real, differently-titled section on BM25's length-normalized
scoring. The real fix belongs upstream (the OCR/header-detection step,
requiring a reindex); shipped the cheap retrieval-time mitigation instead —
`RulesStore.search()` now drops any single-letter-section candidate
(`_is_drop_cap_artifact()`) before reranking, covering all 29 known
instances without touching the index.

**Bug 2 — no punctuation stripping or stopword filtering in the BM25
tokenizer, and it was the bigger one.** `BM25Index`'s tokenizer
(`backend/rag/hybrid.py`) was a bare `text.lower().split()`. Diagnosed by
inspecting BM25's raw top-30 for "What is the Wizard class like?" directly
— it came back **completely unrelated to Wizard**: sections like
"CREATING A RACE OR SUBRACE," "SETTLEMENTS," "INVOLVING THE CHARACTERS."
Root cause: that query tokenizes to `['what','is','the','wizard','class',
'like?']` — five near-universal filler tokens (one, `"like?"`, couldn't
match anything at all, since the corpus's own equally-naive tokenizer would
never produce a token with a trailing `?`) diluting the one token that
actually mattered. BM25Okapi sums a score per query token with no concept
of which ones carry signal, so "class" alone — common across every
`"{CLASS} CLASS FEATURES"` section *and* loads of unrelated prose — was
enough to outrank the real match, which never appeared anywhere in BM25's
top 30. Fix: a shared `_tokenize()` (regex word-extraction + a small
stopword list) applied identically to corpus text at build time and queries
at search time — the two absolutely must stay in lockstep, or scores stop
being comparable at all. Unlike the drop-cap fix, this one **isn't**
purely retrieval-time — the corpus side of the tokenization changed, so
`data/bm25_rules.pkl` needed rebuilding. Ran `_rebuild_bm25()` directly
(bypassing the full `build_index.py` CLI) — pure CPU, reads already-indexed
Chroma text, no LLM/embedding calls — done in well under a minute for
227,776 child chunks. No `make reindex-full` needed; embeddings and Chroma
itself were untouched.

Verified directly: BM25 alone now ranks "WIZARD CLASS FEATURES" #1 for that
query (was absent from the top 30). Re-ran the full 62-question suite:

| Mode | Recall@6 |
|---|---|
| `--baseline` (plain dense) | 58/62 = 93.5% |
| hybrid, after backfill fix only | 60/62 = 96.8% |
| **hybrid, after drop-cap + tokenizer fixes** | **61/62 = 98.4%** |

Wizard now passes. The one remaining miss ("What are a villain's methods?")
fails identically in baseline, unrelated to anything fixed tonight. Three
fixes, three different root causes, three different remediation costs
(retrieval-logic-only, retrieval-logic-only, cheap-CPU-only rebuild) — worth
keeping straight, since "fixed the reranker" would have been a wrong and
much vaguer description of any of them.

### Tonight's chapter: the pipeline was sound, its ingestion path wasn't yet proven at real scale (2026-07-07/08)

The Stage 0–2 techniques above had only been exercised against adventure
books (thousands to tens of thousands of chunks). Running the same pipeline
against the full core rulebooks (PHB: 110k+ chunks) surfaced two separate
reliability gaps that smaller runs never hit — a Docker Desktop VM memory
ceiling that OOM-killed the indexing process partway through (root-caused
and fixed by running natively instead, see the "Prep Scripts" incident
write-up below), and an unbounded, non-batched LLM call in
`extract_entities.py`'s `canonicalize()` step that stalled for 5+ hours
against DMG's unusually large candidate list with zero progress visibility
(fixed by batching, same write-up). Neither was a flaw in the retrieval
design itself — both were ingestion-pipeline robustness gaps that only
showed up at a scale the pipeline hadn't been proven against before.

### The world-prep freeze (2026-07-08): three plausible fixes, then the real one

Separate incident, same night, different subsystem — the live app itself,
not retrieval. Clicking "create campaign" (with an adventure selected) or
navigating into Session 0 started **freezing the entire app for every
user**, repeatedly, confirmed each time by a plain `GET /` timing out
completely. Three attempted fixes in sequence, each real and necessary but
each insufficient on its own — worth recording all three, not just the one
that worked, since "the fix didn't fully work" was itself the signal that
led to the next, better fix:

1. **Found three unwrapped synchronous calls** in `world_prep.py` and
   `dm_agent.py`'s `summarize_session()` — plain `def` methods on
   `RulesStore` (blocking Ollama embed/rerank calls) invoked directly from
   `async def` functions with no `await`/`asyncio.to_thread`, on a
   single-worker `uvicorn --reload` process with exactly one event loop.
   Same bug *class* as a 2026-06-30 audit finding (`add_session` blocking
   the loop the same way) that was fixed in one place but never swept
   elsewhere. Wrapped all three in `asyncio.to_thread`. **Necessary, not
   sufficient** — froze again on the next campaign.
2. **Bounded those with `asyncio.wait_for(..., timeout=30.0)`.** Confirmed
   live that the identical query, run directly/natively/single-threaded,
   returned in under a second — yet the app-embedded call still froze
   everything, which was itself a clue that per-call cancellation wasn't
   reaching whatever was actually stuck. Root cause at the time: cancelling
   an `asyncio.to_thread` wait does **not** stop the underlying OS thread —
   Python cannot forcibly terminate a running thread — so a genuinely stuck
   call leaks one permanent slot from the shared default `ThreadPoolExecutor`
   every time it happens. **Necessary, not sufficient** — froze again.
3. **Added real `httpx` client timeouts** (`client_kwargs={"timeout": ...}`,
   60s embeddings / 120s chat) to every live-app Ollama client construction
   site (`rules_store.py`, `history_store.py`, `dm_agent.py` ×3,
   `reranker.py`, `grading.py`) — verified directly that the timeout config
   really does reach the underlying httpx client
   (`c._client.timeout == Timeout(timeout=60.0)`, confirmed by inspection).
   **Still froze again**, well past both thresholds — the most surprising
   result of the night, since this should have been airtight.

At that point the pattern across every freeze became the real clue: `ollama
ps` always showed `nomic-embed-text` loaded normally (healthy, not stuck) —
the seed queries always succeeded — and the freeze always landed immediately
*after*, exactly where the code needed to switch to `gemma4:26b-mlx` for the
actual agent. That's a **model swap** (Ollama evicting one model to load
another), and this codebase already had a documented prior incident with
that *exact* signature: `_get_mechanics_model()`'s own docstring in
`dm_agent.py` notes a previously-chased hang where "a request landing
mid-idle-eviction seemed to leave the MLX runner stuck reporting
'Stopping...' indefinitely." Not a new bug — a known MLX-engine rough edge,
now recurring reliably because world-prep's own workflow (embed → chat,
every single run) manufactured that exact transition on every campaign
creation.

**The actual fix: stop causing the swap, rather than better-surviving it.**
World-prep's seed-query step (`_SEED_QUERIES`, three embedding searches for
generic context like "regional overview" / "travel distances") got replaced
with `_gather_seed_context()` — a fully deterministic, zero-Ollama-call
function: a plain read of the adventure's own introduction section (the
text before its curated `opening_section_marker` — confirmed against Lost
Mine of Phandelver that this is exactly where "Background"/"Overview"/
"Adventure Hook" content already lives) plus `LocationExtractor`'s own
pre-computed `_connections` data (already-grounded travel routes,
extracted once at ingest time — *better* grounded than a fresh live search
would be, since real source citations went into producing it, and it's the
literal "travel distances between locations" content the old query was
trying to find). After this change, every remaining Ollama call in
world-prep's whole pipeline is `gemma4:26b-mlx` — no embedding model is ever
touched, so there's no swap to trigger the bug at all.

Two follow-on fixes, found while validating: (a) `world_prep_error` was
silently ending up empty on a real failure — `httpx.TimeoutException`
subclasses carry no message text, so bare `str(e)` produced `""`; fixed to
`f"{type(e).__name__}: {e}"`, so a failure at least shows `"ReadTimeout: "`
instead of nothing. (b) a **single bounded retry** on any agent step that
still times out (`_ainvoke_with_retry`, one retry, not a loop — same
discipline as Stage 2's CRAG grading) — the underlying MLX flakiness didn't
disappear (it's Ollama/MLX-engine-level, outside app code's control), it's
just fully absorbed now: bounded, retried once, cleanly reported on failure.

Verified live, three consecutive campaign-creation runs, each watched with
a 5–10-second polling loop against both `GET /` and the campaign's
`world_prep_status`:

| Run | App froze? | Outcome |
|---|---|---|
| 1 (seed-context fix only) | No | Failed — `world_prep_error` empty (the message bug) |
| 2 (+ error-message fix) | No | Failed — error now readable (`ReadTimeout: `) |
| 3 (+ bounded retry) | No | **Completed** — 2 separate timeouts hit and recovered from automatically (confirmed in logs: `npc-prep` and `opening-location` steps each retried once, both succeeded) |

Zero freezes across all three — the core problem — and the retry took a
run that would have failed 100% of the time down to a clean success.

**A fourth, complementary change, not a fix but a UX one**: `game.html` and
`session_zero_index.html` now gate their normal content behind
`campaign.world_prep_status`, showing a spinner + self-terminating 3-second
poll while `in_progress`/`not_started`, and a non-blocking amber banner
(never a hard block) if `failed`. Before this, a still-preparing or
failed-but-recoverable campaign looked indistinguishable from a genuinely
broken app — the whole debugging session tonight started from exactly that
confusion. Worth noting for its own sake: world-prep's *total* runtime is
allowed to be genuinely long (a location-dense adventure's agent run can
legitimately take several minutes across many tool calls) — the fix here
bounds each *individual* Ollama request, never the workflow as a whole, so
this gate needed to reflect "still working" accurately rather than assume
anything past N seconds means something's wrong.

One diagnostic dead end worth recording so it isn't retried blindly next
time: attempted a live `py-spy dump` on the frozen container to get a
definitive stack trace before restarting. Installed cleanly
(`pip install py-spy` inside the container), but this container's process
topology defeated it — `docker top`'s host-side PIDs don't match what's
visible inside the container's own PID namespace via `docker exec`, and
even after finding the container-relative PIDs, the actual worker thread
wasn't independently attachable (`py-spy dump --pid 1 --subprocesses` found
the reloader and a `multiprocessing.resource_tracker` helper, but a third
PID reported "Failed to get process executable name" — likely a thread
sharing PID 1's process image, not a genuinely separate process). Not a
dead end in "py-spy is bad" — a dead end in "this specific uvicorn
`--reload` + Docker PID-namespace combination needs more setup (e.g.
`--cap-add=SYS_PTRACE`, or running natively) before it's useful here."

### The freeze recurs mid-session (2026-07-09): the same trigger, in the one place the 2026-07-08 fix didn't reach

Reported live, twice in about fifteen minutes, mid-game: the app went fully
unresponsive (`curl` timing out completely, ~0% CPU, no new log lines) —
the exact signature as the world-prep freeze above, but this time nothing
to do with world-prep at all. The user's own instinct nailed it before the
logs did: "I really think it's something about starting combat."

They were right, and the mechanism is the same embed↔chat model swap
already root-caused on 2026-07-08 — just triggered from a different call
site the previous fix never touched. `search_rules` and `search_lore`
(`backend/tools/rules.py`, `backend/tools/lore.py`) both call
`RulesStore.search()`, whose hybrid pipeline was still doing dense (Chroma,
`nomic-embed-text`) retrieval **and then** a separate `LLMJudgeReranker`
call on `gemma4:26b-mlx` — two genuinely different model architectures
(a small embedding model vs. the full 26B chat model), not "the same model
taking longer with more context." Every `search_rules` call forced Ollama to
evict one and load the other. The 2026-07-08 fix only ever touched two
call sites — world-prep's seed step and end-of-session summarization —
`search_rules`/`search_lore` (the live-gameplay tools) kept the swap intact
the whole time. Combat-start is exactly where this gets hit hardest:
`create_monster`'s own docstring recommends a `search_rules` call per
monster for stat-block grounding, so starting a multi-monster encounter can
trigger the swap several times in one turn — far more concentrated than a
typical exploration turn.

**The fix, same discipline as 2026-07-08 (stop the swap, don't
better-survive it):** `RulesStore.search()` gained a `use_reranker: bool =
False` parameter — the reranker call is now opt-in, not automatic.
Dense+BM25 fused via Reciprocal Rank Fusion is a legitimate hybrid retrieval
result on its own even without an LLM-judged reorder on top; this trades a
little ranking precision for not freezing the app on the single
most-called live lookup in the game. `scripts/eval_retrieval.py` (which
specifically measures reranked quality, offline, where a slow-but-recoverable
call is fine) opts back in explicitly with `use_reranker=True`.

Noted but deliberately not fixed tonight (out of scope for what was asked,
flagged for later): `search_lore`'s CRAG-style grading loop
(`grade_sufficiency`/`reformulate_query`, `backend/rag/grading.py`) makes
its own separate chat calls interleaved with `search()`'s now-reranker-free
dense/BM25 retrieval — still an embed→chat→(maybe embed→chat again)
sequence, just a different shape than the one fixed here. Lower combat-time
risk than `search_rules` (lore lookups aren't the tool `create_monster`
leans on), but the same underlying swap risk exists there too if it's ever
hit as hard.

Also fixed in passing while investigating: a `docker compose exec` test
script copied into `/app` to inspect the running container was itself
picked up by `uvicorn --reload`'s file watcher, triggering a reload —
explains a batch of unexplained "N changes detected" log lines from
earlier the same night that briefly looked like a mystery background
process. Not a bug in the app; a lesson for debugging it (copy scratch
scripts outside the watched directory, or invoke inline).

---

## Prep Scripts

All scripts below live in `scripts/` (moved from repo root 2026-07-05 for a
cleaner top level) — run as `python scripts/<name>.py` from the repo root.

### `ocr_ingest.py` — PDF → Markdown

**Superseded 2026-07-07 — this doc had gone stale describing the old pipeline until corrected 2026-07-14.** Every PDF (digital or scanned) now goes through one uniform path: **MinerU's `vlm-engine` backend** (MLX-accelerated on Apple Silicon), a purpose-built PDF→Markdown pipeline with a real layout/reading-order model. This replaced the two-tier PyMuPDF-native-text + Apple-Vision-OCR pipeline described below until this correction — that pipeline had neither a real layout model (raw PyMuPDF text extraction has no layout awareness at all) nor reliable multi-column handling (Vision, the engine behind Live Text, scrambled some multi-column pages and let running headers/footers bleed into body text). MinerU strips running headers/footers/page numbers natively instead of needing the two-tier approach's `page.get_text()`-vs-Live-Text detection heuristics.

Tradeoff, stated directly in the script's own docstring: this costs meaningfully more time on already-digital PDFs than the old fast PyMuPDF path did (MinerU is a page-image VLM read, tens of seconds per page, regardless of whether the PDF already had a clean text layer) — but a uniform path is what actually fixes the reading-order/header bugs, and per-PDF speed isn't the bottleneck for a personal library ingested once. PyMuPDF (`fitz`) is still imported, but now only for page counts, not extraction.

**Chunked, checkpointed extraction:** MinerU's `vlm-engine` batches a multi-page run into internal 64-page windows and hands off between them — confirmed live to crash reproducibly ("Timed out waiting for result of task") at exactly page 128 (2×64) on two separate full-book runs, while an isolated 15-page run spanning that same page range completed cleanly (the window-to-window handoff itself is broken, not any particular page). `_extract_with_mineru` sidesteps this by keeping every individual MinerU invocation at or under `_CHUNK_PAGES` (60) and checkpointing each window's raw text to a `.{stem}.ocr_chunks/` cache dir before concatenating into the final `.md` — a crash loses at most one window's work, not the whole book; a re-run skips windows whose cache file already exists (same discipline as `extract_entities.py`'s per-entity checkpointing). A `.partial` staging file (written first, renamed to the real output only on success) means a mid-run crash leaves no file at the final path, so a later run without `--force` correctly treats the book as not-yet-done rather than trusting a truncated file.

**Backend-name portability caveat:** the `-b`/`--backend` value (`--mineru-backend`, default `vlm-engine`) isn't stable across MinerU versions/platforms — confirmed live that a fresh `pip install -U "mineru[all]"` on Windows resolved a newer MinerU release whose valid choices are `pipeline`/`vlm-http-client`/`hybrid-http-client`/`vlm-auto-engine`/`hybrid-auto-engine` (no bare `vlm-engine`/`hybrid-engine` at all), while this project's Mac install (3.4.2) only has the un-`"auto"` names. Run `mineru --help` and check the `-b`/`--backend` valid-choices list if you hit "invalid value for -b."

**Windows console codec fix:** Windows' default console codec (cp1252/"charmap") can't encode the em-dashes/box-drawing/arrow characters this script prints for readability — confirmed live as a real `UnicodeEncodeError` crash on a fresh Windows venv (Mac/Linux default to UTF-8 stdout so this never surfaced there). stdout/stderr are reconfigured with `errors="replace"` so a genuinely unencodable character degrades to "?" instead of crashing an otherwise-successful book mid-run.

**Requirements:** `uv pip install -U "mineru[all]"`. The MLX backend needs macOS 13.5+ on Apple Silicon and `mlx-vlm` to import cleanly, or MinerU silently falls back to a much slower CPU/transformers path — verify before a big job with `python -c "from mineru.utils.engine_utils import _select_mac_engine; print(_select_mac_engine())"` (must print `"mlx"`, not `"transformers"`; if it prints `"transformers"`, `import mlx_vlm` is failing, e.g. a Python built without `_lzma` support breaking the transformers/torchvision import chain).

**Open question, surfaced 2026-07-14 during the map/text-linking survey (`research/map-text-linking-survey.md`):** whether the paragraph-density bug documented below (found against the *old* Tier-1 PyMuPDF pipeline's Out of the Abyss output) is still representative of current MinerU output was not confirmed — would need a re-ingest to check, out of scope for that survey. Historical record, from the now-replaced pipeline: `get_text("text")` only inserted a paragraph break (`\n\n`) *between* pages, not within one, which for some PDFs (Out of the Abyss: 64 paragraphs across 5803 lines, vs. 800+ for similarly-sized adventures) collapsed whole pages into a few giant blobs, starving `add_headers.py`'s candidate detection (which requires a heading to be the first line of a `\n\n` paragraph) down to zero hits — silently, no error. `check_paragraph_density()` (`scripts/validate_source.py`) was built as a permanent, reusable check for this (flags any `.md` whose paragraphs-to-lines ratio falls under `PARAGRAPH_RATIO_THRESHOLD`, 0.05) and remains a sound audit tool regardless of which extraction pipeline produced a given `.md` — just re-run it (`python scripts/validate_source.py --input docs/source/adventures --severity error`) against any newly-produced MinerU output to confirm the bug class doesn't recur there. The still-open per-book audit gap described below (7 adventures with no source `.md` on this machine to check) is unaffected by this pipeline change — those books simply haven't been (re-)ingested here yet, under either pipeline.

```bash
python scripts/ocr_ingest.py                                    # whole docs/raw/ folder
python scripts/ocr_ingest.py --file docs/raw/foo.pdf --pages 5   # smoke test
python scripts/ocr_ingest.py --force                             # re-process even if .md exists
```

### `clean_source.py` — LLM artifact cleanup

Scans extracted `.md` files for garbled paragraphs. Sends only flagged paragraphs to a local Ollama text model for correction. Length ratio guard (0.5–2.0×) rejects bad LLM output.

```bash
python scripts/clean_source.py --model qwen2.5:3b     # recommended — fast enough
python scripts/clean_source.py --dry-run              # detect only, no writes
```

### `validate_source.py` — QA report

Heuristic validation: OCR failure comments, repeated-line clusters, garbled numbers, HP dice math mismatches, ability scores out of range, incomplete stat blocks.

### `build_index.py` — Postgres/pgvector indexer

Reads `docs/source/core/` and `docs/source/adventures/{slug}/`, splits each `##`/`###` section into a parent chunk (max 1500 chars, 50-word overlap on size-split oversized parents) and further into ~350-char child chunks, contextualizes each child (Anthropic-style — see "Evolution" section above; contextualization is a chat call, served via `vllm_chat`/`--vllm-url` since the 2026-07-13 vLLM-metal migration, separate from `--ollama-url` which is embeddings-only now) unless `--skip-contextualization`, embeds children with `nomic-embed-text` (via Ollama), and upserts into the `rule_chunks` table in batches of 8 (small on purpose — a kill loses at most one small in-flight batch; resumability is via a plain `chunk_id` existence check against the table itself, not a separate cache). `content_tsv` (the sparse/keyword half of hybrid search) is a Postgres `GENERATED ALWAYS AS` column — always in sync automatically, no separate rebuild step (was a `data/bm25_rules.pkl` rebuild before the 2026-07-12 ChromaDB→pgvector migration). Was ChromaDB (`data/chroma_db/`) before that same migration. A scoped run (`--book`/`--adventure`/`--source-type`) is incremental by default — existing rows are left alone unless `--fresh` (delete just that scope first) or `--wipe` (delete everything first) is passed explicitly.

Metadata per chunk: `book`, `section`, `source_type` (`"core"` | `"adventure"`), `adventure` (slug, empty for core), `granularity` (`"parent"` | `"child"`), `chunk_id`, `parent_chunk_id` (child only).

```bash
make index                                         # full reindex
python build_index.py --wipe                       # clear and rebuild
python build_index.py --adventure "Tyranny of Dragons"  # one adventure only
python build_index.py --source-type core           # core books only
```

### Full prep pipeline

```bash
python scripts/ocr_ingest.py
python scripts/clean_source.py --model qwen2.5:3b
python scripts/validate_source.py
make index
```

### Incident (2026-07-07/08): bulk ingestion via Docker OOM-killed repeatedly — use `ingest-book-native` even on the canonical machine

A three-book overnight batch (PHB + Monster Manual + DMG, `scripts/overnight_queue_phb_mm.sh`,
routed through `make ingest-book` → `docker compose exec app python
build_index.py`) died from an OOM kill (`Killed: 9` / `Error 137`) on three
separate attempts — at 5%, then 35% through PHB's reindex, and once by
killing the `app`/`db` containers outright mid-session. Root cause, confirmed
via `docker info`: this laptop's Docker Desktop VM had **~965MB of total
RAM**, shared across Postgres + the app + any `docker compose exec` process.
Nowhere near enough for embedding 100k+ chunks, and flaky enough that it also
killed unrelated containers under casual diagnostic load (a plain `chromadb`
metadata query over `docker exec` triggered the same 137).

**Fix: run the heavy scripts natively instead of through `docker compose
exec`.** `make ingest-book-native` + `make setup-venv` already existed in the
Makefile for a *different* reason — `docs/engineering-notes/desktop-native-ingestion.md`
built them so a second, Docker-less desktop could do offline bulk OCR/ingestion.
Turns out the exact same escape hatch fixes the Docker-VM-memory problem on
the **primary** laptop too, and required zero config changes to work here:
- `data/chroma_db` is a bind mount (`./data/chroma_db:/app/data/chroma_db` in
  `docker-compose.yml`), so native and containerized processes read/write the
  identical files on disk.
- `docker-compose.yml` publishes Postgres to `localhost:5432`, and
  `backend/config.py`'s own defaults already point at `localhost` (the
  container-only hostnames — `db`, `host.docker.internal` — are env-var
  overrides layered on top for the containerized app) — so
  `make ingest-book-native ... write_postgres=1` writes straight into the
  same canonical Postgres the live app uses, no JSON-registry round-trip
  needed (that round-trip is for the genuinely-Docker-less desktop case).
- Ollama already runs natively on the host, not in Docker, so no networking
  change needed there either.

Net effect: the native process runs against the host's full 32GB of RAM
instead of the VM's ~1GB ceiling. Re-run of the same PHB job natively
finished clean end-to-end (no OCR needed, cached from before) in 3h19m —
indexing 110,061 chunks + extracting 587 entities — with memory staying flat
around 1.3–1.6GB throughout (confirmed by sampling RSS over time: growth was
front-loaded startup cost — BM25 pickle load, Chroma/HNSW init — not a
per-chunk leak). **Takeaway: prefer `make ingest-book-native` over
`make ingest-book` for any bulk/overnight ingestion job on this machine too,
not just the secondary desktop** — the Docker route only makes sense for
small one-off scoped runs where the VM's memory ceiling won't matter, or
until that ceiling is deliberately raised in Docker Desktop's settings
(untested — 4–8GB would likely be plenty given 32GB host RAM, but no
profiling was done to confirm a minimum).

### Incident (2026-07-08): `extract_entities.py`'s `canonicalize()` had no batching or output cap — silent multi-hour stall on DMG

Same overnight run, next symptom: after PHB and Monster Manual finished
clean (native), DMG's entity-extraction step went **5+ hours with zero new
log output** after its per-window discovery pass completed. Not actually
hung — `ollama ps` showed the model pinned at 100% GPU and the runner
process's cumulative CPU time was still climbing when sampled — but there
was no way to tell from outside whether it was almost done or stuck forever.

Root cause, found by reading `canonicalize()`: it sent the **entire**
deduped candidate-name list for a kind (`npc`/`location`/`item`) as **one**
`temperature=0`, no-`max_tokens` LLM call, expecting one output line per
distinct entity back. Fine for a book with a few hundred candidates; DMG's
item/table-heavy chapters (huge magic-item tables, sample treasure, etc.)
almost certainly produced a candidate list far larger than anything
previously run through this path, and the resulting single call had no
progress bar and no ceiling on how long it could run.

**Fix:** `canonicalize()` now batches the sorted candidate list at
`CANONICALIZE_BATCH_SIZE = 150` names per LLM call (see `scripts/extract_entities.py`),
with a `tqdm` progress bar per batch when there's more than one. Batching
over the *sorted* list (not a random split) was a deliberate choice — alias
variants of the same name usually share a prefix (e.g. "Rose" /
"Rosavalda"), so sorting keeps them likely to land in the same or an
adjacent batch even though canonicalization is no longer attempted *across*
a batch boundary. That's the same skip-on-doubt tradeoff the function
already made pre-fix (worse dedup in rare cases, but an entity is never
silently dropped) — just applied per-batch instead of globally. **Takeaway:
any single-shot LLM call whose input size scales with book content (not a
fixed small prompt) needs either a hard batch size or an explicit token cap
before it's trusted on the biggest/densest book in the corpus (DMG, not
PHB/MM) — "worked fine on the first two books" isn't evidence it'll work on
the third.**

---

## Docker & Deployment

### Local development

```bash
touch .env        # can be empty — overrides set in docker-compose.yml
make up
make setup        # migrate DB + build index if rule_chunks is empty
# visit http://localhost:8000
```

Source is volume-mounted (`./:/app`, plus `scripts/`, `docs/source/core`, `docs/source/adventures` explicitly — see docker-compose.yml's own comments for why each is listed) so `uvicorn --reload` picks up changes without rebuilding. `OLLAMA_BASE_URL=http://host.docker.internal:11434` reaches Ollama on the host (embeddings only, since the 2026-07-13 vLLM-metal migration); `VLLM_BASE_URL=http://host.docker.internal:8100/v1` reaches the vLLM-metal chat server the same way.

**Postgres/pgvector portability (was "ChromaDB portability" — `data/chroma_db/` — before the 2026-07-12 migration):** rule/session-chronicle vectors now live in the same Postgres instance as everything else — no separate bind-mounted directory to copy between machines. A fresh machine just needs `DATABASE_URL` pointed at a reachable Postgres and `make index`/`make reindex-full` run once.

**Stale check — fixed.** The Makefile's `index-if-empty` target used to literally check whether `data/chroma_db` was empty on disk to decide whether to reindex — a leftover from before the Postgres migration that always tripped true (that directory is never populated anymore), so `setup` always re-triggered a full reindex instead of actually detecting whether `rule_chunks` already had data. Now queries `SELECT count(*) FROM rule_chunks` directly via `psql` and only reindexes when it's actually empty.

### Production (Railway)

- Add PostgreSQL plugin → `DATABASE_URL` injected automatically (also now holds the rule/session-chronicle vector tables, see above)
- Set `OLLAMA_BASE_URL` to wherever Ollama is hosted (embeddings only)
- Set `VLLM_BASE_URL` to wherever the vLLM-metal chat server is hosted
- Same Docker image; `CMD` runs `alembic upgrade head` before uvicorn

### Makefile targets

| Target | Does |
|---|---|
| `make up` | Start all services detached |
| `make down` | Stop all services |
| `make build` | Rebuild app image (no cache) |
| `make restart` | Rebuild + restart app container only (db keeps running) |
| `make migrate` | `alembic upgrade head` inside app container |
| `make migration name=…` | Generate a new Alembic revision |
| `make rollback` | `alembic downgrade -1` |
| `make logs` | Tail app logs |
| `make db-logs` | Tail db logs |
| `make psql` | Open psql shell in db container |
| `make shell` | Bash into app container |
| `make fresh` | Tear down volumes, restart clean, run migrations |
| `make index` | Reindex `rule_chunks` (Postgres/pgvector) from `docs/source/` — incremental by default, see build_index.py's `--fresh`/`--wipe` |
| `make index-if-empty` | Reindex only on a fresh clone — checks `rule_chunks`' real row count via `psql`, skips if already populated |
| `make setup` | `migrate` + `index-if-empty` — the one-command new-machine bootstrap |
| `make ingest-book book="…" source_type=core` | Reindex + extract lore/monsters for one book, via `docker compose exec` — fine for small scoped runs, but see the ingestion-incident writeup above before using this for a big overnight job |
| `make ingest-book-native book="…" source_type=core write_postgres=1` | Same, but native (host `.venv`, no Docker) — bypasses the Docker Desktop VM's memory ceiling entirely; **preferred for bulk/overnight ingestion, even on this machine** |
| `make setup-venv` | Create/refresh the host `.venv` used by the `-native` targets |
| `make merge-chroma source=…` | Merge a second machine's `data/chroma_db/` into this one's (native) |
| `make load-lore-json book="…"` | Load a `-native` run's JSON entity registry (from a Postgres-less machine) into this machine's canonical Postgres |
| `make qa-campaign` | Build/rebuild the "QA Test Campaign" (`scripts/qa_smoke_test.py`) — see "QA smoke-test campaign" below |

### QA smoke-test campaign (2026-07-13)

`scripts/qa_smoke_test.py` (`make qa-campaign`) — a small, self-contained
scenario ("The Rusty Anchor," a tavern building) exercising every mechanic
from the grid-maps/opportunity-attacks/pronouns/chargen-fidelity work in one
deterministic pass: an authored grid (walls/doors/furniture), a two-PC party
via real chargen (pronouns, subclass validation reject-then-accept, real
starting equipment/gold), an allied DM-companion NPC, ability checks/saves,
a combat encounter with opportunity attacks (a standard-reach monster and a
10 ft reach-weapon monster both firing off one retreat — multi-attacker
resolution — plus a separate reaction-pause demonstration via a War
Caster-style feature), loot via both paths (`end_encounter`'s automatic
CR-scaled roll and manual `reveal_loot`), a magic item, a map-item location
unlock, a level-up (subclass validation again), fog-of-war (partial reveal
from real combat movement), and a session-log export.

Calls the real tool functions directly — no LLM, deterministic and fast,
since this is testing mechanics, not narration (the same style used
throughout this project's manual live verification). Safe to re-run
(deletes the previous "QA Test Campaign" by name first) and leaves the
result in Postgres afterward, printing direct links — a tangible fixture to
click through in the browser (`game.html`, the Maps browser, Session
History), not just a pass/fail check. 31 checks, all passing as of writing.

**A real, indexed adventure module, so it can also be played live (not just
scripted) — added same day.** The deterministic script above proves the
mechanics work in isolation; it doesn't exercise the actual RAG/DM-agent
pipeline (`search_rules`, world-prep, a live LLM session) at all. To close
that gap, the same scenario was back-written as a real adventure document —
`docs/source/adventures/The Rusty Anchor/The Rusty Anchor.md` (prose,
`# HEADING`-per-section, same style as the other indexed books) +
`_meta.json` (`opening_hook`/`opening_location`/`opening_section_marker`,
same fields every other adventure uses) — then indexed for real via
`make ingest-book-native adventure="The Rusty Anchor" source_type=adventure
skip_context=1` (the documented low-memory-safe path, not the Docker
`docker compose exec` path — see the ingestion-incident notes elsewhere in
this doc; running the Docker path concurrently with other memory-heavy work
OOM-killed it, exit 137, during this same session). 78 chunks indexed;
verified retrievable with real, on-topic content via
`RulesStore.search_adventure_only("smugglers cellar loot strongbox",
adventure="The Rusty Anchor")` — grid ingestion pipeline confirmed sound
end to end, not just the mechanics tools in isolation. A real campaign can
now be created with `books_in_play: ["The Rusty Anchor"]` and actually
played through live. `extract_entities.py`'s lore/monster-registry
extraction step was skipped for this content — it still requires Ollama
(intentionally not running this session for memory reasons, see
`ollama_base_url`'s note in `config.py`), which isn't needed for the
adventure to be searchable/playable, only for the separate Lore Registry
feature.

---

## Status

### Completed ✅

**Infrastructure**
- `Dockerfile`, `docker-compose.yml`, `Makefile`, `alembic/`
- `backend/config.py` — pydantic-settings
- `backend/models.py` — all Pydantic v2 domain models, 17 enums

**Database**
- `backend/stores/tables.py` — 13 SQLAlchemy Core tables
- `backend/stores/campaign_store.py` — full CRUD + roll log
- Alembic migration `0001_initial_schema.py`

**RAG / Vector stores**
- `docs/source/` restructured: `core/` + `adventures/{slug}/` with `_meta.json`
- `backend/stores/rules_store.py` — Postgres/pgvector `rule_chunks` table, `books_in_play` filter (was ChromaDB `rules` collection before the 2026-07-12 migration)
- `backend/stores/history_store.py` — Postgres/pgvector `session_chronicle_chunks` table (was ChromaDB `session_chronicles` collection)
- `build_index.py` — full indexer with core/adventure metadata, targeted re-index flags

**Tools (37+ total — this count predates the combat-resolution refactor's resolution.py tools; not re-audited)**
- `dice.py` (1), `rules.py` (1), `memory.py` (1)
- `party.py` (8), `npc.py` (4), `combat.py` (5), `world.py` (4), `quest.py` (3), `campaign.py` (2)
- `chargen.py` (6) — Session 0 only
- `companion.py` (1) — Session 0 and in-game; DM-steered companion generation to fill out a party toward the adventure's recommended size
- `levelup.py` (1) — in-game only; `level_up` recomputes HP/proficiency/spell slots/hit dice from real data, see the Tools section above
- `backend/tools/registry.py`

**Agent**
- `backend/agent/prompts.py` — separate mechanics/narrator system prompts, campaign history hint
- `backend/agent/dm_agent.py` — two-model mechanics/narrator LangGraph `StateGraph`, message trimmers, SSE streaming (narrator-only), session summarizer, transcript formatter, Session 0 agent
- `backend/agent/session_zero_prompt.py` — campaign pitch + character creation prompt
- `backend/data/fivee_options.py` — hardcoded PHB data

**Application**
- `backend/main.py` — all 19 routes
- `templates/` — base, index, game, sessions, session_zero_index, session_zero
- `static/style.css` — dark fantasy theme, all page layouts
- Game sidebar: world clock/weather, faction/NPC relationship list, safety tool (X-card) — see **Feature Brainstorm** below

**Prep pipeline**
- `ocr_ingest.py` — PDF extraction via MinerU's vlm-engine (MLX-accelerated, macOS only) — was two-tier native-text/Apple-Vision-OCR until the 2026-07-07 migration; see "Prep Scripts" above
- `clean_source.py` — LLM artifact cleanup
- `validate_source.py` — heuristic QA

### Planned Future Features

- **TODO, partially done (2026-07-13):** audit other campaigns for the Tier 1 `get_text("text")` low-paragraph-density bug described under `ocr_ingest.py` above. Now a permanent one-command check (`scripts/validate_source.py`'s `check_paragraph_density`) instead of a one-off script. Curse of Strahd and Lost Mine of Phandelver checked clean; the other 7 (Ghosts of Saltmarsh, Icewind Dale, Storm King's Thunder, Tales of the Yawning Portal, Tomb of Annihilation, Tyranny of Dragons, Waterdeep) have no source markdown on this machine to check — blocked, not skipped, see the full note above.
- Live party tracker panel updating during combat (HTMX polling) — partially done: `game.html`'s sidebar already renders the initiative order/round while combat is active (see "Initiative tracker UI panel" below), but whether it live-updates via HTMX polling mid-turn or only refreshes on the next full page load hasn't been re-verified — worth confirming before calling this fully done.
- ~~Initiative tracker UI panel in sidebar~~ — Done: `templates/game.html`'s `.combat-active` sidebar section (round counter + initiative order, current-turn highlighted).
- ~~Long-rest / short-rest quick buttons~~ — Done, see "Rest buttons (2026-07-03)" below.
- ~~Session summary export (markdown or PDF)~~ — Markdown half done 2026-07-13: `backend/session_export.py`'s `render_session_export_markdown()` (pure function, no DB) flattens every recorded `Session` (summary, key events, adventure progress, XP, loot, quest names resolved from ID, notes), oldest first, into one document. `GET /campaigns/{id}/sessions/export/markdown` (`main.py`) loads the campaign and returns it as a downloadable `.md` (`Content-Disposition: attachment`); a "⬇ Export Session Log" link was added to `sessions.html`'s sidebar, gated on there being at least one session. Verified live against a real campaign ("The Lost Mine of Ragdelver") via curl — correct filename, content-type, and rendered body. PDF still not built — flagged as a trivial follow-up (pipe this same markdown through a converter) rather than adding a new rendering dependency for this pass. Regression coverage: `tests/test_session_export.py`.
- Multi-player support (Redis pub/sub replacing in-memory queue)
- Railway deployment guide + one-click deploy button
- NPC/faction relationships, world clock/weather, safety tool, and map support — see **Feature Brainstorm** below for full detail; these are accepted and superseding the old "Grid map renderer" / "NPC relationship display" bullets.

---

## Feature Brainstorm — Phase 2 Candidates

Brainstormed 2026-06-30. Each idea has a high-level pitch, a prototype sketch grounded in the current stack, and a feasibility call. Status reflects what's been decided so far — not build order within a tier.

### Accepted — Next Up — Done ✅ (2026-06-30)

**NPC / faction relationship graph**
Surface the relationship web already captured in the data model — `Faction.relationships` and `NPC.attitude` are tracked but never visualized. Render as a small force-graph or relationship list in the game sidebar; a CDN script tag keeps this npm-free, consistent with the existing HTMX CDN approach.
*Feasibility: High — all data already modeled, this is a rendering pass only.*
*Shipped as: a "Factions & NPCs" sidebar section in `templates/game.html` — relationship list (no graph rendering needed in practice), faction reputation badges, NPC attitude badges color-coded green/yellow/red. Only shows factions/NPCs the party has actually met. Styles in `static/style.css`.*

**World clock & weather sidebar**
`days_elapsed`, `time_of_day`, and `last_long_rest_day` already exist on `Campaign` but aren't shown anywhere. A small always-visible widget surfaces them.
*Feasibility: High — no new data, pure display.*
*Shipped as: a "World" sidebar section in `templates/game.html` showing in-game date, time of day, days elapsed, weather, and long-rest status. No backend changes needed.*

**Safety tool (X-card)**
A quiet player-facing "pause/skip this" control that posts a flag the DM agent is instructed to respect (steer away from the topic) and logs it to campaign notes for the human DM to review later.
*Feasibility: High — one small tool + UI affordance, genuinely useful for real play groups.*
*Shipped as: `Campaign.safety_flags: list[str]` (new field, `backend/models.py`), `POST /campaigns/{id}/safety-flag` and `POST /campaigns/{id}/safety-flag/clear` (`backend/main.py`), a "Safety Tools" sidebar section with an 🛑 X-Card button + optional note (`templates/game.html`). Active flags are injected into both the in-game and Session 0 system prompts (`backend/agent/prompts.py`, `backend/agent/session_zero_prompt.py`) instructing the agent to steer away without drawing attention to it; every flag is also permanently logged to `Campaign.notes` as an audit trail, independent of the DM clearing the active list.*

### Accepted — Lower Priority (mapping & spatial reasoning — four ideas, shared foundations)

These four ideas all reduce to the same two abstractions, one of which is already half-built in the data model:

- **A scale-aware graph edge.** `LocationConnection` already has the right shape — `distance_ft`, `is_passable`, `is_visible` — for a 15-ft dungeon corridor. A 12-mile forest road between two settlements is *the same edge*, just a different scale and a non-instant traversal cost. Tagging `Location` with a `scale: LocationScale` (`SITE` | `REGION`) and letting `LocationConnection` carry either `distance_ft` (site) or `distance_miles` + `terrain` (region) means one node/edge model serves both the combat-grid work and the travel-logistics work below, instead of a parallel `Region`/`TravelRoute` table that duplicates connection, visibility, and passability logic.
- **A coordinate on a `Location` for rendering.** `CombatantPosition.coordinates` has sat unused since it was added "for when a grid map is eventually introduced" — the same idea (one xy/hex pair per node) applies one level up, to placing a `Location` on a rendered map. Building the renderer once, parameterized by scale, means item 2's CSS-grid/token component and a future region-scale map are the same code path, not a rewrite.
- **A revealed/visibility gate.** `LocationConnection.is_visible` is already a boolean fog-of-war primitive at site scale. "Has the party discovered this route/region yet" (item 4) is the identical mechanic one level up. No new concept needed — just apply the existing field at both scales.

Net effect: design the scale-aware model *before* building any one of these, so the site-grid renderer (item 2) isn't thrown away when regional travel (item 4) needs its own map.

**1. Map grids (ASCII/XY) — Done ✅ (2026-07-13)**
Add `grid: list[str]` (ASCII rows) + `legend: dict[str, str]` to `Location` as a JSONB field, gated to `scale: SITE`. A new tool, `get_location_grid()`, lets the agent reason spatially ("you're 15ft from the door"). `combat.py`'s `set_combatant_position` extends to validate `(x, y)` against grid bounds — finally populating `CombatantPosition.coordinates`.
*Feasibility: High for DM-authored grids (small JSON or in-app editor). Low–Medium for auto-extracting maps from PDF map images — would need a vision-capable model to interpret floor plans, experimental and error-prone. Ship authored grids first; auto-extraction is a stretch goal, not a dependency.*
*Shipped as: `Location.grid`/`.legend` (`backend/models.py`, ANY scale — not gated to SITE after all, since a region-scale settlement's street layout turned out to want a grid just as much as a dungeon room; see the Maps-browser note below). `set_location_grid`/`get_location_grid` (`backend/tools/world.py`, in `make_authoring_tools` so both live play and the world-prep pass get them) — validates a rectangular grid and that every non-`.` symbol has a legend entry, same "no silent invented data" discipline as `validate_subclass`/`build_spells_known`. `CombatantPosition.coordinates` finally populated via `set_combatant_position`'s new `x`/`y` params (`combat.py`), validated against grid bounds and wall cells. `start_encounter` hard-refuses without a grid on the party's current location — **every fight now requires one, wilderness included** (a direct, explicit user requirement — no zone-only fallback). Went beyond the original sketch in one real way: this directly replaced a rejected zone-transition heuristic for opportunity attacks (see that item below) rather than shipping as an isolated rendering feature — real coordinates existing is what made a real distance check possible instead of a heuristic. Auto-extraction from PDF map images is still not built (stretch goal, as originally scoped) — grids are DM/model-authored, live during play or precomputed for settlements during world-prep (see the Maps-browser note below).*

**2. Visual map for players — Done ✅ (2026-07-13)**
A pure CSS-grid table in `game.html`'s sidebar — cells colored by terrain/lighting, combatant tokens placed by coordinate, refreshed via the same HTMX polling pattern already planned for the live party tracker. No npm, no JS framework. Built generically enough (grid renderer takes a node list + coordinates) that it can later be reparameterized for the region-scale map in item 4 rather than rewritten.
*Feasibility: High, once (1) exists.*
*Shipped as: `GET /campaigns/{id}/combat-map` (`backend/main.py`) — reuses `render_grid()` (`map_render.py`) as-is for terrain, plus a `combatants` array (name/x/y/side/is_current_turn/hp_pct) built from `Encounter.combatant_positions` cross-referenced with `initiative_order`. Refreshed via `refreshCombatMap()` (`game.html`), hooked into the exact same trigger points the existing `refreshCombatPanel()` already used — page load + the SSE `"done"` event after each DM turn — not a new polling timer, since that event-driven pattern already existed for the combat turn-builder. Renders into a `#combat-map` div inside the sidebar's `.sidebar-section.combat-active` block (paired with round/initiative), reusing the Maps browser's own `.map-grid`/`.map-row`/`.map-cell-*` CSS classes for terrain and a small marker overlay (`.combatant-marker`, party/hostile colors, current-turn outline, HP-tint via `.hp-bloodied/.hp-critical/.hp-downed`) — simpler than a separate absolute-position overlay grid: a combatant's marker is just rendered *inside* its own cell's span instead of the raw terrain symbol. Deliberately unfogged (see item 3) — this is real-time battlefield awareness for a fight the party is already in, not exploration secrecy, so it always shows the true state.*

**3. Fog of war (hide what players shouldn't see) — Done ✅ (2026-07-13, coarse version)**
Coarse version: only render cells belonging to *revealed* rooms, reusing the existing `reveal_hidden_element` tool pattern and `LocationConnection.is_visible`. True line-of-sight (raycasting per cell) is a real algorithm, not just data plumbing.
*Feasibility: High for room-level concealment. Medium for true LOS — likely not worth the complexity for a narrative-first tool.*
*Shipped as (Maps browser only — never `get_location_grid`, the DM/mechanics model always sees the real grid): `Location.revealed_positions: list[tuple[int,int]]` (`backend/models.py`) — append-only, populated by `set_combatant_position` (`combat.py`) whenever a `side=="party"` combatant's real `(x,y)` is set (a monster walking through a cell doesn't mean the party has seen it). New `render_grid_fogged(grid, legend, revealed_positions, radius=2)` (`map_render.py`) — same Chebyshev-distance convention `check_opportunity_attacks` already established (`_helpers.py`), any cell farther than `radius` squares from every revealed position becomes a `{"symbol": "?", "kind": "fog"}` placeholder, same output shape as `render_grid` so `maps.html` needed zero template changes beyond a `.map-cell-fog` CSS class. The `/campaigns/{id}/maps` route calls `render_grid_fogged` only when `revealed_positions` is non-empty, falling back to the original unfogged `render_grid` otherwise — a location nobody's ever fought in (most non-combat visits) still shows its whole layout immediately once visited, matching the original ask ("if we're in town in Phandalin, having the streets laid out would be nice") rather than defaulting to a useless solid-black map. Not true LOS raycasting, as scoped — a radius-around-visited-cells reveal, the "likely not worth the complexity" version this note already called out.*

**4. Regional travel & distance logistics — Done ✅ (2026-06-30)**
Give the DM agent grounded answers to "how far is it from A to B, and how long does it take to get there" instead of inventing numbers — the same philosophy as `roll_dice` replacing invented rolls. Reuses `LocationConnection` at `scale: REGION`: add `distance_miles: float | None` and `terrain: TravelTerrain` (road / trail / wilderness / mountain / swamp / water) alongside the existing `distance_ft` / `is_passable` / `is_visible`, so travel routes are edges in the same graph as dungeon connections, not a parallel model. Two new tools: `get_travel_estimate(destination)` walks the region-scale subgraph (simple BFS/Dijkstra over `distance_miles` — branching factor is small enough that no real pathfinding library is needed) and returns distance plus days at normal/slow/fast pace per DMG travel rules (24/18/30 mi/day, with mounted/wagon modifiers); `travel_to(destination, pace)` works like `move_party` but advances `Campaign.days_elapsed` and `time_of_day` by the computed duration — the first tool to ever actually increment `days_elapsed`, which today is tracked but dead. Stretch: a per-day random-encounter roll during multi-day travel, reusing indexed monster data via `search_rules`.
*Feasibility: High for the graph model and tools — small, well-scoped, reuses an existing shape rather than inventing one. Medium for the region-scale map rendering (the item-2 renderer reparameterized) — same no-npm, CDN-script approach as the NPC/faction graph. Auto-sourcing real-world distances between named PHB/module locations is out of scope; distances are DM-authored, same as grids in item 1.*
*Shipped as: `Location.scale`, `LocationConnection.distance_miles`/`terrain` (`backend/models.py`); `create_location`/`connect_locations`/`get_travel_estimate`/`travel_to` in `backend/tools/world.py` (split into `make_movement_tools`/`make_authoring_tools`/`make_travel_tools`); `advance_clock`/`find_connection` helpers in `backend/tools/_helpers.py`. Direct connections only, as scoped — no multi-hop pathfinding (deferred, per the original brainstorm's stretch framing). Beyond the original brainstorm: since there was no DM persona to author locations by hand (the AI is the DM), an automatic background pass now seeds region-scale locations/distances from a campaign's `books_in_play` — `backend/agent/world_prep.py` + `world_prep_prompt.py`, a one-shot non-checkpointed agent (`get_world_prep_agent` in `dm_agent.py`) fired via `asyncio.create_task` from `POST /campaigns` and `POST /campaigns/{id}/books` (`Campaign.world_prep_status`/`world_prep_error` track progress). Grounded only — only distances the adventure text states or clearly implies get created; gaps are left for later. Seed retrieval uses a new `RulesStore.search_adventure_only()` rather than the mixed core+adventure `search()`, since core rulebooks (~5k chunks) drown out a single adventure's own text (~500 chunks) for generic "regional overview" queries. Verified end-to-end against "Tyranny of Dragons": 6 locations, grounded mileage converted from stated travel times, `get_travel_estimate`/`travel_to` correctly advance `days_elapsed`/`time_of_day` and refuse ungrounded requests instead of inventing numbers.*

**5. Maps browser (persistent world-map viewer) — Done ✅ (2026-07-13)** (idea from user, expanded scope during planning for item 1)
Not originally scoped as its own idea — grew out of item 1's planning conversation when the user asked for a way to browse "what does this town look like" at any time, not just mid-combat, plus purchased-map unlocking and zoom navigation between a settlement's street grid and the wider region. Reuses item 1's `Location.grid`/`.legend` (now ungated from `scale: SITE` — a region-scale settlement gets a grid too) and item 4's already-shipped `LocationConnection` travel graph for zoom navigation, rather than inventing a new hierarchy/coordinate-transform concept.
*Shipped as:* `Location.visited`/`.map_known` (`backend/models.py`) — `visited` set automatically on arrival (`move_party`/`travel_to`, `backend/tools/world.py`); `map_known` set independently via a purchased/found map item (`Item.is_map`/`.map_location_id`, new `apply_map_reveal_if_needed()` in `_helpers.py`, wired into `add_item_to_character`'s new `map_of_location` param, `backend/tools/party.py`). `GET /campaigns/{id}/maps` (`backend/main.py`) lists every location where `visited or map_known`; the detail pane renders the selected location's grid as colored HTML cells — `backend/map_render.py`'s `classify_symbol()`/`render_grid()` (pure, testable) buckets each symbol into a small fixed palette (wall/door/water/vegetation/rock/difficult/furniture/floor/other) by keyword-matching the legend's free text, rendered black-background/monospace, roguelike-style (`static/style.css`'s `.map-cell-*` classes). "Zoom" navigation is just links to a location's already-existing `LocationConnection`s that also happen to be visited/map_known — no new coordinate-transform math. `templates/maps.html` mirrors `sessions.html`'s list+detail layout; a "🗺 Maps" link was added to `game.html`'s and `sessions.html`'s nav. World-prep's region-seeding pass (`world_prep_prompt.py`) also got an optional, best-effort instruction to draft a coarse settlement grid when the adventure text grounds one, so a town can already have a browsable layout before the party ever arrives — explicitly skipped (not invented) when the source text doesn't describe one. Site-scale room-by-room precomputation (individual dungeon rooms) was NOT added — bigger, separate future work, same "ship live-authoring first" call as the original PDF-auto-extraction stretch goal.

### Under Consideration

**4. Character portrait generation**
Generate a portrait from a player's physical description + race/class at the end of Session 0. Lives behind a swappable `backend/imagegen/` interface (mirrors the tool-registry pattern), stores to `static/portraits/{character_id}.png`, adds `Character.portrait_url`.

**5. Scene illustration generation**
Same backend as (4), triggered by an agent tool (`generate_scene_image(prompt)`) for key story beats. Reuses the existing `Handout` model (`handout_type=DRAWING`) to store and surface results.

*Feasibility for 4 & 5: Medium. Local image generation (Stable Diffusion via ComfyUI/A1111) needs a GPU and a separate service beyond Ollama's text models — but it could plausibly join `docker-compose.yml` as another container alongside `ollama` and `postgres`, same bind-mount pattern as `data/chroma_db/`, keeping the project's "no cloud services" stance intact. Not solving the infra question now — flagged here so it's not forgotten when this gets picked up. A cloud image API would be easier to prototype but breaks that design goal, so it's a fallback, not the default.*

**11. Clickable character cards in the Session 0 lobby — Done ✅ (2026-07-13)**
During a live game session, clicking a party member in `game.html`'s sidebar (`.party-member[data-char-id]`, `onclick="loadCharacterSheet(id)"`) fetches `GET /campaigns/{id}/party/{character_id}` and renders the full sheet into a side pane (`renderCharacterSheet()` → `#sheet-preview`). The Session 0 lobby (`session_zero_index.html`) had the equivalent party grid (`.sz-char-card`) but the cards were static — only a "Remove" button, no way to inspect a finalized character's full sheet the way you can mid-session.
*Shipped as: `renderCharacterSheet()`/`abilityRow()`/`itemLink()`/`rarityClass()`/`escapeHtml()` extracted verbatim out of `game.html`'s inline script into a new shared `static/character-sheet.js`, included by both templates instead of duplicated. `.sz-char-card` gained `data-char-id` + `onclick="loadSzCharacterSheet(id)"` (mirroring `game.html`'s pattern exactly), a `#sheet-preview` `<aside>` was added next to the party list, and a small `loadSzCharacterSheet()` reuses the shared renderer against the same `/party/{character_id}` endpoint — no backend changes. New CSS: `.sz-party-body` (flex row for list + sheet), `.sz-char-card.clickable/.active`, and a scoped `.sz-party-body .sz-sheet` override so the shared sheet-panel style fits the lobby's plain page layout instead of the full-height chat layout it was built for.*

**10. Homebrew content (per-campaign)**
Let a DM register custom rules/monsters/items scoped to a single campaign, without code edits and without polluting other campaigns' RAG results. Mirrors the existing `docs/source/adventures/{slug}/` pattern but keyed by `campaign_id` instead of an opt-in slug list: a new `docs/source/per_campaign_rules/{campaign_id}/` folder, indexed with metadata `source_type: "homebrew"`, `campaign_id`. `RulesStore.search()` gains a third `$or` branch — `{"campaign_id": {"$eq": campaign.id}}` — alongside the existing `core` and `adventure` branches, and is always active for its own campaign (no `books_in_play` opt-in needed).
*Feasibility: Medium — same indexing path as adventures, but needs a campaign_id metadata filter and an upload/management UI for the DM.*

**11. Intra-session rolling memory summarization — Done ✅ (2026-07-13)**
`_MAX_MESSAGES` (mechanics) and `_NARRATOR_MAX_TURNS` (narrator) are still hard cutoffs, but older narrative turns falling out of the narrator's window are no longer simply lost. `DMState` gained `session_recap: str` (the running recap) and `session_recap_through: int` (how many narrative turns are already folded in, so each update only sends the LLM the NEW turns that just crossed the boundary, not an ever-growing prefix). `_maybe_update_session_recap()` (`dm_agent.py`) is called once per player turn from `stream_response` (not per mechanics-loop iteration or per auto-resolved combatant turn) — a cheap early-exit when nothing's about to trim, otherwise a small LLM call (same `_get_model()`/timeout/fallback-on-failure shape as `summarize_session`, plus the same `_party_ground_truth` anti-fabrication grounding) that merges the new turns into a short 2-4 sentence recap. Both `_make_mechanics_modifier` and `_make_narrator_modifier` prepend it (right after the system message, before the trimmed/narrative window) wrapped in a `[SESSION RECAP — internal, not player dialogue...]` marker that explicitly tells the model to treat it as background color and call `get_character`/`get_current_location`/`get_party_status` for anything it needs to act on precisely — matching this file's existing "don't trust narrated state" doctrine (`_MECHANICS_BASE`, `build_session_kickoff_message`) rather than inventing new phrasing. Regression coverage: `tests/test_session_recap.py` (LLM and checkpointer both monkeypatched — exercises the trim/fold/merge decision logic, not real model output).

**12. Subclass mechanics modeling**
`Character.subclass` is a bare free-text string (already noted above for `level_up`) and `Character.features` is explicitly freeform — nothing validates a subclass against its class's real subclass list, and no data table encodes what a subclass actually *changes* mechanically (Ranger's Fey Wanderer bonus cantrip, Gloom Stalker's bonus first-round attack, Beast Master's companion rules, Hunter's combat options, etc.). In practice a subclass's rule alterations only exist if the model remembers and correctly applies them from freeform text each time they're relevant — the same shape of problem `SPELL_MENUS`/`SPELL_REQUIREMENTS` (spells.py) already solved for base-class cantrips/level-1 spells, just unaddressed for subclasses. Surfaced 2026-07-04 during a live Session 0 conversation — not the cause of that session's actual bug (an unrelated tool-call-fidelity failure; see the chargen.py Tools section's "Verified live..." narrative and Agent Architecture's "Session 0 agent" section for the fix, the same session's two-node mechanics/narrator restructure), but adjacent enough to flag while fresh.
*Feasibility: Medium-large — a `SUBCLASS_FEATURES` table mirroring `SPELL_MENUS`'s shape is straightforward for well-known feature names, but many subclass features are genuinely bespoke mechanics (a companion creature, a save-or-suck rider, a resource pool) rather than a consistent shape like "N spells from a list."*

**The name-validation slice — done ✅ (2026-07-13).** `SUBCLASSES: dict[str, list[str]]` (`backend/data/fivee_options.py`) extracts the real subclass names straight out of each class's own `level_3_features` string above (e.g. Fighter → Battle Master/Champion/Eldritch Knight/Psi Warrior; Warlock is a special case — its subclass IS the level-1 Otherworldly Patron choice, same 4 names). New `validate_subclass(char_class, subclass)` (`_helpers.py`, mirrors `build_spells_known`'s shape/return convention) does a case-insensitive check, returning the canonical-cased name on a match or a corrective error listing the real options — deliberately soft, not a hard universe: an unset subclass, or a class/homebrew subclass this table doesn't cover, passes through unchanged. Wired into `finalize_character` (`chargen.py`), `generate_companion_character` (`companion.py`), and `level_up` (`levelup.py`, validated *before* any stat mutation — a rejected level-up leaves the character completely untouched, same discipline the spell-selection check there already follows). Regression coverage: `tests/test_subclass_validation.py`.

**The level-3 mechanical-modeling slice — done ✅ (2026-07-13), higher levels still open.** `SUBCLASS_FEATURES: dict[str, dict[str, dict[int, list[str]]]]` (`fivee_options.py`) transcribes each of the 48 real subclasses' actual level-3 feature text from `docs/source/core/D&D 5.5E - Player's Handbook.md`, condensed to the same terse freeform-string style as `CLASSES[cls]["level_1_features"]` — scope deliberately limited to level 3 (the universal subclass-unlock level), same incremental-slice precedent as the name-validation table itself; levels 6/7/9/10+ are a documented follow-up, not attempted here, since many later features are genuinely bespoke (resource pools, summoned companions, save-or-suck riders) rather than a uniform shape. A companion `SUBCLASS_BONUS_SPELLS` table covers the small subset of level-3 grants that are both "always prepared" spell lists AND resolve against a spell already in `ALL_SPELLS`' curated cantrip/level-1 subset (Life Domain, Oath of the Ancients, Aberrant Sorcery, Archfey Patron) — most subclass spell-list grants name spells outside that curated set and aren't modeled, same "don't claim to be the only universe" caveat `SPELL_MENUS` already carries. New `apply_subclass_features(char)` (`_helpers.py`) appends any newly-unlocked feature text onto `Character.features` and any resolvable bonus spells onto `spells_known`/`spells_prepared`, idempotently (safe to call again on a re-level). Wired into `finalize_character`, `generate_companion_character`, and `level_up` (which now also reports "New subclass features" in its resolution text when a level-up crosses into one). Regression coverage: `tests/test_subclass_mechanics.py`.

**13. Mass combat / mob rules for large enemy groups** (idea from user, 2026-07-05)
Surfaced while adding the mechanics prompt's turn-auto-continuation rule (see Agent Architecture — the mechanics model now resolves every non-player combatant's turn in a row within one response, stopping only once initiative comes back to a player-controlled character). A large hostile group (many individual monsters) queued between two of the player's own turns can burn a lot of the per-message `recursion_limit=60` LangGraph step budget (`backend/main.py`, ~2 graph steps per combatant round-trip) in a single reply. Checked the currently indexed core rulebooks (2024 PHB, DMG) for an official mass-combat/mob rule to ground this against — not present in `docs/source/core/`. The well-known version is the unofficial community "mob rule" (one attack roll for a mob of N identical creatures, with a to-hit/damage bonus scaling by group size) — not an indexed sourcebook rule, so it'd need to be flagged as a DM improvisation the same way homebrew monster stats already are, unless a book containing it gets indexed later.
*Prototype sketch: likely a `mob` flag (or new `CombatantType`) on `Monster`/`InitiativeEntry` representing a group as a single initiative slot with a `count`, resolved with one roll per mob turn (scaled damage/to-hit) instead of N individual `resolve_attack` calls — sidesteps the step-budget risk entirely rather than just raising `recursion_limit`.*
*Feasibility: Medium — no urgent trigger yet (most encounters are small enough that the step budget isn't a real risk); revisit if the recursion limit is actually hit in play, or if encounters routinely run 10+ hostile combatants.*

**14. Freeform narrative time advancement — Done ✅ (2026-07-13)** (idea from user, 2026-07-09)
Reported live: the DM narrated dusk falling, but the sidebar's World clock (`campaign.time_of_day`) still read "morning" — the underlying field was never updated to match. Root cause: `time_of_day`/`days_elapsed` only ever advanced through `travel_to` and the two rest routes — there was no tool, no prompt guidance, and no guardrail for a non-travel time-skip (a stakeout, an evening in town, "a week passes"). Shipped the same three-layer pattern the loot guardrail already established: (1) two new tools, `advance_time(hours, reason)` and `take_rest(kind)` (`backend/tools/world.py`, thin wrappers over the existing `advance_clock`/`apply_long_rest`/`apply_short_rest` helpers — no changes to the deterministic rest math itself); `take_rest` closes the other real gap here too — rests were previously UI-button-only, with no way for the model to apply them when narrating the party making camp as part of the story. (2) A new "Time passage" prompt section (`prompts.py`, parallel to the existing Travel section, not folded into it) giving the model concrete anchors for estimating how much time a narrated scene covers — sub-minute actions need no call, an hour-plus scene calls `advance_time`, an explicit skip always does, and actual resting/making camp calls `take_rest` — so the model can call these proactively rather than relying solely on the backstop. (3) `_detect_missing_time_advance_followup` (`dm_agent.py`, own `time_guardrail_count` budget, same starvation-avoidance shape as `lore_guardrail_count`/`stalled_turn_guardrail_count`) — a regex over the resolution report for time-skip/resting language with no `advance_time`/`take_rest`/`travel_to` call backing it, deliberately un-gated from combat, same reasoning as `_detect_missing_loot_followup`. Regression coverage: `tests/test_freeform_time_and_rests.py`.

**15. Feats / Ability Score Improvement at level-up**
`level_up` (`levelup.py`) currently has no feat-or-ASI choice at the levels 2024 rules grant one — a character just gains HP/proficiency bonus/class features with no player decision point. Needs a small `FEATS` data table (mirrors `SUBCLASS_FEATURES`'s shape: name → prerequisite/effect text) and a new `level_up` param/tool step to pick one, or take the flat +2/+1/+1 ASI instead.
*Feasibility: Medium — the data table is straightforward transcription work (same discipline as `SPELL_MENUS`/`SUBCLASS_FEATURES`), but many feats have bespoke mechanical effects (not a uniform "+N to a skill" shape) that would need individual handling wherever they matter (e.g. Lucky's reroll, Alert's initiative bonus).*

**16. Spell content beyond level 1**
`backend/data/spells.py`'s `ALL_SPELLS`/`SPELL_MENUS`/`SPELL_REQUIREMENTS` only cover cantrips and level-1 spells — Ranger/Sorcerer/Bard/Warlock/Wizard/Cleric/Druid's higher-level spell-slot progression and 2nd-level+ spell selection is entirely unmodeled today. A natural follow-on to the existing interactive spell-selection system (see chargen.py's Tools section above), same transcription-and-validation approach.
*Feasibility: Medium-large — mechanically it's the same shape already solved for level 1 (curated menu + flat per-tier counts + `cast_spell`'s `resolution_type` dispatch), but the sheer number of additional spells to transcribe and verify against the source PHB text is a much bigger lift than level 1 was.*

**17. NPC daily schedules**
NPCs currently sit static wherever they were created/placed — a shopkeeper is always "at the shop," a guard is always "on the wall," regardless of `time_of_day`. A simple time-of-day routine (open/closed hours, a patrol location swap) would make `advance_time`/`take_rest` (idea 14, already shipped) feel consequential to the world rather than just a clock number ticking over.
*Feasibility: Medium — needs a small schedule data shape on `NPC` (e.g. a list of `(time_range, location)` pairs) and a check wired into `get_current_location`/`get_npc`, but no new subsystem; can start with only NPCs that matter (shopkeepers, quest-givers) rather than every NPC in the campaign.*

**18. Downtime activities**
Crafting, training, carousing, running a business, and similar between-adventure activities have real rules in the 2024 DMG. This hooks naturally into the existing world clock (`Campaign.days_elapsed`, `advance_time`) rather than needing a new time-tracking concept.
*Feasibility: Medium — the DMG's downtime rules are a real, boundable rule set (not open-ended homebrew), but modeling activity outcomes (crafting progress, business income, carousing complications) is more state than a single tool call; likely its own small subsystem.*

**19. Dynamic shops**
Buying/selling against a real per-settlement stock list (with prices, limited quantities) instead of ad hoc narrated purchases the model currently has to invent on the spot. Pairs naturally with the existing `Container`/currency model already used for loot and the party treasury.
*Feasibility: Medium — the data model is simple (a `Container`-shaped inventory per shop, keyed to a `Location`), but populating realistic per-settlement stock (grounded vs. invented) raises the same "don't invent, ground or abstain" question this app already applies to loot and world-prep.*

**20. Encounter-budget guardrail**
`Encounter.xp_budget` already exists as a field but nothing currently checks a generated fight against the real DMG XP budget/thresholds for the actual party (level, size). A deterministic warn-or-block check when a `start_encounter`/`create_monster` combo is wildly over or under budget fits this repo's existing "never trust the model for arithmetic it can get wrong" principle — same spirit as the loot, time-advancement, and turn-order guardrails already shipped (see Tools section above).
*Feasibility: High — pure arithmetic against already-modeled fields (`Encounter.xp_budget`, party levels, monster CR/XP), no new data needed; same shape as the deterministic rest routes.*

**21. In-combat tactical hint mode**
An opt-in toggle where the narrator surfaces a plain-language suggestion ("your reaction is available; Shield would help here," "you're both flanking — consider Help") for newer players, without touching the mechanics layer's actual resolution logic at all — purely a narrator-prompt addition gated behind a campaign setting.
*Feasibility: Medium — the mechanics layer already tracks the exact state a hint would need (`reaction_available`, positions, HP); the work is mostly prompt design plus a per-campaign toggle field, not new state.*

**22. Milestone vs. XP leveling toggle**
A small DM-facing campaign preference: award levels at story milestones instead of `end_encounter`'s XP-based awarding — a real 2024 DMG-supported alternative leveling scheme, not homebrew.
*Feasibility: High — a boolean `Campaign` field and a branch in the leveling-trigger logic; XP can still be tracked/displayed even when it's not what actually triggers a level-up.*

**23. Campaign keepsake export**
A compiled PDF/ebook of an entire campaign — all session chronicles, final character sheets, and discovered maps bound together as a keepsake. Bigger in scope than the already-listed single-session markdown/PDF export (see "Deferred" bullets above) — this is a whole-campaign document, not a per-session one.
*Feasibility: Medium — the underlying data (sessions, characters, maps) all already exists and is queryable; the new work is a rendering/layout pass (likely HTML-to-PDF) plus assembling it into one coherent document rather than separate exports.*

**24. Accessibility pass** (long-horizon — not near-term)
Screen-reader labeling for the map/fog-of-war grid (currently a CSS-grid of colored cells with no semantic markup) and a colorblind-safe palette option for `map_render.py`'s symbol classification. Flagged here mainly so it isn't forgotten, not because it's next in line.
*Feasibility: Medium-large — the map rendering path would need real semantic markup (ARIA labels per cell or a text-equivalent description) added without disrupting the existing visual layout, plus a second palette mode threaded through `classify_symbol()`'s CSS classes; a genuinely separate project from the nearer-term ideas above, not a quick add-on.*

### Deferred from the combat resolution refactor (2026-07-03)

Everything below was explicitly scoped out of the `resolution.py`/reaction-system/
`Spell`-schema work landed this date, each for a stated reason — not oversights.
Collected here in one place rather than left scattered across old plan-file prose.

**NPC combatants can't take damage — fixed 2026-07-13.** `resolve_attack`/
`resolve_pending_action`/`cast_spell`'s attack-roll and automatic-effect paths only
resolved `Character`/`Monster` — there was no `update_npc_hp` tool anywhere, despite
`NPC.combat_stats` (`CombatStatBlock`) and `CombatantType.NPC` clearly anticipating
NPCs fighting. Turned out to be a bit more than the originally-estimated ~10 lines,
since `Character`/`Monster` both expose `ac`/`current_hp`/`max_hp`/`attacks`/
`ability_scores` at the top level while `NPC` nested them under `combat_stats` — every
generic attacker/target code path in `resolution.py` reads those names directly off
whatever it's given. Fixed by adding matching read-only `@property` proxies to `NPC`
(`models.py`) that delegate to `combat_stats` (defaulting to inert values — ac 10, 0
HP, no attacks — when `combat_stats` is `None`, though `find_combatant` below never
hands out a combat_stats-less NPC in the first place), rather than threading an
NPC-specific branch through every call site. Added: `apply_damage_to_npc` (mirrors
`apply_damage_to_character`/`_monster` exactly, `_helpers.py`); `find_combatant()`
(3-way `find_char`/`find_npc`/`find_monster` lookup, only returns an NPC if
`combat_stats is not None`) and `apply_damage_to_combatant()` (isinstance dispatch)
to replace the repeated `find_char(...) or find_monster(...)` + Character/Monster
ternary pattern at every attack/automatic-damage site in `resolution.py`; a new
`update_npc_hp` tool (`combat.py`, mirrors `update_monster_hp`) for freeform NPC
damage/healing outside a resolved attack; and a fix to `start_encounter`'s initiative-
modifier lookup (`combat.py`), which previously always reported an NPC combatant as
"not found in campaign" and rolled it at +0 DEX regardless of its real stats.
**Follow-up closed same day:** `resolve_saving_throw`/`resolve_check`/`cast_spell`'s
saving-throw branch also now resolve NPCs for real, not just attacks. The gap wasn't
actually a proxy-vs-real distinction to begin with — `saving_throw_bonuses`/
`skill_bonuses`/`conditions` just didn't exist anywhere on `NPC`/`CombatStatBlock` yet.
Added all three to `CombatStatBlock` (same dict/list shape Monster already uses —
"only list what this NPC is actually proficient in," same convention `_save_bonus`/
`resolve_check` already followed for Monster) plus matching proxy properties on `NPC`
(`conditions` returns `combat_stats`' own list object, not a copy, so `.append()` at
the call site mutates the real persisted list). `find_combatant` (not the old
`find_char`/`find_monster` pair) now backs `resolve_saving_throw`, `resolve_check`,
and `cast_spell`'s `SAVING_THROW` branch, same as the attack-roll paths above — an NPC
with `combat_stats.saving_throw_bonuses={"dexterity": 5}` and `skill_bonuses=
{"deception": 4}` now rolls its own real dex save and deception check instead of
either being invisible to these tools or silently falling back to a bare ability mod.
Verified live: an NPC stat-blocked with real save/skill overrides rolled both
correctly, a failed save applied damage AND a condition, and the condition landed in
`combat_stats.conditions` (not lost). Full pytest suite still green throughout.
Permanent regression coverage for all of this (damage as attacker/target, freeform
`update_npc_hp`, the no-`combat_stats` refusal path, save/check overrides, condition
persistence, and `start_encounter`'s initiative fix): `tests/test_npc_combat.py`.

**Time-windowed resurrection magic (Revivify, Raise Dead, ...).** Death saves
(`resolve_death_save`) and the "ordinary healing can't revive a truly-dead character"
guard in `apply_damage_to_character` (both wired up 2026-07-03, verified live — both
the death and stabilize paths were observed through real model turns, landing exactly
on the mechanically-implied outcome each time) correctly distinguish dead
(`death_save_failures >= 3`) from merely downed/unconscious, and correctly stop
ordinary Cure-Wounds-style healing from working on the dead — but there's no dedicated
revival path yet, and no way to enforce a spell like Revivify's "within 1 minute of
death" window, since nothing currently records *when* a character died. Needs a
`died_at` timestamp (or a round/turn counter) on `Character`, plus a tool that bypasses
the ordinary-healing dead-block specifically for a real resurrection spell and checks
that window. A dead character does already correctly stay in `campaign.party`
indefinitely today (no tool anywhere removes a party member), so a future revival tool
has something to target — that part needs no fix.

**Opportunity attacks (movement-triggered reactions) — Done ✅ (2026-07-13).** Found
live during testing 2026-07-03, and higher-priority than Counterspell/Absorb-Elements
below since it's a *universal* reaction every combatant with `reaction_available` has
by default (not a special spell/feature) and is likely the single most common reaction
in actual 5e play. `has_plausible_reaction()` only modeled "does this character have a
reaction *spell or feature*" — it had no concept of "a hostile creature just left my
reach without disengaging."

A cheap fix was proposed first — trigger off `ZoneType`'s existing melee/adjacent/
near/far/distant abstraction whenever a combatant's zone transitions away from
`melee` — and explicitly rejected: zone is a single abstract value per combatant, not
pairwise, so it can't answer "who exactly was adjacent" when multiple combatants share
a zone, meaning it would need a compounding pile of guardrails/prompt instructions to
patch edge cases. Built the real thing instead, once item 1 above gave the app actual
`(x, y)` positions: `check_opportunity_attacks()` (`_helpers.py`) — real Chebyshev
distance (5e diagonal-counts-as-5ft) against each reacting combatant's own real reach
(see below) — wired into `set_combatant_position` (`combat.py`).
`resolve_opportunity_attack()` (`resolution.py`) rolls the swing immediately (the
attacker's own reaction isn't a decision point this app models — hostile creatures
already resolve automatically elsewhere) and, on a hit against a player-controlled
`Character` with a reaction available, pauses using the exact same `PendingAction`
shape `resolve_attack` itself already produces — **no changes needed to
`resolve_pending_action`/`resolve_pending_action_impl` at all**, since that machinery
was already generic over which call site created the pending action. Verified live
end-to-end, including the reaction-pause path (a mover with Shield known correctly
pauses, then resolves via the unmodified `resolve_pending_action`). Regression
coverage: `tests/test_location_grids_and_opportunity_attacks.py`.

**Three follow-up gaps closed same day (2026-07-13):**
- **Reach weapons.** `Attack.reach_ft: int = 5` (`models.py`) — 10 for a real reach
  weapon (Glaive/Halberd/Lance/Pike, added to `WEAPONS` with the `"reach"` property;
  `weapon_reach_ft()` in `equipment.py` derives it) or a monster/NPC with genuinely
  long reach (`create_monster`'s `attacks` dicts now accept an optional `reach_ft`).
  Wired everywhere an `Attack` gets built from a real weapon lookup — chargen's
  `_starting_equipment`, `add_weapon_attack`, `create_magic_item` (all in
  `_helpers.py`/`party.py`). `check_opportunity_attacks` now compares against the
  reacting combatant's own `attacks[0].reach_ft // 5` squares instead of a flat 1.
- **NPCs excluded from opportunity attacks.** Real gap, not a guess this time:
  `InitiativeEntry.side: str = "hostile"` (`models.py`) — set from `start_encounter`'s
  new per-combatant `"side": "party"|"hostile"` key, defaulting by `combatant_type`
  when omitted (character → party, monster/npc → hostile) — an NPC fighting *for* the
  party must have `"side": "party"` passed explicitly. `_is_hostile_pair()`
  (`_helpers.py`) now checks `side` instead of `combatant_type`. Surfaced a second,
  genuinely separate bug while wiring this in: `NPC`/`CombatStatBlock` had no
  `reaction_available` field at all (unlike `Character`/`Monster`, which both have
  one) — an NPC could never qualify as an attacker or a reaction-eligible target no
  matter what `side` said. Added `CombatStatBlock.reaction_available: bool = True`
  plus a read/write `NPC.reaction_available` property (needs a setter, unlike the
  other `NPC` proxies, since `resolve_pending_action_impl` writes `target.reaction_available
  = False` after a declared reaction).
- **Only the first qualifying attacker resolved.** Turned out to be a one-line fix:
  `resolve_opportunity_attack`'s `eligible_for_pause` check already required
  `not enc.pending_action`, so looping over *every* qualifying attacker (instead of
  just `qualifying[0]`) means the first hit against an eligible target still pauses
  correctly, and every attacker after that just resolves immediately (rolls, applies
  damage) since the pause slot is already taken — nobody is silently skipped anymore,
  and no new queuing logic was needed.

**Downed-character narration lagging the mechanics.** Also found live 2026-07-03: when
`resolve_attack` returns PENDING, the narrator sometimes describes the hit as already
landed ("the blade catches you... leaving a stinging heat") before the player has
decided whether to react — mechanically correct (no damage applied, `pending_action`
set) but narratively jumping ahead of the still-open decision. The PENDING-result
prompt instruction was tightened the same day to explicitly require describing an
attack as incoming-and-unresolved rather than landed; worth re-verifying live that this
actually fixed the phrasing rather than just narrowed it.

**Counterspell.** Its trigger point ("a creature begins casting a spell") is earlier
than anything any tool represents — spellcasting is narrated, then resolved after the
fact via whatever `resolve_attack`/`resolve_saving_throw`/`cast_spell` calls apply.
A real trigger needs a `declare_spell_cast` pre-resolution step that *every*
spellcasting turn, not just attacks, would have to route through — materially larger
scope than a single interrupt type. `PendingAction.trigger_type` (currently only
`"incoming_attack"`) is left as a free string specifically so a `"spell_cast"` value
can be added later without a schema migration.

**Absorb-Elements-style reactions on `resolve_saving_throw`.** That tool resolves N
targets atomically in one call by design; a single-slot `PendingAction` can only pause
on one thing at a time. Whether a multi-target pause should defer *all* targets or
only offer the window to the first eligible one is a real design fork that deserves
its own pass. `trigger_type="incoming_save_damage"` is reserved for this.

**Spell data population — superseded, done.** This item (written before 2026-07-03)
described `chargen.py` never setting `spells_known`/`spellcasting_ability`/etc. and
sketched three tiers of fix, up to and including a full interactive Session-0
spell-choice step as the ambitious end state. That end state has since shipped in
full — see "Interactive spell selection (2026-07-03)" under `chargen.py` above:
`SPELL_MENUS`/`SPELL_REQUIREMENTS` (`spells.py`), real per-player spell choice via
`update_character_draft('spells_known', ...)`, validated at `finalize_character`, plus
`generate_companion_character` and a `backfill_character_spells.py` for characters
created before this landed. Left here only as a pointer so this stale paragraph
doesn't get mistaken for open work again.

**Verbal/Somatic component enforcement.** `cast_spell` (`resolution.py`) checks Material
("M") components as of 2026-07-04 — refuses if the spell needs a focus/pouch (or, for a
costly named component like "a diamond worth 300gp", that specific inventory item) and
the caster doesn't have one (`_material_requirement`/`_has_focus_or_pouch`/
`_has_named_material`). Verbal and Somatic requirements are still completely
unenforced: nothing checks whether a caster is gagged/silenced (V) or has a hand free/
isn't bound (S) before letting a cast through. Surfaced live (2026-07-04) via a captured
party in Out of the Abyss — manacled hands are exactly the kind of situation that should
block S-component spells, and currently doesn't. Would need a way to represent "hands
bound" / "unable to speak" on a `Character` (the existing `ConditionType` enum doesn't
have a clean fit — 5e's real Restrained condition doesn't map to "manacled but full
speed," as already noted in this campaign's captivity note) and a check in `cast_spell`
mirroring the new material-component one.

**`use_spell_slot` removal from `party.py`.** Only safe once `cast_spell` is the
exclusive path for slot consumption — gated on the spell-data-population item above,
since `use_spell_slot` is still the only way to track slot use for a spell cast via
the ungrounded `resolve_attack` fallback or a spell narrated without any tool call.

**Formal action-economy engine.** A structured `ActionOption` model
(`name`/`kind`/`slot`, linking back to an `Attack`/`Spell` entry) with a
`list_available_actions(char)` **live-computed view** (deliberately not a persisted
field — storing it risks the same staleness/dual-source-of-truth bug class this
session hit more than once, e.g. a new `Attack` from `create_magic_item` not
appearing in a cached list) would let the mechanics model see "my available actions
are: sword attack, cast spell, dash, dodge, ..." directly, and `action_used_this_turn`
/ `bonus_action_used_this_turn` flags would be the natural sibling of
`Character.reaction_available` (reset wherever `advance_combatant_turn` already resets
it). `has_plausible_reaction()` (`_helpers.py`) is already a special case of this —
exactly `list_available_actions(char)` filtered to a reaction slot — so it's the seed
of the general version, not throwaway. A future `resolve_action` router would dispatch
to the *same* `resolve_attack`/`resolve_saving_throw` this pass built, not replace
them. Deliberately not built now: it doesn't reduce tool-call count for the actions
that were never the source of the bloat (Dodge/Help/Ready already cost ~0-1 calls
today), and it needs `features` converted from freeform prose into structured data
across chargen, companion generation, and `fivee_options.py` — real content modeling,
a different project from a tool-schema change.
