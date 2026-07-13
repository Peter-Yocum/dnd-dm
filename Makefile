.PHONY: up down build restart logs psql migrate migration rollback shell fresh index index-if-empty setup extract-lore reindex-full recontextualize backfill-history-chunks eval-retrieval backfill-lore-links seed-relation-graph setup-venv ingest-book-native merge-chroma load-lore-json test

## ── Services ──────────────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build --no-cache

## Rebuild the app image and restart only that container (db keeps running).
restart:
	docker compose up -d --build app

logs:
	docker compose logs -f app

db-logs:
	docker compose logs -f db

## ── Database ──────────────────────────────────────────────────────────────────

migrate:
	docker compose exec app alembic upgrade head

## Usage: make migration name=add_users_table
migration:
	@test -n "$(name)" || (echo "Usage: make migration name=describe_the_change" && exit 1)
	docker compose exec app alembic revision --autogenerate -m "$(name)"

rollback:
	docker compose exec app alembic downgrade -1

psql:
	docker compose exec db psql -U dnd_dm -d dnd_dm

## ── Development ───────────────────────────────────────────────────────────────

shell:
	docker compose exec app bash

## Run the BDD-style tool-layer test suite (tests/) inside the app container.
## Runs against the same dev Postgres the app uses — no separate test DB —
## every test cleans up its own scratch campaign row. Usage: make test
## [k="expression"] to filter (pytest -k).
test:
	docker compose exec app python -m pytest -v $(if $(k),-k "$(k)",)

## Tear down all containers + volumes, start fresh, and run migrations.
fresh:
	docker compose down -v
	docker compose up -d
	@echo "Waiting for services..."
	@until docker compose exec app alembic upgrade head 2>/dev/null; do \
		printf '.'; sleep 2; \
	done
	@echo " done."

## Build/refresh the Postgres/pgvector rules index from docs/source/*.md
## inside the container. Resumable: safe to Ctrl-C/kill and re-run —
## already-indexed chunk_ids are skipped, so a re-run only does remaining
## work. This is also the correct way to RESUME an interrupted `make
## reindex-full` — do NOT repeat reindex-full itself, which would re-wipe and
## lose progress. Scope to one book at a time with adventure= (adventure
## folder slug) or book= (core rulebook, exact filename stem) + source_type=core.
## Incremental by default (existing chunks in scope are left alone); pass
## fresh=1 to delete that scope's existing chunks first (only needed after a
## chunking-schema change — see build_index.py --fresh's help for why this
## isn't the default). skip_context=1 skips the LLM contextualization pass
## (fast dev path).
## Usage: make index adventure="Curse of Strahd"
##        make index book="D&D 5E - Monster Manual" source_type=core
##        make index book="D&D 5E - Monster Manual" source_type=core fresh=1
index:
	docker compose exec app python scripts/build_index.py \
		$(if $(adventure),--adventure "$(adventure)",) \
		$(if $(book),--book "$(book)",) \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(skip_context),--skip-contextualization,) \
		$(if $(fresh),--fresh,)

## One-time full corpus rebuild under the parent/child + contextual-
## augmentation chunk schema. Expect a long multi-hour/multi-day first run
## (many thousands of child chunks, one local LLM call each) — safe to leave
## running overnight. If interrupted, resume with `make index` (NOT by
## re-running this target, which would re-wipe the collection).
reindex-full:
	docker compose exec app python scripts/build_index.py --wipe

## One command for the full nightly per-book pipeline: reindex (with
## contextualization) THEN extract lore/monsters, in sequence. Same params
## as `index`/`extract-lore` combined — adventure= or book=+source_type=core,
## plus kinds= (passed to extraction only, e.g. kinds=monster for the
## Monster Manual) and skip_context=1 (passed to reindexing only). Both
## steps show a live tqdm progress bar. Resumable — safe to Ctrl-C/kill and
## re-run, same as running the two steps separately.
## Usage: make ingest-book adventure="Curse of Strahd"
##        make ingest-book book="D&D 5E - Monster Manual" source_type=core kinds=monster
ingest-book:
	@test -n "$(adventure)$(book)" || (echo "Usage: make ingest-book adventure=\"Name\" OR book=\"Core Book\" source_type=core [kinds=...] [skip_context=1] [fresh=1]" && exit 1)
	@echo "=== [1/2] Reindexing: $(or $(adventure),$(book)) ==="
	docker compose exec app python scripts/build_index.py \
		$(if $(adventure),--adventure "$(adventure)",) \
		$(if $(book),--book "$(book)",) \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(skip_context),--skip-contextualization,) \
		$(if $(fresh),--fresh,)
	@echo "=== [2/2] Extracting lore/monsters: $(or $(adventure),$(book)) ==="
	docker compose exec app python scripts/extract_entities.py --write-postgres \
		--book "$(or $(adventure),$(book))" \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(kinds),--kinds "$(kinds)",)
	@echo "=== Done: $(or $(adventure),$(book)) ==="

## Resumable LLM-contextualization pass over rule_chunks that were indexed
## without it (contextualized=false — e.g. an initial reindex run with
## skip_context=1 to unblock live play quickly). Safe to run whenever you
## want, Ctrl-C/kill at any point, and re-run later — it only ever picks up
## rows still missing contextualization, never redoes finished ones and
## never wipes anything. Scope to one book/adventure/source-type at a time
## the same way `index` does. Run UNSCOPED (no adventure=/book=/source_type=)
## and it does core rulebooks FIRST, then adventures, as two sequential
## phases — core is the higher-priority corpus (used by every campaign),
## so a kill after phase 1 still leaves all of core done before any
## adventure work starts.
## Usage: make recontextualize                        # core first, then adventures
##        make recontextualize adventure="Curse of Strahd"
##        make recontextualize source_type=core
ifeq ($(adventure)$(book)$(source_type),)
recontextualize:
	@echo "=== [1/2] Recontextualizing core rulebooks ==="
	docker compose exec app python scripts/build_index.py --recontextualize --source-type core \
		$(if $(context_model),--context-model "$(context_model)",)
	@echo "=== [2/2] Recontextualizing adventures ==="
	docker compose exec app python scripts/build_index.py --recontextualize --source-type adventure \
		$(if $(context_model),--context-model "$(context_model)",)
else
recontextualize:
	docker compose exec app python scripts/build_index.py --recontextualize \
		$(if $(adventure),--adventure "$(adventure)",) \
		$(if $(book),--book "$(book)",) \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(context_model),--context-model "$(context_model)",)
endif

## Re-embed existing session chronicles (all campaigns) into the new
## per-event chunk schema. Idempotent — safe to re-run.
backfill-history-chunks:
	docker compose exec app python scripts/backfill_history_chunks.py

## Run the hand-labeled retrieval recall@k eval against the current index.
## Usage: make eval-retrieval [baseline=1] [k=8]
eval-retrieval:
	docker compose exec app python scripts/eval_retrieval.py $(if $(baseline),--baseline,) $(if $(k),--k $(k),)

## Only index if rule_chunks is empty (safe to run on first clone). Was a
## data/chroma_db/ empty-directory check before the 2026-07-12 Postgres
## migration — that directory is never populated anymore, so the old check
## was always true and silently forced a full reindex on every `make setup`.
## Requires migrate to have run first (rule_chunks must exist to query it).
index-if-empty:
	@count=$$(docker compose exec -T db psql -U dnd_dm -d dnd_dm -tAc "SELECT count(*) FROM rule_chunks;" 2>/dev/null || echo 0); \
	if [ "$$count" = "0" ]; then \
		echo "rule_chunks empty — building index (this takes a while)..."; \
		docker compose exec app python scripts/build_index.py; \
	else \
		echo "rule_chunks already populated ($$count rows) — skipping. Run 'make index' to force rebuild."; \
	fi

## First-time setup on a new machine: migrate DB + index if needed.
setup: migrate index-if-empty

## Precompute the canon Lore Registry (NPC/Location/Item/Monster entities,
## aliases, source data) for one book. Resumable per-entity: safe to
## Ctrl-C/kill and re-run — continues from the last completed entity rather
## than restarting the book (see scripts/extract_entities.py).
## Usage: make extract-lore book="Curse of Strahd"
##        make extract-lore book="D&D 5E - Monster Manual" source_type=core kinds=monster
extract-lore:
	@test -n "$(book)" || (echo "Usage: make extract-lore book=\"Name\" [source_type=adventure|core] [kinds=npc,location,item,monster]" && exit 1)
	docker compose exec app python scripts/extract_entities.py --book "$(book)" --write-postgres \
		--source-type "$(or $(source_type),adventure)" $(if $(kinds),--kinds "$(kinds)",)

## Backfill lore_entity_id/aliases/source_chunk_ids/spoiler_tier onto an
## existing campaign's already-created NPCs/Locations/Items, fuzzy-matched
## against the Lore Registry. Idempotent — only touches records with
## lore_entity_id IS NULL; never touches attitude/is_alive/notes/quantity.
## Always dry-run first on a real campaign.
## Usage: make backfill-lore-links campaign_id=<uuid> [dry_run=1]
backfill-lore-links:
	@test -n "$(campaign_id)" || (echo "Usage: make backfill-lore-links campaign_id=<uuid> [dry_run=1]" && exit 1)
	docker compose exec app python scripts/backfill_npc_lore_links.py --campaign-id "$(campaign_id)" $(if $(dry_run),--dry-run,)

## Seed/refresh a campaign's incremental relation graph (entity_relations)
## from its already-loaded NPC/Location/Item data. No LLM calls — pure Python
## derivation from Postgres, safe to re-run any time (set-merge upsert via
## unique constraint, never duplicates).
## Usage: make seed-relation-graph campaign_id=<uuid> [dry_run=1]
seed-relation-graph:
	@test -n "$(campaign_id)" || (echo "Usage: make seed-relation-graph campaign_id=<uuid> [dry_run=1]" && exit 1)
	docker compose exec app python scripts/seed_relation_graph_from_existing.py --campaign-id "$(campaign_id)" $(if $(dry_run),--dry-run,)

## ── Native bulk ingestion (no Docker) ──────────────────────────────────────────
## This section runs scripts directly via a host Python venv instead of
## `docker compose exec app`. Two independent reasons this exists:
##   1. A desktop with no virtualization enabled can't run Docker at all.
##   2. Confirmed live on the laptop: routing a heavy bulk read (merge_chroma.py
##      pulling embeddings for tens of thousands of chunks) through
##      `docker compose exec` OOM-killed the container — it shares its memory
##      ceiling with the already-running `uvicorn --reload` app server, and
##      this whole Docker Desktop VM has under 1GB total. build_index.py's own
##      small (8-chunk) batches don't hit this in practice, but merge_chroma.py
##      and load_lore_json.py always run natively for exactly this reason.
## See docs/engineering-notes/desktop-native-ingestion.md for full setup.

## Create/refresh the native venv used for host-side bulk ingestion (OCR,
## indexing, entity extraction, merge scripts) — NOT the Docker app image.
## Cross-platform (Mac/Linux/Windows, detected via $(OS)). Installs the same
## requirements.txt the Docker image uses. MinerU (OCR) is a deliberately
## SEPARATE install — its GPU backend (MLX vs. CUDA) is platform-specific,
## see docs/engineering-notes/desktop-native-ingestion.md.
## Usage: make setup-venv
setup-venv:
ifeq ($(OS),Windows_NT)
	python -m venv .venv
	.venv/Scripts/python.exe -m pip install --upgrade pip
	.venv/Scripts/python.exe -m pip install -r requirements.txt
	@echo "Venv ready at .venv\Scripts\python.exe"
else
	python3 -m venv .venv
	./.venv/bin/python -m pip install --upgrade pip
	./.venv/bin/python -m pip install -r requirements.txt
	@echo "Venv ready at ./.venv/bin/python"
endif
	@echo "OCR (MinerU) is a separate install — see docs/engineering-notes/desktop-native-ingestion.md"

## Native equivalent of `ingest-book` — reindex + extract lore/monsters for
## one book, via the host venv instead of Docker. Defaults to NOT writing to
## Postgres (write_postgres=1 to opt in, e.g. if you set up a native Postgres
## on this machine too) — a machine with no Docker typically has no Postgres
## either, so extraction just produces its JSON registry (the normal debug/
## audit artifact); sync that back to the canonical machine and run
## `make load-lore-json` there. context_model=/model= override the
## contextualization/extraction model (see build_index.py --context-model's
## help for why NOT to silently fall back to the Docker default here).
## Usage: make ingest-book-native book="D&D 5.5E - Player's Handbook" source_type=core context_model=gemma4:e4b model=gemma4:e4b
ingest-book-native:
	@test -n "$(adventure)$(book)" || (echo "Usage: make ingest-book-native adventure=\"Name\" OR book=\"Core Book\" source_type=core [kinds=...] [context_model=...] [model=...] [write_postgres=1] [fresh=1]" && exit 1)
	@echo "=== [1/2] Reindexing (native): $(or $(adventure),$(book)) ==="
	./.venv/bin/python scripts/build_index.py \
		$(if $(adventure),--adventure "$(adventure)",) \
		$(if $(book),--book "$(book)",) \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(context_model),--context-model "$(context_model)",) \
		$(if $(skip_context),--skip-contextualization,) \
		$(if $(fresh),--fresh,)
	@echo "=== [2/2] Extracting lore/monsters (native): $(or $(adventure),$(book)) ==="
	./.venv/bin/python scripts/extract_entities.py \
		--book "$(or $(adventure),$(book))" \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(kinds),--kinds "$(kinds)",) \
		$(if $(model),--model "$(model)",) \
		$(if $(write_postgres),--write-postgres,)
	@echo "=== Done: $(or $(adventure),$(book)) ==="

## 2026-07-12: obsolete post-ChromaDB-removal. The rules corpus now lives in
## Postgres (rule_chunks, pgvector) — a real networked client-server DB, not
## a local file store like Chroma's persist directory was. A desktop doing
## bulk OCR/indexing no longer needs a separate "merge" step: if it can reach
## the canonical machine's Postgres (same LAN, port 5432 exposed per
## docker-compose.yml), just point DATABASE_URL at it directly and run
## `make ingest-book-native` there — build_index.py writes straight into the
## shared canonical DB, upserted on chunk_id (deterministic, content-derived —
## same idempotent-merge safety scripts/merge_chroma.py used to provide).
## A genuinely air-gapped desktop (no network path to the canonical Postgres
## at all) has no replacement workflow yet — flagged, not built, since it's
## not this project's common case.
merge-chroma:
	@echo "Obsolete — see this target's comment in the Makefile. Point DATABASE_URL at the canonical Postgres and run ingest-book-native directly instead."

## Load extract_entities.py JSON registries (produced by a `write_postgres=0`
## native run, e.g. on a desktop with no Postgres) into this machine's
## canonical Postgres, without re-running any LLM extraction. Idempotent —
## safe to re-run, same upsert key as extract_entities.py --write-postgres.
## Usage: make load-lore-json book="D&D 5.5E - Player's Handbook" source_type=core
##        make load-lore-json all_core=1
##        make load-lore-json all_adventures=1
load-lore-json:
	@test -n "$(book)$(all_core)$(all_adventures)" || (echo "Usage: make load-lore-json book=\"Name\" [source_type=adventure|core] | all_core=1 | all_adventures=1" && exit 1)
	./.venv/bin/python scripts/load_lore_json.py \
		$(if $(book),--book "$(book)",) \
		$(if $(source_type),--source-type "$(source_type)",) \
		$(if $(all_core),--all-core,) \
		$(if $(all_adventures),--all-adventures,) \
		$(if $(force_incomplete),--force-incomplete,)
