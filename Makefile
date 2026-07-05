.PHONY: up down build restart logs psql migrate migration rollback shell fresh index index-if-empty setup

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

## Tear down all containers + volumes, start fresh, and run migrations.
fresh:
	docker compose down -v
	docker compose up -d
	@echo "Waiting for services..."
	@until docker compose exec app alembic upgrade head 2>/dev/null; do \
		printf '.'; sleep 2; \
	done
	@echo " done."

## Build the ChromaDB vector index from docs/source/*.md inside the container.
index:
	docker compose exec app python scripts/build_index.py

## Only index if data/chroma_db is empty (safe to run on first clone).
index-if-empty:
	@if [ -z "$$(ls -A data/chroma_db 2>/dev/null)" ]; then \
		echo "ChromaDB empty — building index (this takes a while)..."; \
		docker compose exec app python scripts/build_index.py; \
	else \
		echo "ChromaDB already populated — skipping. Run 'make index' to force rebuild."; \
	fi

## First-time setup on a new machine: migrate DB + index if needed.
setup: migrate index-if-empty
