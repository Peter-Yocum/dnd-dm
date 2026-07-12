#!/bin/bash
# Native overnight queue: OCR + reindex + extract a manifest of core/adventure
# books, entirely via the host venv (no Docker) — see design.md's
# "Evolution" section, 2026-07-07/08 incidents, for why: Docker Desktop's
# VM on this laptop has under 1GB of RAM, which OOM-killed a Docker-routed
# bulk ingest repeatedly at real (100k+ chunk) scale. Routes around that
# ceiling entirely rather than hoping this run doesn't hit it. Each book is
# OCR -> reindex -> extract, gated (a failure stops the chain rather than
# cascading bad/missing data into the next book) — same discipline as
# overnight_queue_phb_mm.sh, generalized to a manifest instead of one
# hardcoded set of books so future batches don't need a new script.
#
# skip_context=1 on every book, same tradeoff as overnight_queue_phb_mm.sh:
# contextualization matters most for prose-heavy narrative content, much
# less for structured reference content (spell lists, stat blocks, tables)
# — and it's the difference between an overnight run and a multi-day one.
#
# write_postgres=1 on every book — Postgres is reachable at localhost:5432
# (published by docker-compose) even though indexing itself runs natively;
# see design.md for why this is safe (RulesStore/config.py's own defaults
# already point at localhost).
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="overnight-ingest-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

verify_md() {
  local path="$1" min_bytes="$2"
  if [ ! -s "$path" ]; then
    log "FAIL: $path missing or empty"
    return 1
  fi
  local size
  size=$(stat -f%z "$path")
  if [ "$size" -lt "$min_bytes" ]; then
    log "FAIL: $path only $size bytes (expected at least $min_bytes)"
    return 1
  fi
  log "OK: $path is $size bytes"
  return 0
}

spot_check() {
  local path="$1"
  log "--- spot-check sample from $path (eyeball this for column garbling later) ---"
  ./.venv/bin/python -c "
import sys
t = open(sys.argv[1], encoding='utf-8').read()
i = t.find('Armor Class')
if i == -1:
    print('(no \"Armor Class\" occurrence found to sample)')
else:
    print(t[max(0, i - 200):i + 500])
" "$path"
}

ocr_with_retry() {
  local pdf="$1" out_md="$2" label="$3" output_dir="$4"
  local attempt
  for attempt in 1 2; do
    log "=== OCR attempt $attempt/2: $label ==="
    if ./.venv/bin/python scripts/ocr_ingest.py --file "$pdf" --output "$output_dir" && verify_md "$out_md" 500000; then
      spot_check "$out_md"
      return 0
    fi
    log "OCR attempt $attempt/2 failed for $label"
  done
  log "ABORT: $label OCR failed twice — giving up"
  return 1
}

# Manifest: one book per line, "|"-separated:
#   pdf filename (no path/extension) | source_type (core|adventure) | adventure slug (adventure only, else empty) | kinds (empty = default all kinds)
# All entries below are core supplements — round 2, per 2026-07-08 priority
# call (core supplements before adventure modules). Add adventure entries
# the same way; source_type=adventure needs the 4th field (slug) and its
# .md lands in docs/source/adventures/{slug}/ instead of docs/source/core/.
BOOKS=(
  "D&D 5E - Tasha's Cauldron of Everything|core||"
  "D&D 5E - Sword Coast Adventurer's Guide|core||"
)

total=${#BOOKS[@]}
i=0
for entry in "${BOOKS[@]}"; do
  i=$((i + 1))
  IFS='|' read -r name source_type slug kinds <<< "$entry"
  pdf="docs/raw/${name}.pdf"

  if [ "$source_type" = "adventure" ]; then
    output_dir="docs/source/adventures/${slug}"
  else
    output_dir="docs/source/core"
  fi
  out_md="${output_dir}/${name}.md"

  ocr_with_retry "$pdf" "$out_md" "$name" "$output_dir" || exit 1

  log "=== [$i/$total] make ingest-book-native: $name (kinds=${kinds:-default}) ==="
  if [ "$source_type" = "adventure" ]; then
    if [ -n "$kinds" ]; then
      make ingest-book-native adventure="$slug" source_type="$source_type" kinds="$kinds" skip_context=1 write_postgres=1 \
        || { log "ABORT: $name ingest failed"; exit 1; }
    else
      make ingest-book-native adventure="$slug" source_type="$source_type" skip_context=1 write_postgres=1 \
        || { log "ABORT: $name ingest failed"; exit 1; }
    fi
  else
    if [ -n "$kinds" ]; then
      make ingest-book-native book="$name" source_type="$source_type" kinds="$kinds" skip_context=1 write_postgres=1 \
        || { log "ABORT: $name ingest failed"; exit 1; }
    else
      make ingest-book-native book="$name" source_type="$source_type" skip_context=1 write_postgres=1 \
        || { log "ABORT: $name ingest failed"; exit 1; }
    fi
  fi
  log "$name ingest complete"
done

log "=== ALL DONE — ${total} book(s) fully OCR'd and ingested ==="
