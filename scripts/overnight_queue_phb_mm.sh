#!/bin/bash
# One-off overnight queue: OCR the Player's Handbook, Monster Manual, and
# Dungeon Master's Guide, verifying and fully ingesting each in turn before
# moving to the next. Each step gates the next — a failure stops the chain
# rather than cascading bad/missing data into the next stage. Safe to leave
# running unattended; re-running any of the underlying steps by hand later
# is also safe (build_index.py/extract_entities.py are both resumable).
#
# Contextualization deliberately SKIPPED (skip_context=1 on every book) —
# PHB's own contextualization pass was projected at ~56 more hours (110,061
# chunks at ~2s/chunk), and the laptop needs to be turned off, not run for
# days unattended. Reasonable tradeoff: contextualization matters most for
# prose-heavy narrative content, much less for structured reference content
# (spell lists, stat blocks, tables) that's mostly self-contained and already
# covered by BM25 keyword search regardless — exactly what PHB/MM/DMG are.
# One side effect: PHB already has ~3,792 chunks indexed WITH contextual
# blurbs from before this switch (see chunk_id-based resumability — those
# already-done ids are skipped, not re-indexed) — a harmless one-time
# inconsistency, not worth a slower full re-embed to "fix."
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
  log "--- spot-check sample from $path (eyeball this for column garbling in the morning) ---"
  # Path passed as argv, not interpolated into the Python source itself —
  # book names with an apostrophe (e.g. "Dungeon Master's Guide") broke the
  # single-quoted Python string literal when interpolated directly.
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

# Root-caused the earlier crash: MinerU's vlm-engine hands off between
# internal 64-page "windows" on a multi-page run, and that handoff crashed
# reproducibly ("Timed out waiting for result of task") at exactly page 128
# (2 * 64) on two separate full-book runs — an isolated 15-page run spanning
# that same boundary completed cleanly, confirming it's the window handoff
# itself, not a bad page. ocr_ingest.py now chunks internally into <=60-page
# windows with per-chunk checkpointing to route around this, so a retry here
# is now just a safety net, not the primary fix. Also: ocr_ingest.py's
# DEFAULT_OUTPUT is docs/source/, NOT docs/source/core/ — core books need
# --output passed explicitly or the .md lands one directory too high and
# build_index.py (which only globs docs/source/core/*.md for source_type
# core) never finds it. Confirmed live: an earlier manual diagnostic run
# without --output silently wrote to the wrong path.
ocr_with_retry() {
  local pdf="$1" out_md="$2" label="$3"
  local attempt
  for attempt in 1 2; do
    log "=== OCR attempt $attempt/2: $label ==="
    if ./.venv/bin/python scripts/ocr_ingest.py --file "$pdf" --output "docs/source/core" && verify_md "$out_md" 500000; then
      spot_check "$out_md"
      return 0
    fi
    log "OCR attempt $attempt/2 failed for $label"
  done
  log "ABORT: $label OCR failed twice — giving up"
  return 1
}

ocr_with_retry "docs/raw/done/D&D 5.5E - Player's Handbook.pdf" "docs/source/core/D&D 5.5E - Player's Handbook.md" "Player's Handbook" || exit 1

log "=== [1/6] make ingest-book-native: Player's Handbook (contextualization SKIPPED — speed over quality, see below) ==="
make ingest-book-native book="D&D 5.5E - Player's Handbook" source_type=core skip_context=1 write_postgres=1 || { log "ABORT: PHB ingest failed"; exit 1; }
log "PHB ingest complete"

ocr_with_retry "docs/raw/done/D&D 5E - Monster Manual.pdf" "docs/source/core/D&D 5E - Monster Manual.md" "Monster Manual" || exit 1

log "=== [3/6] make ingest-book-native: Monster Manual (monster-only extraction, contextualization SKIPPED) ==="
make ingest-book-native book="D&D 5E - Monster Manual" source_type=core kinds=monster skip_context=1 write_postgres=1 || { log "ABORT: MM ingest failed"; exit 1; }
log "MM ingest complete"

ocr_with_retry "docs/raw/done/D&D 5E - Dungeon Master's Guide.pdf" "docs/source/core/D&D 5E - Dungeon Master's Guide.md" "Dungeon Master's Guide" || exit 1

log "=== [5/6] make ingest-book-native: Dungeon Master's Guide (contextualization SKIPPED) ==="
make ingest-book-native book="D&D 5E - Dungeon Master's Guide" source_type=core skip_context=1 write_postgres=1 || { log "ABORT: DMG ingest failed"; exit 1; }
log "DMG ingest complete"

log "=== [6/6] ALL DONE — PHB + Monster Manual + Dungeon Master's Guide fully re-OCR'd and re-ingested ==="
