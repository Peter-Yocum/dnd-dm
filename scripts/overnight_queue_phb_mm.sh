#!/bin/bash
# One-off overnight queue: OCR the Player's Handbook, Monster Manual, and
# Dungeon Master's Guide, verifying and fully ingesting each in turn before
# moving to the next. Each step gates the next — a failure stops the chain
# rather than cascading bad/missing data into the next stage. Safe to leave
# running unattended; re-running any of the underlying steps by hand later
# is also safe (build_index.py/extract_entities.py are both resumable).
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
  ./.venv/bin/python -c "
t = open('$path', encoding='utf-8').read()
i = t.find('Armor Class')
if i == -1:
    print('(no \"Armor Class\" occurrence found to sample)')
else:
    print(t[max(0, i - 200):i + 500])
"
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

log "=== [1/6] make ingest-book: Player's Handbook ==="
make ingest-book book="D&D 5.5E - Player's Handbook" source_type=core || { log "ABORT: PHB ingest failed"; exit 1; }
log "PHB ingest complete"

ocr_with_retry "docs/raw/done/D&D 5E - Monster Manual.pdf" "docs/source/core/D&D 5E - Monster Manual.md" "Monster Manual" || exit 1

log "=== [3/6] make ingest-book: Monster Manual (monster-only extraction) ==="
make ingest-book book="D&D 5E - Monster Manual" source_type=core kinds=monster || { log "ABORT: MM ingest failed"; exit 1; }
log "MM ingest complete"

ocr_with_retry "docs/raw/done/D&D 5E - Dungeon Master's Guide.pdf" "docs/source/core/D&D 5E - Dungeon Master's Guide.md" "Dungeon Master's Guide" || exit 1

log "=== [5/6] make ingest-book: Dungeon Master's Guide ==="
make ingest-book book="D&D 5E - Dungeon Master's Guide" source_type=core || { log "ABORT: DMG ingest failed"; exit 1; }
log "DMG ingest complete"

log "=== [6/6] ALL DONE — PHB + Monster Manual + Dungeon Master's Guide fully re-OCR'd and re-ingested ==="
