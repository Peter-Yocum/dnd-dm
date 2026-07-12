#!/usr/bin/env python3
"""
add_headers.py — detect real section/chapter/area headings in OCR'd markdown
and promote them to `## ` markdown headers, so build_index.py's existing
header-based chunker (chunk_markdown in build_index.py, which already works
for hand-authored files like Curse of Strahd) can split on real structure
instead of falling back to blind ~1500-char word-count chunks labeled
"Document (N)".

Runs after clean_source.py, before build_index.py:
    ocr_ingest.py  →  clean_source.py  →  add_headers.py  →  build_index.py

Same paragraph model as clean_source.py (split on "\n\n"), same two-stage
approach: a cheap regex prefilter finds short, punctuation-light paragraphs
that *might* be headings (deliberately over-inclusive), then a single batched
LLM call per group classifies each candidate as a real heading (returning a
cleaned version) or not. Skip-on-doubt: any candidate the model doesn't
clearly confirm, or any batch whose response doesn't parse cleanly, is left
untouched rather than guessed at — an unpromoted heading just falls back to
today's word-count chunking (no worse than before); a wrongly-promoted one
would corrupt chunk boundaries for real content, which is worse.

Usage:
    python add_headers.py                        # uses gemma4:26b-mlx by default
    python add_headers.py --model qwen2.5:14b     # override with a lighter/faster model
    python add_headers.py --file docs/source/adventures/Waterdeep/*.md
    python add_headers.py --dry-run               # report candidates without writing
    python add_headers.py --ollama-url http://gaming-rig:11434
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Make `backend` importable (for backend.llm's Ollama client factory) when run
# as `python scripts/add_headers.py` from anywhere — Python sets sys.path[0]
# to this script's own directory, not the repo root. Same shim as the
# backfill_* scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_INPUT  = "docs/source"
DEFAULT_MODEL  = "gemma4:26b-mlx"
DEFAULT_OLLAMA = "http://localhost:11434"
BATCH_SIZE     = 25  # heading candidates per LLM call

# ── candidate detection ────────────────────────────────────────────────────
# Deliberately over-inclusive — the LLM batch call is the real filter.
#
# ocr_ingest.py's paragraph detection groups a heading together with the body
# text that follows it into one \n\n-delimited block (e.g. one real paragraph
# is 'CREDITS \nLead Designer: Christopher Perkins \n...', thousands of chars
# long) rather than isolating the heading as its own short paragraph. So the
# candidate unit is the FIRST LINE of each paragraph, not the whole paragraph:
# short, no terminal sentence punctuation, not already a header, not a page
# marker.

_PAGE_MARKER_RE = re.compile(r'^<!--\s*page\s')
_ALREADY_HEADER_RE = re.compile(r'^#{1,6}\s')
_SENTENCE_END_RE = re.compile(r'[.!?]\s*$')
_WORD_RE = re.compile(r"[A-Za-z']+")
_MAX_CANDIDATE_CHARS = 100
_MIN_CAPITALIZED_WORD_RATIO = 0.7  # headings run ALL CAPS or Title Case in
                                    # this OCR'd style; body-text lines that
                                    # happen to land as a paragraph's first
                                    # line (mid-sentence wraps, dialogue) read
                                    # as normal sentence case and fail this —
                                    # the single biggest false-positive filter,
                                    # cheaper than sending them to the LLM


def _first_line_and_rest(paragraph: str) -> tuple[str, str]:
    first, _, rest = paragraph.partition("\n")
    return first.strip(), rest.strip()


def _is_mostly_capitalized(text: str) -> bool:
    words = _WORD_RE.findall(text)
    if not words:
        return False
    capped = sum(1 for w in words if w[0].isupper())
    return capped / len(words) >= _MIN_CAPITALIZED_WORD_RATIO


def _is_heading_candidate(first_line: str) -> bool:
    if not first_line:
        return False
    if _ALREADY_HEADER_RE.match(first_line) or _PAGE_MARKER_RE.match(first_line):
        return False
    if len(first_line) > _MAX_CANDIDATE_CHARS:
        return False
    if _SENTENCE_END_RE.search(first_line):
        return False
    if not re.search(r'[A-Za-z]', first_line):
        return False
    if not _is_mostly_capitalized(first_line):
        return False
    return True


# ── LLM classification ──────────────────────────────────────────────────────

_SYSTEM = (
    "You classify short lines extracted from a Dungeons & Dragons rulebook or "
    "adventure module. For each numbered line, decide whether it is a genuine "
    "section/chapter/area/stat-block heading (a title that would appear as a "
    "heading in the book), as opposed to body text, a list fragment, a name, "
    "a table row, or other non-heading content. Fix any obvious OCR garbling "
    "in headings you confirm (e.g. stray digits, split capitals, tilde "
    "artifacts) but do not invent or rephrase words."
)

_PROMPT = """Lines:
{numbered}

For EVERY numbered line above, output exactly one line of the form:
N: HEADING <corrected text>
or
N: NONE

Use HEADING only for genuine titles — chapter names, section names, named \
areas/rooms ("Area 12: The Vault"), monster/NPC stat-block names introducing \
a new entry, part/appendix titles. Use NONE for anything else, including \
character names in a credits list, single words with no title-like context, \
or sentence/dialogue fragments — even ones that happen to be capitalized, \
e.g. a line beginning mid-description like "Three Crates Are Stacked \
Against The Wall" is body text, not a heading, regardless of capitalization.

Output exactly {n} lines, one per input line, in the same order, nothing else."""

_RESPONSE_LINE_RE = re.compile(r'^\s*(\d+)\s*:\s*(HEADING|NONE)\s*(.*)$')


def _classify_batch(
    candidates: list[str], model: str, ollama_url: str
) -> dict[int, str]:
    """candidates is a list of raw paragraph texts. Returns {index: cleaned_heading}
    for confirmed headings only (index into the candidates list).

    Skip-on-doubt applies per-line, not per-batch: any index the model didn't
    return, or returned in a form that doesn't parse, is simply left out of
    the result (falls back to no header, same as today) rather than voiding
    every other classification in the batch — a single malformed line out of
    25 shouldn't cost 24 good ones."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.llm import ollama_chat

    numbered = "\n".join(f"{i + 1}: {c.strip()}" for i, c in enumerate(candidates))
    prompt = _PROMPT.format(numbered=numbered, n=len(candidates))

    llm = ollama_chat(model=model, base_url=ollama_url, timeout=None)
    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    lines = [l for l in response.content.strip().splitlines() if l.strip()]

    parsed: dict[int, tuple[bool, str]] = {}
    unparsed = 0
    for line in lines:
        m = _RESPONSE_LINE_RE.match(line)
        if not m:
            unparsed += 1
            continue
        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(candidates)):
            continue
        is_heading = m.group(2) == "HEADING"
        parsed[idx] = (is_heading, m.group(3).strip())

    missing = len(candidates) - len(parsed)
    if unparsed or missing:
        print(
            f"      ! LLM response: {unparsed} unparsed line(s), {missing} "
            f"candidate(s) not classified — leaving those without a header",
            file=sys.stderr,
        )

    result = {}
    for idx, (is_heading, text) in parsed.items():
        if is_heading and text:
            result[idx] = text
    return result


# ── per-file processing ─────────────────────────────────────────────────────

def add_headers_to_file(
    path: Path, model: str, ollama_url: str, dry_run: bool = False
) -> tuple[int, int]:
    """Returns (candidates_found, headers_confirmed).

    Checkpoints to disk after every batch that confirms at least one heading,
    rather than once at the end, so an interrupted run keeps whatever headings
    it already confirmed on this file. Safe to do per-batch (unlike
    clean_source.py's per-range checkpointing) because each confirmed heading
    replaces its own paragraph index in place — the list never shrinks or
    grows, so no index ever shifts and batch order doesn't matter.
    """
    text = path.read_text(encoding="utf-8")
    paragraphs = text.split("\n\n")

    split_paragraphs = [_first_line_and_rest(p) for p in paragraphs]
    candidate_indices = [
        i for i, (first, _) in enumerate(split_paragraphs) if _is_heading_candidate(first)
    ]
    if not candidate_indices:
        return 0, 0

    confirmed = 0
    for batch_start in range(0, len(candidate_indices), BATCH_SIZE):
        batch_indices = candidate_indices[batch_start: batch_start + BATCH_SIZE]
        batch_first_lines = [split_paragraphs[i][0] for i in batch_indices]

        for i, t in zip(batch_indices, batch_first_lines):
            print(f"    candidate para {i}: {t[:60]!r}")

        if dry_run:
            continue

        headings = _classify_batch(batch_first_lines, model, ollama_url)
        batch_confirmed = 0
        for local_idx, cleaned in headings.items():
            para_idx = batch_indices[local_idx]
            rest = split_paragraphs[para_idx][1]
            paragraphs[para_idx] = f"## {cleaned}\n\n{rest}" if rest else f"## {cleaned}"
            batch_confirmed += 1

        confirmed += batch_confirmed
        if batch_confirmed:
            path.write_text("\n\n".join(paragraphs), encoding="utf-8")

    return len(candidate_indices), confirmed


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Promote real headings in OCR'd markdown to ## headers.")
    ap.add_argument("--input",      default=DEFAULT_INPUT, help="folder of .md files, searched recursively")
    ap.add_argument("--file",       default=None,          help="process a single .md file")
    ap.add_argument("--model",      default=DEFAULT_MODEL, help="Ollama model to use")
    ap.add_argument("--ollama-url", default=None,           help="Ollama base URL")
    ap.add_argument("--dry-run",    action="store_true",    help="report candidates without writing")
    args = ap.parse_args()

    ollama_url = (
        args.ollama_url
        or os.environ.get("OLLAMA_BASE_URL")
        or DEFAULT_OLLAMA
    )

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path(args.input).glob("**/*.md"))

    if not files:
        print("No .md files found.")
        return

    total_candidates = 0
    total_confirmed = 0
    print(f"{'DRY RUN — ' if args.dry_run else ''}Scanning {len(files)} file(s) with {args.model}\n")

    for f in files:
        try:
            label = f.relative_to(args.input)
        except ValueError:
            label = f.name
        print(f"── {label}")
        n_candidates, n_confirmed = add_headers_to_file(f, args.model, ollama_url, dry_run=args.dry_run)
        if n_candidates:
            action = "would confirm" if args.dry_run else "confirmed"
            print(f"   {n_candidates} candidate(s), {action} {n_confirmed} heading(s)")
        else:
            print("   no candidates")
        total_candidates += n_candidates
        total_confirmed += n_confirmed

    action = "would be confirmed" if args.dry_run else "confirmed"
    print(f"\nDone. {total_candidates} candidate(s) found, {total_confirmed} heading(s) {action} across {len(files)} file(s).")


if __name__ == "__main__":
    main()
