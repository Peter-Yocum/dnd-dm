#!/usr/bin/env python3
"""
clean_source.py — LLM-assisted cleanup of garbled text in extracted markdown.

Recursively scans docs/source/**/*.md — this matches build_index.py's own
layout (docs/source/core/*.md, docs/source/adventures/{slug}/*.md) — for
paragraphs with suspicious tokens: artifacts from decorative PDF typography,
dropped drop caps, split capitalised words, tilde ligature artifacts,
digit/letter confusion, encoding glyphs. Only flagged paragraphs are sent to
a local Ollama text model for correction. Everything else is left untouched.

Usage:
    python clean_source.py                        # uses gemma4:26b-mlx by default
    python clean_source.py --model qwen2.5:14b     # override with a lighter/faster model
    python clean_source.py --file docs/source/foo.md
    python clean_source.py --dry-run     # report issues without writing files
    python clean_source.py --ollama-url http://gaming-rig:11434
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Make `backend` importable (for backend.llm's Ollama client factory) when run
# as `python scripts/clean_source.py` from anywhere — Python sets sys.path[0]
# to this script's own directory, not the repo root. Same shim as the
# backfill_* scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_INPUT  = "docs/source"
DEFAULT_MODEL  = "gemma4:26b-mlx"
DEFAULT_OLLAMA = "http://localhost:11434"

# ── suspicious pattern detection ──────────────────────────────────────────────
# Each pattern targets a specific class of PDF extraction artifact.

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("tilde_in_word",    re.compile(r'\b\w*~\w*\b')),
    ("digit_in_word",    re.compile(r'\b[a-zA-Z]+\d[a-zA-Z]+\b')),
    ("truncated_word",   re.compile(r'(?<![.!?])\b[a-z]\?\s')),
    ("glyph_artifact",   re.compile(r'[a-zA-Z]{2,}[!]{1}[a-zA-Z<>]{1,2}\b')),
    ("split_caps_word",  re.compile(r'\b[A-Z]{2,}\s[A-Z]{2,}\b')),
    ("mixed_case_brand", re.compile(r'\b[A-Z][a-z][A-Z]{2,}\b')),
]

def _is_suspicious(paragraph: str) -> list[str]:
    """Return list of matching pattern names, empty if paragraph looks clean."""
    hits = []
    for name, pat in _PATTERNS:
        if pat.search(paragraph):
            hits.append(name)
    return hits


# ── LLM cleanup ───────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a text correction assistant. Fix ONLY garbled or corrupted words "
    "caused by PDF extraction artifacts (decorative fonts, drop caps, encoding "
    "issues). Do not rephrase, summarise, add, or remove any content. "
    "Preserve all whitespace, line breaks, and markdown exactly."
)

_PROMPT = """The text below was extracted from a D&D PDF and contains garbled words.

Common artifacts to recognise and fix:
- Drop cap losing its letter: "T'S GOOD" → "IT'S GOOD", "d? you" → "do you"
- Tilde ligature: "y~u" → "you", "t~e" → "the"
- Digit for letter: "v1llams" → "villains", "1t" → "it"
- Split capitalised word: "DU NGEON" → "DUNGEON", "WI ZARD" → "WIZARD"
- Glyph artifact: "f!f>" → "&", "DuNGEONS f!f> DRAGONS" → "DUNGEONS & DRAGONS"
- Mixed-case brand: "DuNGEONS" → "DUNGEONS"

Context (do not modify):
{before}

TEXT TO FIX (return only this part, corrected):
{target}

Context (do not modify):
{after}

Return ONLY the corrected version of the TEXT TO FIX section. No explanation."""

# Literal fragments of the prompt template above. If any of these show up in
# the model's response, it echoed part of the prompt's own scaffolding back
# instead of (or alongside) the actual correction — observed in practice with
# smaller models: the length-ratio guard alone doesn't catch this, since an
# echoed label plus a near-verbatim copy of the input is still close to a 1.0
# ratio. Caught this after it silently wrote "TEXT TO FIX" and "Context (do
# not modify):" directly into indexed book content.
_PROMPT_ECHO_MARKERS = [
    "TEXT TO FIX",
    "Context (do not modify)",
    "Return ONLY the corrected version",
]


def _clean_paragraph(text: str, before: str, after: str, model: str, ollama_url: str) -> str:
    """Send one paragraph to Ollama for cleanup and return the corrected version."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.llm import ollama_chat

    # reasoning=False matters especially here (now factory-enforced): for
    # thinking-capable models (e.g. gemma4:26b-mlx) a hidden reasoning trace
    # can dwarf the actual output (observed: 8779 eval tokens/255s with
    # thinking vs. 1169 eval tokens/42s without, for the same paragraph).
    llm = ollama_chat(model=model, base_url=ollama_url, timeout=None)
    prompt = _PROMPT.format(
        before=before or "(start of file)",
        target=text,
        after=after or "(end of file)",
    )
    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    cleaned = response.content.strip()

    # Guard 1: the LLM echoed prompt scaffolding into its response rather
    # than returning only the corrected text.
    echoed = [m for m in _PROMPT_ECHO_MARKERS if m in cleaned]
    if echoed:
        print(f"      ! LLM echoed prompt scaffolding ({', '.join(echoed)}) — keeping original", file=sys.stderr)
        return text

    # Guard 2: if the LLM returned dramatically more or less text, skip it.
    ratio = len(cleaned) / max(len(text), 1)
    if not (0.5 <= ratio <= 2.0):
        print(f"      ! LLM output length ratio {ratio:.1f} — keeping original", file=sys.stderr)
        return text
    return cleaned


# ── per-file processing ───────────────────────────────────────────────────────

MAX_GROUP_CHARS = 6000  # keep each LLM call comfortably inside the model's context window


def _group_adjacent(indices: list[int], gap: int = 2) -> list[list[int]]:
    """Merge indices within `gap` of each other into groups."""
    if not indices:
        return []
    groups = [[indices[0]]]
    for idx in indices[1:]:
        if idx - groups[-1][-1] <= gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _split_range_by_size(paragraphs: list[str], first: int, last: int, max_chars: int) -> list[tuple[int, int]]:
    """Split a contiguous paragraph range into smaller (first, last) sub-ranges
    so no single LLM call processes more than max_chars of text.

    Defense in depth, independent of source formatting quality: a run of
    flagged paragraphs dense enough (e.g. a stat-block-heavy page, where
    false-positive-prone patterns like ALL-CAPS headers and digit/letter
    stats are everywhere) can chain-merge via _group_adjacent into a span
    that dwarfs the model's context window — that's what was happening here:
    one merged group spanning hundreds of "paragraphs" (which, before the
    ocr_ingest.py paragraph-detection fix, meant hundreds of whole *pages*)
    sent well over a million characters in a single call. Ollama silently
    truncates/mishandles a prompt that large, and the model returns nothing
    useful — visible as "LLM output length ratio 0.0" for every group.
    """
    ranges: list[tuple[int, int]] = []
    start = first
    length = 0
    for i in range(first, last + 1):
        p_len = len(paragraphs[i])
        if i > start and length + p_len > max_chars:
            ranges.append((start, i - 1))
            start = i
            length = 0
        length += p_len
    ranges.append((start, last))
    return ranges


def clean_file(
    path: Path,
    model: str,
    ollama_url: str,
    dry_run: bool = False,
    context_paras: int = 1,
) -> int:
    """Clean one markdown file. Returns number of paragraphs fixed.

    Checkpoints to disk after every individual fix (rather than once at the
    end) so an interrupted run keeps whatever progress it already made on
    this file, instead of losing the whole file's work.

    Ranges are processed in descending `first`-index order rather than the
    order they were found in. A fix splices a (first, last) range down to a
    single paragraph, which shifts the index of everything after `last` —
    processing high-to-low guarantees any range not yet handled sits at a
    lower index than every splice done so far, so it's never invalidated by
    an earlier fix (this isn't just a 1-off drift: any multi-paragraph range,
    which is the common case, shifts everything after it by last - first).
    """
    text = path.read_text(encoding="utf-8")
    paragraphs = text.split("\n\n")

    # Find and group suspicious paragraphs.
    flagged_indices = [i for i, p in enumerate(paragraphs) if _is_suspicious(p)]
    if not flagged_indices:
        return 0

    groups = _group_adjacent(flagged_indices, gap=context_paras + 1)

    ranges: list[tuple[int, int]] = []
    for group in groups:
        ranges.extend(_split_range_by_size(paragraphs, group[0], group[-1], MAX_GROUP_CHARS))
    ranges.sort(key=lambda r: r[0], reverse=True)

    fixed = 0

    for first, last in ranges:
        target = "\n\n".join(paragraphs[first: last + 1])
        before = "\n\n".join(paragraphs[max(0, first - context_paras): first])
        after  = "\n\n".join(paragraphs[last + 1: last + 1 + context_paras])

        reasons = _is_suspicious(target)
        print(f"    [{', '.join(reasons)}] para {first}–{last} ({len(target)} chars): {target[:80].strip()!r}…")

        if dry_run:
            fixed += last - first + 1
            continue

        cleaned = _clean_paragraph(target, before, after, model, ollama_url)

        if cleaned != target:
            # Splice the cleaned block back in, then checkpoint immediately.
            paragraphs[first: last + 1] = [cleaned]
            fixed += last - first + 1
            path.write_text("\n\n".join(paragraphs), encoding="utf-8")

    return fixed


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="LLM cleanup of garbled PDF extraction artifacts.")
    ap.add_argument("--input",       default=DEFAULT_INPUT, help="folder of .md files, searched recursively")
    ap.add_argument("--file",        default=None,          help="clean a single .md file")
    ap.add_argument("--model",       default=DEFAULT_MODEL, help="Ollama model to use")
    ap.add_argument("--ollama-url",  default=None,          help="Ollama base URL")
    ap.add_argument("--dry-run",     action="store_true",   help="detect only, don't write")
    ap.add_argument("--context",     type=int, default=1,   help="context paragraphs around each fix")
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
        print(f"No .md files found.")
        return

    total_fixed = 0
    print(f"{'DRY RUN — ' if args.dry_run else ''}Cleaning {len(files)} file(s) with {args.model}\n")

    for f in files:
        try:
            label = f.relative_to(args.input)
        except ValueError:
            label = f.name
        print(f"── {label}")
        n = clean_file(f, args.model, ollama_url, dry_run=args.dry_run, context_paras=args.context)
        if n:
            action = "would fix" if args.dry_run else "fixed"
            print(f"   {action} {n} paragraph(s)")
        else:
            print(f"   clean")
        total_fixed += n

    action = "would be fixed" if args.dry_run else "fixed"
    print(f"\nDone. {total_fixed} paragraph(s) {action} across {len(files)} file(s).")


if __name__ == "__main__":
    main()
