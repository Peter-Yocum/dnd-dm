#!/usr/bin/env python3
"""
validate_source.py — heuristic quality check on OCR'd markdown source files.

Scans docs/source/*.md for common OCR problems and D&D stat block anomalies,
then outputs a prioritised report so you know exactly which pages to re-check.

Checks performed:
  1. OCR failure comments left by ocr_ingest.py
  2. Repeated-line clusters (repetition-loop bleed-through)
  3. Garbled numbers — l/I/O characters inside digit sequences (e.g. "l5", "1O")
  4. Stat block presence detection (AC + HP + ability scores)
  5. Ability score range validation (each score 1–30)
  6. HP dice average cross-check (stated HP vs. dice expression average)

Usage:
    python validate_source.py
    python validate_source.py --input docs/source --output report.txt
    python validate_source.py --input docs/source --severity error
"""

import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "docs/source"
DEFAULT_OUTPUT = "-"          # stdout

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}

# How many times a line must repeat within a window to be flagged.
REPEAT_THRESHOLD = 3
REPEAT_WINDOW    = 60    # lines

# Below this paragraphs/lines ratio, flag the get_text("text") pagination-
# collapse bug (see check_paragraph_density) — healthy adventures checked so
# far sit around 0.2+, the one known-bad sample (Out of the Abyss) sits
# around 0.011.
PARAGRAPH_RATIO_THRESHOLD = 0.05

# Stat block anchor phrases (case-insensitive).
STAT_BLOCK_ANCHORS = [
    r"armor class",
    r"hit points",
    r"speed\s+\d",
    r"str\s+dex\s+con",       # ability score header row
]

# Dice expression: "7d8 + 14", "2d6", "1d10-1", etc.
DICE_RE = re.compile(r'(\d+)d(\d+)\s*([+-]\s*\d+)?', re.IGNORECASE)

# HP line: "Hit Points 45 (7d8 + 14)" or "**Hit Points** 45 (7d8 + 14)"
HP_LINE_RE = re.compile(
    r'hit points[*\s]+(\d+)\s*\(([^)]+)\)',
    re.IGNORECASE,
)

# Ability score table row: "10 (+0) 14 (+2) 12 (+1) 8 (−1) 13 (+1) 11 (+0)"
# Unicode minus (−) and ASCII minus (-) both accepted.
ABILITY_ROW_RE = re.compile(
    r'(\d{1,2})\s*\([+−-]\d+\)\s+'   # STR
    r'(\d{1,2})\s*\([+−-]\d+\)\s+'   # DEX
    r'(\d{1,2})\s*\([+−-]\d+\)\s+'   # CON
    r'(\d{1,2})\s*\([+−-]\d+\)\s+'   # INT
    r'(\d{1,2})\s*\([+−-]\d+\)\s+'   # WIS
    r'(\d{1,2})\s*\([+−-]\d+\)'      # CHA
)

# Garbled number pattern: digit(s) adjacent to l/I/O where a digit is expected.
GARBLED_RE = re.compile(r'\b(?:\d+[lIoO]\d*|\d*[lIoO]\d+)\b')

# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class Issue:
    severity: str        # "error" | "warning" | "info"
    check:    str
    file:     str
    line:     int        # 1-based; 0 = whole-file issue
    snippet:  str

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.severity.upper():7}] {self.check:30}  {loc}\n           {self.snippet}"


# ── individual checks ─────────────────────────────────────────────────────────

def check_ocr_failures(lines: list[str], fname: str) -> list[Issue]:
    issues = []
    for i, line in enumerate(lines, 1):
        if "<!-- OCR FAILED" in line:
            issues.append(Issue("error", "ocr_failure_comment", fname, i, line.strip()[:120]))
    return issues


def check_repeated_lines(lines: list[str], fname: str) -> list[Issue]:
    """Flag lines that repeat >= REPEAT_THRESHOLD times within REPEAT_WINDOW lines."""
    issues = []
    n = len(lines)
    flagged_starts: set[int] = set()
    for i in range(n):
        line = lines[i].strip()
        if not line or len(line) < 8:
            continue
        window_end = min(n, i + REPEAT_WINDOW)
        count = sum(1 for j in range(i, window_end) if lines[j].strip() == line)
        if count >= REPEAT_THRESHOLD and i not in flagged_starts:
            # Mark all occurrences so we only emit one issue per cluster.
            for j in range(i, window_end):
                if lines[j].strip() == line:
                    flagged_starts.add(j)
            issues.append(Issue(
                "error", "repeated_lines", fname, i + 1,
                f"'{line[:80]}' repeated {count}x near line {i + 1}"
            ))
    return issues


def check_garbled_numbers(lines: list[str], fname: str) -> list[Issue]:
    issues = []
    for i, line in enumerate(lines, 1):
        for m in GARBLED_RE.finditer(line):
            issues.append(Issue(
                "warning", "garbled_number", fname, i,
                f"suspect token '{m.group()}' in: {line.strip()[:100]}"
            ))
    return issues


def check_hp_dice_math(lines: list[str], fname: str) -> list[Issue]:
    """Verify that stated HP matches the average of the dice expression (±2)."""
    issues = []
    for i, line in enumerate(lines, 1):
        m = HP_LINE_RE.search(line)
        if not m:
            continue
        stated_hp = int(m.group(1))
        dice_expr = m.group(2)
        # Sum up all dice terms in the expression.
        expected = 0.0
        for dm in DICE_RE.finditer(dice_expr):
            n_dice, sides = int(dm.group(1)), int(dm.group(2))
            modifier_str = (dm.group(3) or "").replace(" ", "").replace("−", "-")
            modifier = int(modifier_str) if modifier_str else 0
            expected += n_dice * (sides + 1) / 2 + modifier
        if abs(stated_hp - expected) > 2:
            issues.append(Issue(
                "warning", "hp_dice_mismatch", fname, i,
                f"stated {stated_hp} HP but dice avg is {expected:.1f} — expr: ({dice_expr.strip()})"
            ))
    return issues


def check_ability_scores(lines: list[str], fname: str) -> list[Issue]:
    """Flag ability score rows with any score outside 1–30."""
    issues = []
    for i, line in enumerate(lines, 1):
        m = ABILITY_ROW_RE.search(line)
        if not m:
            continue
        scores = [int(m.group(k)) for k in range(1, 7)]
        bad = [s for s in scores if not (1 <= s <= 30)]
        if bad:
            issues.append(Issue(
                "error", "ability_score_range", fname, i,
                f"scores out of range {bad} in: {line.strip()[:100]}"
            ))
    return issues


def check_paragraph_density(lines: list[str], fname: str) -> list[Issue]:
    """Flag the Tier-1 PyMuPDF get_text("text") pagination-collapse bug (see
    ocr_ingest.py/design.md): that extraction method only inserts a paragraph
    break (blank line) BETWEEN pages, not within one, so a PDF whose pages
    don't happen to have enough internal blank lines survives extraction as
    a handful of giant blobs instead of real paragraphs — starving
    add_headers.py's candidate detection (requires a heading to be the first
    line of a blank-line-delimited paragraph) down to zero hits, silently.

    Found live 2026-07-04 in Out of the Abyss: 64 paragraphs across 5803
    lines (ratio ~0.011), vs. 800+ paragraphs for similarly-sized adventures
    (ratio ~0.2+) extracted cleanly. PARAGRAPH_RATIO_THRESHOLD sits well
    below every healthy sample checked so far and well above the one known-bad
    sample, so it should catch the same failure mode elsewhere without
    false-positiving on ordinary short-paragraph-style writing."""
    n = len(lines)
    if n < 200:
        return []  # too short for the ratio to be a meaningful signal
    text = "\n".join(l.rstrip() for l in lines)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    ratio = len(paragraphs) / n
    if ratio < PARAGRAPH_RATIO_THRESHOLD:
        return [Issue(
            "error", "low_paragraph_density", fname, 0,
            f"{len(paragraphs)} paragraphs across {n} lines (ratio {ratio:.4f}, "
            f"threshold {PARAGRAPH_RATIO_THRESHOLD}) — likely the get_text(\"text\") "
            f"pagination-collapse bug; re-extract with get_text(\"blocks\") instead."
        )]
    return []


def check_incomplete_stat_blocks(lines: list[str], fname: str) -> list[Issue]:
    """Warn when AC is present but HP or ability scores are missing nearby (±30 lines)."""
    issues = []
    n = len(lines)
    for i, line in enumerate(lines):
        if not re.search(r'armor class', line, re.IGNORECASE):
            continue
        window = "\n".join(lines[max(0, i - 5): min(n, i + 30)])
        missing = []
        if not re.search(r'hit points', window, re.IGNORECASE):
            missing.append("Hit Points")
        if not ABILITY_ROW_RE.search(window):
            missing.append("ability score row")
        if missing:
            issues.append(Issue(
                "warning", "incomplete_stat_block", fname, i + 1,
                f"'Armor Class' found but missing: {', '.join(missing)}"
            ))
    return issues


# ── runner ────────────────────────────────────────────────────────────────────

CHECKS = [
    check_ocr_failures,
    check_repeated_lines,
    check_garbled_numbers,
    check_paragraph_density,
    check_hp_dice_math,
    check_ability_scores,
    check_incomplete_stat_blocks,
]


def validate_file(path: Path) -> list[Issue]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return [Issue("error", "file_read_error", str(path), 0, str(e))]
    fname = str(path)
    issues: list[Issue] = []
    for check in CHECKS:
        issues.extend(check(lines, fname))
    return issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate OCR'd markdown source files.")
    ap.add_argument("--input",    default=DEFAULT_INPUT, help="folder of .md files")
    ap.add_argument("--output",   default=DEFAULT_OUTPUT, help="report file path, or - for stdout")
    ap.add_argument("--severity", default="info",
                    choices=["error", "warning", "info"],
                    help="minimum severity to report (default: info)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    files = sorted(in_dir.rglob("*.md"))
    if not files:
        print(f"No .md files found in {in_dir}/")
        sys.exit(1)

    min_sev = SEVERITY_ORDER[args.severity]
    all_issues: list[Issue] = []
    for f in files:
        all_issues.extend(validate_file(f))

    filtered = [iss for iss in all_issues if SEVERITY_ORDER[iss.severity] <= min_sev]
    filtered.sort(key=lambda x: (x.file, x.line))

    # ── summary counts ──────────────────────────────────────────────────────
    counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    by_check: dict[str, int] = {}
    for iss in filtered:
        counts[iss.severity] += 1
        by_check[iss.check] = by_check.get(iss.check, 0) + 1

    lines_out: list[str] = []
    lines_out.append(f"D&D Source Validation Report — {len(files)} file(s) scanned")
    lines_out.append("=" * 70)
    lines_out.append(f"  Errors:   {counts['error']}")
    lines_out.append(f"  Warnings: {counts['warning']}")
    lines_out.append(f"  Info:     {counts['info']}")
    lines_out.append("")
    lines_out.append("Issues by check:")
    for check, n in sorted(by_check.items(), key=lambda x: -x[1]):
        lines_out.append(f"  {n:>5}  {check}")
    lines_out.append("")
    lines_out.append("=" * 70)
    lines_out.append("")

    current_file = None
    for iss in filtered:
        if iss.file != current_file:
            current_file = iss.file
            lines_out.append(f"\n── {iss.file} ──")
        lines_out.append(str(iss))

    report = "\n".join(lines_out)

    if args.output == "-":
        print(report)
    else:
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}  ({len(filtered)} issues)")


if __name__ == "__main__":
    main()
