"""
One-time generator: parses the DMG's Individual Treasure, Treasure Hoard, gem/art,
and Magic Item (A-I) tables directly out of the ingested source markdown's embedded
HTML tables / bare-line lists, and writes backend/data/treasure_tables.py from the
parsed structures. Run once from the repo root: python3 gen_treasure_tables.py

This exists so the ~250-row transcription happens mechanically (parsed straight out
of the same local file already used for RAG ingestion) rather than by hand-retyping,
which is slow and error-prone at this volume. Re-runnable if the source file changes.
"""
import re
from html.parser import HTMLParser

SRC = "docs/source/core/D&D 5E - Dungeon Master's Guide.md"
OUT = "backend/data/treasure_tables.py"


class _TP(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row = None
        self._cell = None
        self._in_td = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag == "td":
            self._in_td = True
            self._cell = []

    def handle_endtag(self, tag):
        if tag == "td":
            self._row.append("".join(self._cell).strip())
            self._in_td = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._in_td:
            self._cell.append(data)


def _parse_tables(html_blocks: list[str]) -> list[list[str]]:
    rows = []
    for block in html_blocks:
        p = _TP()
        p.feed(block)
        rows += p.rows
    return rows


def _tables_after(text: str, marker: str, count: int = 1) -> list[str]:
    idx = text.index(marker)
    rest = text[idx:]
    return re.findall(r"<table>.*?</table>", rest)[:count]


_RANGE_RE = re.compile(r"^(\d{1,2})(?:[-–—](\d{1,2}))?$")


def _parse_range(s: str) -> tuple[int, int]:
    s = s.strip()
    if s == "00":
        return (100, 100)
    m = _RANGE_RE.match(s)
    if not m:
        raise ValueError(f"bad range {s!r}")
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    if hi == 0:
        hi = 100
    return (lo, hi)


_DICE_CELL_RE = re.compile(r"^(\d+d\d+)(?:\s*×\s*([\d,]+))?")


def _parse_coin_cell(cell: str) -> tuple[str, int] | None:
    """'4d6 × 100 (1,400)' -> ('4d6', 100). '5d6 (17)' -> ('5d6', 1). '—' -> None."""
    cell = cell.strip()
    if cell in ("—", ""):
        return None
    m = _DICE_CELL_RE.match(cell)
    if not m:
        raise ValueError(f"bad coin cell {cell!r}")
    dice = m.group(1)
    mult = int(m.group(2).replace(",", "")) if m.group(2) else 1
    return (dice, mult)


_GEMART_RE = re.compile(r"^(\d+d\d+)\s*\(\d+\)\s*([\d,]+)\s*gp\s*(gems|art objects)$")


def _parse_gemart_cell(cell: str) -> tuple[str, int, str] | None:
    """'2d6 (7) 50 gp gems' -> ('2d6', 50, 'gems'). '—' -> None."""
    cell = cell.strip()
    if cell in ("—", ""):
        return None
    m = _GEMART_RE.match(cell)
    if not m:
        raise ValueError(f"bad gem/art cell {cell!r}")
    return (m.group(1), int(m.group(2).replace(",", "")), m.group(3))


_MAGIC_ROLL_RE = re.compile(r"Roll (\d+d\d+|once) (?:times )?on Magic Item Table ([A-I])")


def _parse_magic_cell(cell: str) -> list[tuple[str, str]]:
    """'Roll 1d4 times on Magic Item Table A and 1d6 times on Magic Item Table B.'
    -> [('A', '1d4'), ('B', '1d6')]. '—' -> []."""
    cell = cell.strip()
    if cell in ("—", ""):
        return []
    out = []
    for count, letter in _MAGIC_ROLL_RE.findall(cell):
        out.append((letter, "1" if count == "once" else count))
    return out


def extract_individual_treasure(text: str) -> dict:
    markers = {
        1: "INDIVIDUAL TREASURE: CHALLENGE 0-4",
        2: "INDIVIDUAL TREASURE: CHALLENGE 5-10",
        3: "INDIVIDUAL TREASURE: CHALLENGE 11-16",
        4: "INDIVIDUAL TREASURE: CHALLENGE 17+",
    }
    denoms = ["cp", "sp", "ep", "gp", "pp"]
    result = {}
    for tier, marker in markers.items():
        rows = _parse_tables(_tables_after(text, marker, 1))
        entries = []
        for row in rows[1:]:
            lo, hi = _parse_range(row[0])
            cells = {}
            for denom, cell in zip(denoms, row[1:]):
                parsed = _parse_coin_cell(cell)
                if parsed:
                    cells[denom] = parsed
            entries.append((lo, hi, cells))
        result[tier] = entries
    return result


def extract_treasure_hoards(text: str) -> dict:
    markers = {
        1: "TREASURE HOARD: CHALLENGE 0-4",
        2: "TREASURE HOARD: CHALLENGE 5-10",
        3: "TREASURE HOARD: CHALLENGE 11-16",
        4: "TREASURE HOARD: CHALLENGE 17+",
    }
    denoms = ["cp", "sp", "ep", "gp", "pp"]
    result = {}
    for tier, marker in markers.items():
        rows = _parse_tables(_tables_after(text, marker, 1))
        # row 0: header, row 1: Coins, row 2: "d100 / Gems.../ Magic Items" header, rest: data
        coin_row = rows[1]
        assert coin_row[0] == "Coins", coin_row
        coins = {}
        for denom, cell in zip(denoms, coin_row[1:]):
            parsed = _parse_coin_cell(cell)
            if parsed:
                coins[denom] = parsed

        data_rows = rows[3:]
        # fix known OCR gap in the CR 11-16 table: "07-09" is immediately followed by
        # "11-12" with row 10 missing entirely from the source scan — the row widths
        # around it are otherwise a clean repeating 3-wide pattern, so this is almost
        # certainly a dropped leading digit ("10-12" mis-scanned as "11-12"), not an
        # intentional gap. Patched here, once, rather than silently left as a 1-in-100
        # dead roll.
        if tier == 3:
            for i, r in enumerate(data_rows):
                if r[0] == "11-12":
                    data_rows[i] = ["10-12"] + r[1:]
                    break

        entries = []
        for row in data_rows:
            lo, hi = _parse_range(row[0])
            gemart = _parse_gemart_cell(row[1]) if len(row) > 1 else None
            magic = _parse_magic_cell(row[2]) if len(row) > 2 else []
            entries.append((lo, hi, gemart, magic))
        result[tier] = {"coins": coins, "rows": entries}
    return result


def extract_gem_art_tables(text: str) -> tuple[dict, dict]:
    idx = text.index("USING THE INDIVIDUAL TREASURE")
    chunk = text[idx: idx + 16000]
    tables = re.findall(r"<table>.*?</table>", chunk)
    gems, art = {}, {}
    # Known fixed order in source: gem tables (10,50,100,500,1000,5000 gp), then
    # art object tables (25,250,750,2500,7500 gp), then an unrelated rarity table.
    gem_values = [10, 50, 100, 500, 1000, 5000]
    art_values = [25, 250, 750, 2500, 7500]
    gi = ai = 0
    for t in tables:
        rows = _parse_tables([t])
        if not rows:
            continue
        header = rows[0]
        names = [r[1] for r in rows[1:]]
        if header[-1] == "Stone Description" and gi < len(gem_values):
            gems[gem_values[gi]] = names
            gi += 1
        elif header[-1] == "Object" and ai < len(art_values):
            art[art_values[ai]] = names
            ai += 1
    assert gi == len(gem_values) and ai == len(art_values), (gi, ai)
    return gems, art


_NAME_NUM_RE = re.compile(r"^(\d+(?:[-–—]\d+)?)\s+(.+)$")


def extract_magic_item_tables(text: str) -> dict:
    tables: dict[str, list] = {}

    for letter, count in [("A", 1), ("B", 1), ("C", 1), ("D", 1), ("E", 1), ("F", 2), ("G", 2), ("H", 2)]:
        rows = _parse_tables(_tables_after(text, f"MAGIC ITEM TABLE {letter}", count))
        entries = []
        sub = None  # (die_size, [(lo,hi,name), ...]) accumulating for a nested subtable
        sub_parent_idx = None
        for row in rows[1:]:
            if row == ["d100", "Magic Item"]:
                continue  # repeated header on a continuation <table>
            range_cell, name_cell = row[0], row[1]
            if range_cell == "":
                # nested subtable row, e.g. "1 Bronze griffon" / "6-7 Onyx dog"
                m = _NAME_NUM_RE.match(name_cell)
                if not m:
                    raise ValueError(f"unexpected nested row {row!r}")
                sub_lo_hi = _parse_sub_range(m.group(1))
                sub[1].append((*sub_lo_hi, m.group(2)))
                continue
            if sub is not None:
                entries[sub_parent_idx] = (*entries[sub_parent_idx][:2], entries[sub_parent_idx][2], sub[1])
                sub = None
            lo, hi = _parse_range(range_cell)
            m = re.search(r"\(roll d(\d+)\)", name_cell)
            if m:
                sub = (int(m.group(1)), [])
                sub_parent_idx = len(entries)
                entries.append((lo, hi, name_cell, None))
            else:
                entries.append((lo, hi, name_cell, None))
        if sub is not None:
            entries[sub_parent_idx] = (*entries[sub_parent_idx][:2], entries[sub_parent_idx][2], sub[1])
        tables[letter] = entries

    # Table I is bare markdown lines, not an HTML table, and also contains one nested
    # subtable (76: "Magic armor (roll d12)").
    idx = text.index("MAGIC ITEM TABLE I")
    chunk = text[idx: idx + 4000]
    entries = []
    sub = None
    sub_parent_idx = None
    range_re = re.compile(r"^(\d{1,2})(?:[-–—](\d{1,2}))?$")
    for line in chunk.splitlines():
        line = line.strip()
        m = re.match(r"^(\d{1,2}(?:[-–—]\d{1,2})?)\s+(.+)$", line)
        if not m:
            continue
        range_str, name = m.group(1), m.group(2)
        if range_str == "00":
            lo, hi = 100, 100
        else:
            rm = range_re.match(range_str)
            lo, hi = int(rm.group(1)), int(rm.group(2) or rm.group(1))
            if hi == 0:
                hi = 100
        # OCR fix: "loun stone" -> "Ioun stone" (capital I misread as lowercase l
        # throughout this section — same class of fix as spells.py's OCR corrections).
        name = name.replace("loun stone", "Ioun stone")
        if sub is not None and hi <= sub[0]:
            sub[1].append((lo, hi, name))
            continue
        if sub is not None:
            entries[sub_parent_idx] = (*entries[sub_parent_idx][:2], entries[sub_parent_idx][2], sub[1])
            sub = None
        sub_m = re.search(r"\(roll d(\d+)\)", name)
        if sub_m:
            sub = (int(sub_m.group(1)), [])
            sub_parent_idx = len(entries)
            entries.append((lo, hi, name, None))
        else:
            entries.append((lo, hi, name, None))
    if sub is not None:
        entries[sub_parent_idx] = (*entries[sub_parent_idx][:2], entries[sub_parent_idx][2], sub[1])
    tables["I"] = entries

    return tables


def _parse_sub_range(s: str) -> tuple[int, int]:
    s = s.strip()
    if "-" in s or "–" in s:
        a, b = re.split(r"[-–]", s)
        return int(a), int(b)
    return int(s), int(s)


def _fmt_entries(entries, indent="    "):
    lines = []
    for e in entries:
        if len(e) == 4:
            lo, hi, name, sub = e
            if sub:
                lines.append(f"{indent}({lo}, {hi}, {name!r}, {sub!r}),")
            else:
                lines.append(f"{indent}({lo}, {hi}, {name!r}, None),")
        else:
            lines.append(f"{indent}{e!r},")
    return "\n".join(lines)


def main():
    text = open(SRC, encoding="utf-8").read()

    individual = extract_individual_treasure(text)
    hoards = extract_treasure_hoards(text)
    gems, art = extract_gem_art_tables(text)
    magic = extract_magic_item_tables(text)

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(MODULE_HEADER)

        f.write("\n\nINDIVIDUAL_TREASURE_TABLES: dict[int, list[tuple[int, int, dict]]] = {\n")
        for tier, entries in individual.items():
            f.write(f"    {tier}: [\n")
            for lo, hi, cells in entries:
                f.write(f"        ({lo}, {hi}, {cells!r}),\n")
            f.write("    ],\n")
        f.write("}\n")

        f.write("\n\nTREASURE_HOARD_TABLES: dict[int, dict] = {\n")
        for tier, data in hoards.items():
            f.write(f"    {tier}: {{\n")
            f.write(f"        \"coins\": {data['coins']!r},\n")
            f.write("        \"rows\": [\n")
            for lo, hi, gemart, magic_rolls in data["rows"]:
                f.write(f"            ({lo}, {hi}, {gemart!r}, {magic_rolls!r}),\n")
            f.write("        ],\n")
            f.write("    },\n")
        f.write("}\n")

        f.write("\n\nGEM_TABLES: dict[int, list[str]] = {\n")
        for value, names in gems.items():
            f.write(f"    {value}: {names!r},\n")
        f.write("}\n")

        f.write("\n\nART_OBJECT_TABLES: dict[int, list[str]] = {\n")
        for value, names in art.items():
            f.write(f"    {value}: {names!r},\n")
        f.write("}\n")

        f.write("\n\n# Each entry: (roll_lo, roll_hi, name, subtable). subtable is None, or a list of\n")
        f.write("# (sub_lo, sub_hi, sub_name) resolved by a second roll when the parent entry hits\n")
        f.write("# (e.g. Table G's \"Figurine of wondrous power (roll d8)\").\n")
        f.write("MAGIC_ITEM_TABLES: dict[str, list[tuple[int, int, str, list | None]]] = {\n")
        for letter, entries in magic.items():
            f.write(f"    {letter!r}: [\n")
            f.write(_fmt_entries(entries, indent="        "))
            f.write("\n    ],\n")
        f.write("}\n")

    print(f"wrote {OUT}")


MODULE_HEADER = '''"""
DMG Random Treasure tables (Ch. 7) — Individual Treasure, Treasure Hoard, gem/art
object tables, and Magic Item Tables A-I. Generated by
scripts/gen_treasure_tables.py, which parses these directly out of the ingested
`docs/source/core/D&D 5E - Dungeon Master's Guide.md` (the same source RulesStore
indexes for RAG) rather than hand-transcribed, to eliminate transposition risk across
~250 rows. Re-run that script if the source file ever changes.

Two known source corrections applied at generation time (see gen_treasure_tables.py
for exactly where): the Challenge 11-16 hoard table's row 10 was mis-scanned as
missing (fixed to a contiguous "10-12"), and "Ioun stone" was OCR'd as "loun stone"
throughout Magic Item Table I.

Magic item rarity/requires_attunement are NOT encoded per-item here (the DMG doesn't
tabulate those alongside the roll tables themselves) — see loot_generator.py for the
per-table rarity approximation and name-based attunement heuristic used when an item
is actually rolled.
"""
'''


if __name__ == "__main__":
    main()
