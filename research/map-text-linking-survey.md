# Map ↔ Room-Text Linking Survey

Research spec input for a future "link numbered map locations to room-description
text" parsing pipeline. This is pure research — no extraction/parsing code was
written or modified.

## 0. Scope — what was actually sampled

| Book | Chapters/sites sampled | Pages looked at (PDF page numbers, 1-based) |
|---|---|---|
| **Lost Mine of Phandelver** | Cragmaw Hideout (goblin lair); Redbrand Hideout (start of key) | Read from the already-OCR'd Markdown at `docs/source/adventures/Lost Mine of Phandelver/Lost Mine of Phandelver.md` (no PDF page rendering needed — this book is already fully ingested) |
| **Out of the Abyss** | Ch.1 Velkynvelve (drow outpost, dungeon-style, numbered rooms); Ch.3 Sloobludop (kuo-toa town, sparse numbered/named landmarks) | PDF pages 9–15 (Velkynvelve text + map spread) and 44–46 (Sloobludop text + map) of `docs/raw/D&D 5E - Out of the Abyss.pdf` |
| **Dungeon of the Mad Mage** | Level 1: Dungeon Level (megadungeon, alphanumeric sub-rooms) | PDF pages 14–17 of `docs/raw/D&D 5E - dungeon of the mad mage.pdf` |
| **Ghost of Saltmarsh** | Ch.2 The Sinister Secret of Saltmarsh (1981 TSR-UK reprint — haunted house + caverns) | PDF pages 37–42 of `docs/raw/D&D 5E - ghost of saltmarsh.pdf` |

No planned book was dropped — all four recommended adventures were sampled. Renders were
saved to `/tmp/map-survey/*.png` (scratch, not part of the repo).

One scope note: **only Lost Mine of Phandelver has actually been run through
`ocr_ingest.py`** in this repo. The other three adventure folders under
`docs/source/adventures/` (`Out of the Abyss`, `Ghosts of Saltmarsh`, `Tomb of
Annihilation`, etc.) contain only a hand-authored `_meta.json` summary — no
`.md` body yet. So section 2 below is grounded in the one real example that
exists; conclusions about the other books' *future* Markdown output are
inference from the MinerU pipeline description, not verified against real
output for those books, and are flagged as such.

Also worth noting: the current `scripts/ocr_ingest.py` no longer matches the
"Tier 1 PyMuPDF / Tier 2 Apple Vision OCR" pipeline described in this task's
brief. It was rewritten to shell out to **MinerU's `vlm-engine` backend**
(MLX-accelerated layout/reading-order VLM) uniformly for every PDF, digital or
scanned (see the module docstring). The old PyMuPDF/Vision pipeline is gone
from the codebase; PyMuPDF (`fitz`) is only used today for page counts. This
matters for the spec because MinerU is a real layout model, not raw text
extraction — it should in principle preserve reading order and heading
structure better than either of the tiers described in the brief, which is
good news for a "nearest heading" heuristic, but it was not built to preserve
page-image references (see §2).

## 1. Taxonomy table

| Pattern | Books / chapters | Concrete example |
|---|---|---|
| **Bare number in a circle/badge directly on the map**, room heading exactly echoes it (`N. NAME`) | LMoP Cragmaw Hideout & Redbrand Hideout; OotA Velkynvelve; Mad Mage Level 1 | Mad Mage map: circle-free plain numerals `1`, `4`, `5`... labeled directly inside room outlines; text heading "1. Entry Well". OotA Velkynvelve map: numerals in circular badges (`10`, `11`, `12`...); text heading "10. Guard Tower". |
| **Alphanumeric sub-locations under one parent number** (3a/3b/3c) | Mad Mage (`2a`, `2b`, `6a`–`6e`, `8a`–`8c`, `9a`/`9b`, `30a`/`30b`, `32a`/`32b`...) — pervasive, dozens of instances | Text: "**2. Hall of Many Pillars** ... (area 2a) ... (area 2b)", then bold run-in sub-heads "**2a. Demon Reliefs**" and "**2b. Pillar Forest**" — sub-letters get bold run-in heads, not full numbered `#`-level headings. |
| **Single map node standing for a cluster of numbered rooms** (map compresses vertical stacking) | OotA Velkynvelve — map shows one circle labeled `6-8` on the top-down plan, but a second "side view" elevation panel on the *same page* separates it into stacked `6`, `7`, `8` | Text has three separate headings — "6. Shrine to Lolth", "7. Ilvara's Quarters", "8. Shoor's Quarters" — that all map to one visual node in the primary (top-down) view. Only the secondary side-view disambiguates. |
| **Sparse numbered/named landmarks over a mostly-unlabeled illustrative map** | OotA Sloobludop (kuo-toa town) | Map shows dozens of drawn huts/platforms with **no ID at all**; only 4 features carry a number+caption printed directly on the map (`3` "SEA MOTHER SHRINE", `4` "DEEP FATHER SHRINE", plus bare `1`/`2` at gate/dock positions). Most of the map has nothing to link to text. |
| **Plain numbers in room shapes + separate legend box for map ICONS (not room IDs)** | Ghost of Saltmarsh, "Sinister Secret of Saltmarsh" (Map 2.1, Haunted House) | Room numbers (`1`–`19`) printed directly in each room outline, continuous across the Ground Floor (1–10) and Upper Floor (11–19) of the *same* map sheet. A separate boxed legend at the bottom of the map explains furniture icons (Well, Cupboard, Chest, Sack) — easy to mistake for a room-key legend by a naive parser, but it is not one. |
| **Named-only sub-locations, no numbers, on the exterior of a numbered interior** | Ghost of Saltmarsh haunted house exterior | "House Exterior" section has bare named sub-heads — "The Wall", "The Garden", "The Well" — with no map identifier at all; the interior of the same site then switches to strict numbering once inside. |
| **Prose-only cross-references to numbers not yet introduced** | All four books, pervasive | Mad Mage chapter-opener prose: "areas 6 through 8", "watch posts in areas 23, 28, and 39" — appearing *before* any of those numbered headings exist on the page. Same pattern in Saltmarsh's "Background" ("area 24", "areas 27 and 28") and LMoP ("area 7", "area 12"). |
| **Explicit map-ID / chapter-map binding statement in prose** | Ghost of Saltmarsh ("The following locations are identified on **map 2.1**"); Mad Mage ("All location descriptions for this level are keyed to **map 1**") | Confirms books can have more than one map per chapter/level (Saltmarsh: map per floor-set, `2.1`, presumably `2.2` for the caverns) and that the room-number namespace is scoped to a specific named map, not global to the book. |
| **Cross-era republish packaging noise** | Ghost of Saltmarsh — "About the Original" sidebar reproduces the 1981 TSR-UK module's literal cover art/title, sits right next to real adventure content | A parser must not treat "Dungeon Module U1 / The Sinister Secret of Saltmarsh / ... by Dave J. Brown" as adventure body text — it's packaging/history trivia, visually a captioned image+sidebar, not a room entry. |

## 2. What `ocr_ingest.py`'s Markdown output preserves vs. discards

Grounded in `docs/source/adventures/Lost Mine of Phandelver/Lost Mine of Phandelver.md`
(the only real output file that exists in this repo):

- **No page-boundary markers of any kind.** There is no `<!-- page N -->` comment,
  no page-number footer preserved as a distinguishable token, nothing. The
  original book's own printed page numbers survive only incidentally, inside
  running prose, as cross-reference text the author wrote — e.g. line 5140:
  `"...see the "General Features" section (page 20)"` and line 1: `957: An "S" on the Redbrand Hideout map...`.
  These are the *book's own page numbers* (as printed on the page), not the
  PDF's physical page index, and they're indistinguishable from any other
  inline number without cross-referencing a second source (the PDF itself).
- **Headings are flattened to a single level.** Every heading in the file —
  book title, part title, room-key sidebars, actual numbered room headings —
  renders as a bare `#` (H1), e.g. line 357 `# Cragmaw Hideout`, line 363
  `# GENERAL FEATURES`, line 388 `# 1. CAVE MOUTH`, line 396 `# DEVELOPMENTS`.
  There is no H2/H3 nesting that would let a parser distinguish "this heading
  is a numbered room" from "this heading is a sidebar" or "this heading is a
  chapter title" structurally. The only signal is the **text of the heading
  itself** (does it start with a digit + period?).
- **Reading order is preserved well enough for a naive "next heading after the
  map" heuristic to work most of the time** — the room-key text does follow the
  book's intended top-to-bottom, left-to-right reading order in this file. But
  see the map-embedding behavior below, which actively breaks a naive
  page-based version of that heuristic.
- **Map images ARE embedded inline**, as an OCR'd image + a garbled
  `<details><summary>text_image</summary>...</details>` block containing the
  jumbled legend labels read off the map (see lines 426–452: the Cragmaw
  Hideout map appears as an embedded `.jpg` reference plus scrambled text
  "1 square = 5 feet / 6 4 5 7 3 8 1 2 Brlars Bridge Escarpment..." with no
  spatial relationship preserved — the room numbers 1–8 appear as an
  unordered token list, not tied to coordinates). Critically, **this map
  insert happens mid-room-description**: it appears between the "4. STEEP
  PASSAGE" heading's intro sentence and its continuation, i.e. the map
  physically interrupts a single room's text in the linear Markdown, exactly
  as it does in the source PDF layout (see also OotA below).
- **Sidebars are visually distinct in the PDF (boxed, often shaded) but
  collapse to the same flat `#` heading as everything else in the Markdown.**
  E.g. "GENERAL FEATURES", "WHAT THE GOBLINS KNOW", "ADVENTURE MAPS" (a
  meta/DM-advice sidebar about how maps work, unrelated to any specific room —
  line 468) all look identical, structurally, to "1. CAVE MOUTH" in the .md.
  A naive number-prefix regex is the only thing separating true room headings
  from all this other content.

Because the current pipeline (MinerU vlm-engine) is a genuine layout/VLM read
rather than the old raw-text-extraction path, it is plausible reading order
and header detection would be at least as good on the other three books once
ingested — but this is inference, not verified, since none of them exist as
real `.md` output yet.

## 3. Proposed `LocationRef` abstraction

A `LocationRef` should not assume a single global numbering scheme. Based on
everything above, the minimum viable shape is:

```
LocationRef {
  book_id            # e.g. "Out of the Abyss"
  map_id             # scoped identifier for the source map, e.g. "Velkynvelve",
                      # "Map 2.1", "Map 1" (level 1). Distinct maps within one
                      # book/chapter can reuse the same numeral (see below).
  primary_key        # the printed identifier, e.g. "10", "6-8", "2a", "39d"
  parent_key         # optional — e.g. "2" is the parent of "2a"/"2b"
  label              # optional printed name, e.g. "Guard Tower", "Deep Father Shrine"
  match_strategy     # see below
  confidence         # derived from which strategy matched
  source_span        # pointer into ingested text (heading text + rough offset)
}
```

Matching strategy needed per pattern found:

| Pattern | Strategy |
|---|---|
| Bare number, heading echoes exactly (`"10. Guard Tower"`) | **Exact numeric match** — regex `^(\d+[a-z]?)\.\s+(.*)`, join on `primary_key`. High confidence. |
| Alphanumeric sub-key (`2a`, `2b`) | **Exact match with parent inference** — strip trailing letter to get `parent_key`; both the compound key and the bare parent number must resolve, since prose references both ("area 2" and "area 2a"). |
| One map node covering several text headings (Velkynvelve `6-8`) | **Range/set expansion** — parse `"6-8"` into the set `{6,7,8}`, and expect *three* separate `LocationRef`s in the text, not one. Needs a human-authored or heuristic rule that a hyphenated map label is a range, not a single ID (risk of collision with subtraction/dash formatting elsewhere). |
| Sparse map with mostly no IDs (Sloobludop) | **Give up / no-op for unlabeled geometry.** Only emit `LocationRef`s for the handful of numbered/named features; everything else on the map is deliberately not a location to link. Don't force full map coverage. |
| Named-only sub-locations, no map ID (Saltmarsh house exterior) | **Fuzzy name match against section headings**, no numeric anchor at all — match on heading text similarity only ("The Well" ↔ prose "near the well"), lower confidence, more prone to false positives against generic prose. |
| Prose-only forward/backward reference ("see area 7") | **Not a LocationRef target itself** — this is a reference *to* a LocationRef defined elsewhere; needs to resolve against the set of `primary_key`s already known for that `map_id`, not create a new one. |
| Explicit map-ID binding sentence ("identified on map 2.1" / "keyed to map 1") | **High-value anchor — extract and pin `map_id` for every following numbered heading until the next such statement or chapter boundary.** This is the single best proximity signal found in the survey and should be actively searched for. |
| Continuous numbering across sub-maps of one site (Saltmarsh Ground Floor 1–10, Upper Floor 11–19 on one sheet) | **Exact numeric match, but do not assume numbering restarts at 1 per visual sub-section** — track a running max per `map_id`, not per page/floor label. |
| Cross-era reprint packaging (old module cover sidebar) | **Flag for human review** — heuristically detectable by sidebar/box formatting + presence of a proper-noun product code ("U1", "Dungeon Module") but risky to auto-classify confidently. |

## 4. Cases to flag for human review rather than auto-link

Based only on what was actually observed while sampling (not hypothetical):

1. **Map nodes that merge multiple numbered rooms** (OotA Velkynvelve `6-8`) —
   auto-expanding a hyphenated map label into a room range is a guess; a human
   should confirm the range and that no sub-room was missed.
2. **Any map with mostly-unlabeled illustrative content** (Sloobludop) — a
   parser that tries to force every drawn hut into a `LocationRef` will
   hallucinate locations; a human should set the small allow-list of IDs that
   actually matter.
3. **Named-only locations with no numeric anchor** (Saltmarsh house exterior:
   "The Wall", "The Garden", "The Well") — fuzzy name matching against prose
   is inherently higher-risk; flag matches below a similarity threshold.
4. **Sidebars/boxes that are topically adjacent to a room but not that room's
   description** — "Adventure Maps" (LMoP, general DM advice, appears
   physically between areas 4 and 5), "About the Original" (Saltmarsh, product
   nostalgia), "Roleplaying the Kuo-toa" (OotA, general NPC behavior advice
   physically inside chapter 3). All of these pass a naive "heading exists"
   test but are not room content.
5. **Reused/recurring named locations across multiple maps or levels** — not
   directly confirmed with a concrete duplicate-name example in this sampling
   pass, but the Mad Mage per-level "Map N" scoping (§3) implies room number 1
   recurs on every level (Level 1 has "1. Entry Well", presumably Level 2 has
   its own "1. ..."), so any cross-level reference absolutely needs
   `map_id` disambiguation and should not be resolved by number alone across
   the whole book.
6. **Map-page insertion that splits a single room's own text in half**
   (confirmed twice: LMoP's Cragmaw map between two paragraphs of "4. Steep
   Passage"; OotA's Velkynvelve map spread inserted mid-paragraph of "7.
   Ilvara's Quarters", picked back up verbatim on the next page) — a
   page-proximity heuristic ("nearest heading after the map page") will
   actively point *away* from the correct room here, since the map page itself
   sits inside that room's own text, not between two different rooms. This
   needs the parser to treat the map as a no-op insert to be skipped over,
   not a heading boundary.

## 5. Open questions / ambiguous cases hit while sampling

- **How does Mad Mage's per-level "Map N" numbering interact with the
  appendices/Skullport section?** Not sampled — the TOC lists 23 dungeon
  levels plus Skullport and three appendices; it's unclear whether Skullport
  uses the same "Map N" convention or its own city-map-with-named-districts
  style (more like a town map than a dungeon level). Left unverified.
- **Does OotA's "map N" citation convention (implied by Mad Mage/Saltmarsh)
  actually appear for Velkynvelve or Sloobludop?** Not observed in the pages
  sampled — the OotA text mostly just says "area 10" or gives no explicit map
  binding sentence at all near the room headings sampled. This suggests OotA
  may be less consistent about stating its map ID than Mad Mage or Saltmarsh,
  but this wasn't exhaustively checked across the whole book (only 2 of ~15
  named locations/set-pieces were sampled).
- **What does the "star" icon and "S" marker mean on the Mad Mage map?** Visibly
  present on the map (stars near several room numbers, "S" near others,
  presumably secret doors per the "Dungeon Key" section on PDF page 5 of that
  book) but that legend page itself was not sampled, so the icon meanings are
  inferred from Saltmarsh's analogous "S" = secret door convention, not
  confirmed for Mad Mage specifically.
- **Is the LMoP map's garbled embedded-image text (`Brlars`, `CRAGMAW HIDEOUT`,
  the unordered number list) typical of what MinerU will produce for the same
  page, or specific to the older PyMuPDF/Vision pipeline that produced the
  existing LMoP .md file?** This is unresolved and matters a lot: the existing
  LMoP output was almost certainly produced by the *old* pipeline (before the
  MinerU rewrite — see design.md's migration notes), not the current
  `ocr_ingest.py`. A re-run of LMoP under current MinerU could look
  meaningfully different (better or worse) for exactly this
  map-image-to-text handling, and that difference wasn't (and couldn't be)
  checked without actually re-running ingestion, which is out of scope for
  this research task.
- **Range notation ambiguity**: is `"6-8"` on a map always a range, or could a
  future book use a literal hyphenated ID (e.g. a corridor labeled "6-8" as a
  single compound name rather than shorthand for rooms 6, 7, 8)? Only one
  example of this pattern was found (Velkynvelve), so the "it's a range"
  inference is based on n=1 and corroborated only by the side-view panel on
  the same page — a different book might not offer that corroborating second
  view.
