#!/usr/bin/env python3
"""
ocr_ingest.py — Convert PDF rulebooks to clean Markdown for the DM RAG pipeline.

Two-tier extraction, macOS only:
  Tier 1 — PyMuPDF native text  (digital PDFs — instant, perfect quality)
  Tier 2 — Apple Vision OCR     (scanned/photo PDFs — on-device, the same
                                 engine behind Preview's Live Text)

Vision reads in its own layout-aware result order — verified correct on
genuine two-column pages, do not "improve" this with a manual position
sort, it makes column pages worse (see _ocr_page_with_vision). It
occasionally garbles stylized/decorative sidebar text; clean_source.py's
LLM cleanup pass is the intended fix-up step for that.

Per-PDF detection: the script samples the first few pages. If embedded text
is dense enough, it uses Tier 1. Otherwise it falls through to Tier 2. Most
purchased D&D PDFs (DMs Guild, official releases) are digital and hit Tier 1
immediately. Photographed or photocopied books use Tier 2. Note a PDF that's
*visually* selectable in a viewer isn't necessarily digital at the file
level — macOS Live Text does its own on-the-fly OCR over pure page-image
PDFs, which can look identical to real embedded text until you check the
file itself.

Usage:
    python ocr_ingest.py                         # whole docs/raw/ folder
    python ocr_ingest.py --file docs/raw/foo.pdf
    python ocr_ingest.py --file docs/raw/foo.pdf --start-page 306 --pages 1
    python ocr_ingest.py --force                 # re-process even if .md exists
    python ocr_ingest.py --no-ocr                # skip scanned PDFs entirely

Requirements:
    pip install pyobjc-framework-Vision pyobjc-framework-Quartz
"""

import argparse
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — already in requirements.txt

DEFAULT_INPUT       = "docs/raw"
DEFAULT_OUTPUT      = "docs/source"
DIGITAL_THRESHOLD   = 100   # avg chars/page below this → treat as scanned
DETECTION_SAMPLE    = 5     # pages sampled to decide digital vs scanned
DEFAULT_VISION_DPI  = 300   # page render resolution for Vision OCR


# ── Tier 1: native text extraction ────────────────────────────────────────────

def _is_scanned(doc: fitz.Document) -> bool:
    sample = min(len(doc), DETECTION_SAMPLE)
    total  = sum(len(doc[i].get_text().strip()) for i in range(sample))
    avg    = total / sample if sample else 0
    return avg < DIGITAL_THRESHOLD


def _partial_path(out_path: Path) -> Path:
    """Staging path used while a file is still being written. Writing here
    (not directly to out_path) and renaming only on success means a crash
    mid-run leaves no file at out_path — so a later re-run without --force
    correctly sees the PDF as not-yet-done, instead of silently treating a
    truncated file as complete."""
    return out_path.with_suffix(out_path.suffix + ".partial")


def _extract_digital(doc: fitz.Document, out_path: Path, first: int, last: int) -> int:
    """Write native markdown from embedded PDF text. Returns char count."""
    total  = len(doc)
    chars  = 0
    partial = _partial_path(out_path)

    with partial.open("w", encoding="utf-8") as fh:
        for i in range(first, last):
            md = doc[i].get_text("text").strip()
            if not md:
                md = f"<!-- page {i + 1}/{total}: no embedded text -->"
            fh.write(md + ("\n\n" if i < last - 1 else ""))
            fh.flush()
            chars += len(md)
            print(f"    page {i + 1}/{total}")

    partial.replace(out_path)
    return chars


# ── Tier 2: Apple Vision OCR (macOS only) ──────────────────────────────────────

def _load_vision():
    """Import pyobjc's Vision/Quartz bindings. macOS-only — the Vision
    framework isn't available on other platforms, so this fails fast with a
    clear message rather than letting an ImportError look like a real bug."""
    if sys.platform != "darwin":
        print("Scanned-PDF OCR requires macOS (uses Apple's Vision framework).", file=sys.stderr)
        sys.exit(1)
    try:
        import Quartz
        import Vision
        from Foundation import NSURL
        return Vision, Quartz, NSURL
    except ImportError:
        print(
            "pyobjc Vision bindings not installed. Run:\n"
            "  pip install pyobjc-framework-Vision pyobjc-framework-Quartz",
            file=sys.stderr,
        )
        sys.exit(1)


MAX_RENDER_DIMENSION = 6000  # pixels — comfortably under MuPDF's internal size cap


def _render_page_safely(page: "fitz.Page", dpi: int) -> "fitz.Pixmap":
    """Render a page to a pixmap, capping the effective DPI so the output
    never exceeds MAX_RENDER_DIMENSION per side.

    Most pages are normal book-page sized and render at the requested DPI
    unchanged. But some adventure PDFs bind fold-out maps/posters as pages
    sized to their full physical print dimensions (observed: a page with
    rect 6000x4215 *points* — roughly 83x58 inches — versus ~565x780 points
    for a normal page). Rendering a page that size at 300 DPI asks MuPDF for
    a ~25000x17500px pixmap, well past its internal safety limit, and it
    raises FzErrorLimit("Overly large image") instead of returning one. Map
    labels are low-value for RAG search anyway, so scaling down to fit is an
    acceptable tradeoff, not a quality-critical page.
    """
    rect = page.rect
    scale = dpi / 72.0
    px_w, px_h = rect.width * scale, rect.height * scale
    longest = max(px_w, px_h)
    if longest > MAX_RENDER_DIMENSION:
        dpi = int(dpi * (MAX_RENDER_DIMENSION / longest))
    return page.get_pixmap(dpi=dpi)


def _ocr_page_with_vision(pix: "fitz.Pixmap", Vision, Quartz, NSURL) -> str:
    """Run on-device Vision OCR against a rendered page image, returning
    recognized text in reading order.

    Deliberately does NOT re-sort Vision's `request.results()` order — Vision
    already does its own layout-aware ordering internally (verified: on a
    genuine two-column page, its natural result order reads correctly
    top-to-bottom within each column, while re-sorting purely by y-position
    interleaves the two columns line-by-line and scrambles the text). Trust
    it. Any remaining rough edges (stylized/decorative sidebar text in
    particular) are exactly what clean_source.py's LLM pass is designed to
    catch and fix afterward.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pix.save(tmp.name)
        url = NSURL.fileURLWithPath_(tmp.name)
        source = Quartz.CGImageSourceCreateWithURL(url, None)
        cg_image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)

        observations = []

        def handler(request, error):
            if error:
                return
            observations.extend(request.results())

        request = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(handler)
        request.setRecognitionLevel_(1)       # VNRequestTextRecognitionLevelAccurate
        request.setUsesLanguageCorrection_(True)
        req_handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)
        req_handler.performRequests_error_([request], None)

        return _join_as_markdown(observations)


_PARAGRAPH_GAP_MULTIPLIER = 1.8  # line gap this many times the page's typical
                                  # line spacing is treated as a paragraph break


def _join_as_markdown(observations: list) -> str:
    """Join Vision's line observations into real markdown — a blank line
    between paragraphs, a single newline within one. Vision returns
    line-level text with no paragraph grouping, so paragraph breaks are
    inferred from vertical spacing: line gaps cluster tightly around the
    page's typical line-to-line spacing (verified: stddev is tiny — a real
    body-text page's gaps bunch around one value with a handful of clear
    outliers at 2-5x that value, which are genuine paragraph/section
    breaks). This matters beyond readability — clean_source.py's paragraph
    splitter (`text.split("\\n\\n")`) depends on real paragraph boundaries
    existing; without this, it saw "one page = one paragraph" and merged
    hundreds of pages into single LLM calls that blew past the model's
    context window.

    A negative gap (jumping from the bottom of one column back to the top
    of the next) is excluded from the "typical spacing" baseline but still
    only gets a single newline, not a paragraph break — imperfect at true
    column boundaries, but doesn't reintroduce the group-explosion problem.
    """
    lines = [o.topCandidates_(1)[0].string() for o in observations]
    if len(observations) < 3:
        return "\n".join(lines)

    ys = [o.boundingBox().origin.y for o in observations]  # bottom-left origin
    gaps = [ys[i] - ys[i + 1] for i in range(len(ys) - 1)]  # positive = moving down the page
    positive_gaps = sorted(g for g in gaps if g > 0)
    if not positive_gaps:
        return "\n".join(lines)
    median_gap = positive_gaps[len(positive_gaps) // 2]

    out = [lines[0]]
    for i, gap in enumerate(gaps):
        out.append("\n\n" if gap > median_gap * _PARAGRAPH_GAP_MULTIPLIER else "\n")
        out.append(lines[i + 1])
    return "".join(out)


def _extract_with_vision(pdf_path: Path, out_path: Path, first: int, last: int, dpi: int) -> int:
    """Run Apple Vision OCR on a scanned PDF and write the resulting markdown.

    A single bad page (rendering failure, corrupt image data, etc.) doesn't
    abort a multi-hundred-page run — it's recorded as a placeholder comment
    and extraction continues, matching how a genuinely textless page is
    already handled.
    """
    Vision, Quartz, NSURL = _load_vision()
    doc = fitz.open(pdf_path)
    total = len(doc)
    chars = 0
    partial = _partial_path(out_path)

    with partial.open("w", encoding="utf-8") as fh:
        for i in range(first, last):
            try:
                pix = _render_page_safely(doc[i], dpi)
                text = _ocr_page_with_vision(pix, Vision, Quartz, NSURL).strip()
            except Exception as e:
                print(f"    page {i + 1}/{total}: FAILED ({e}) — writing placeholder")
                text = ""
            if not text:
                text = f"<!-- page {i + 1}/{total}: no text recognized -->"
            fh.write(text + ("\n\n" if i < last - 1 else ""))
            fh.flush()
            chars += len(text)
            print(f"    page {i + 1}/{total}")

    doc.close()
    partial.replace(out_path)
    return chars


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PDFs to Markdown for the DM RAG pipeline.")
    ap.add_argument("--input",             default=DEFAULT_INPUT,  help="folder of source PDFs")
    ap.add_argument("--output",            default=DEFAULT_OUTPUT, help="folder for .md output")
    ap.add_argument("--file",              default=None,           help="process a single PDF instead of the whole folder")
    ap.add_argument("--pages",             type=int, default=None, help="max pages to process (default: all)")
    ap.add_argument("--start-page",        type=int, default=1,    help="1-based page to start from (default: 1)")
    ap.add_argument("--force",             action="store_true",    help="re-process even if .md already exists")
    ap.add_argument("--no-ocr",            action="store_true",    help="skip scanned PDFs rather than running OCR")
    ap.add_argument("--dpi",               type=int, default=DEFAULT_VISION_DPI,
                     help=f"page render resolution for OCR (default: {DEFAULT_VISION_DPI})")
    args = ap.parse_args()

    in_dir   = Path(args.input)
    out_dir  = Path(args.output)
    done_dir = in_dir / "done"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(exist_ok=True)

    if args.file:
        pdfs = [Path(args.file)]
    else:
        pdfs = sorted(in_dir.glob("*.pdf"))

    if not pdfs:
        print(f"No PDFs found in {in_dir}/")
        return

    first = max(0, args.start_page - 1)  # convert to 0-based

    scanned_skipped = 0

    print(f"{len(pdfs)} PDF(s) to process\n")

    for pdf in pdfs:
        out_path = out_dir / f"{pdf.stem}.md"

        if out_path.exists() and not args.force:
            print(f"SKIP  {pdf.name}  (already done — use --force to redo)")
            continue

        doc  = fitz.open(pdf)
        total = len(doc)
        last  = min(total, first + args.pages) if args.pages else total

        print(f"── {pdf.name}  ({total} pages) ──")

        if _is_scanned(doc):
            print(f"  Type: SCANNED")
            if args.no_ocr:
                print(f"  Skipped (--no-ocr set)")
                scanned_skipped += 1
                doc.close()
                continue
            doc.close()
            print(f"  Engine: Vision (on-device OCR)")
            chars = _extract_with_vision(pdf, out_path, first, last, args.dpi)
        else:
            print(f"  Type: DIGITAL (native text)")
            chars = _extract_digital(doc, out_path, first, last)
            doc.close()

        size_kb = out_path.stat().st_size // 1024
        print(f"  Wrote {out_path.name}  ({chars:,} chars, {size_kb} KB)")

        # Don't move the file when targeting a single page range — it may need
        # another pass for other pages.
        if not args.file:
            pdf.rename(done_dir / pdf.name)
            print(f"  Moved → done/")

        print()

    if scanned_skipped:
        print(f"\n{scanned_skipped} scanned PDF(s) skipped. Re-run without --no-ocr to process them.")

    print("Done.")


if __name__ == "__main__":
    main()
