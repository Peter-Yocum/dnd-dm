#!/usr/bin/env python3
"""
ocr_ingest.py — Convert PDF rulebooks to clean Markdown for the DM RAG pipeline.

Uses MinerU's vlm-engine backend (MLX-accelerated on Apple Silicon): a
purpose-built PDF→Markdown pipeline with a real layout/reading-order model,
so it handles multi-column pages and strips running headers/footers/page
numbers natively. This replaced an earlier PyMuPDF-native-text +
Apple-Vision-OCR pipeline, which had neither: raw PyMuPDF text extraction has
no layout awareness at all, and Vision (the engine behind Live Text) scrambled
some multi-column pages and let running headers bleed into body text.

Every PDF — digital or scanned — goes through the same MinerU vlm-engine
path. This costs meaningfully more time on already-digital PDFs than the old
fast PyMuPDF path did (MinerU is a page-image VLM read, tens of seconds per
page, regardless of whether the PDF already had a clean text layer), but a
uniform path is what actually fixes the reading-order/header bugs, and
per-PDF speed isn't the bottleneck for a personal library ingested once.

Usage:
    python ocr_ingest.py                         # whole docs/raw/ folder
    python ocr_ingest.py --file docs/raw/foo.pdf
    python ocr_ingest.py --file docs/raw/foo.pdf --start-page 306 --pages 1
    python ocr_ingest.py --force                 # re-process even if .md exists

Requirements:
    uv pip install -U "mineru[all]"

    The MLX backend needs macOS 13.5+ on Apple Silicon and `mlx-vlm` to
    import cleanly, or MinerU silently falls back to a much slower
    CPU/transformers path. Verify before running a big job:
        python -c "from mineru.utils.engine_utils import _select_mac_engine; print(_select_mac_engine())"
    This must print "mlx", not "transformers". If it prints "transformers",
    `import mlx_vlm` is failing (check for a broken transformers/torchvision
    import chain — e.g. a Python built without _lzma support breaks this).
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF — only used here for page counts, not extraction

DEFAULT_INPUT  = "docs/raw"
DEFAULT_OUTPUT = "docs/source"

# MinerU's vlm-engine batches a multi-page run into internal 64-page
# "windows" and hands off between them. That handoff was confirmed live to
# crash reproducibly ("Timed out waiting for result of task") at exactly
# page 128 (= 2 * 64) on two separate full-book (387-page) runs of the same
# PDF, while an isolated 15-page run spanning that exact page range
# completed cleanly — i.e. it's the window-to-window handoff itself that's
# broken, not any particular page. Keeping every individual mineru
# invocation at or under one window's worth of pages sidesteps the handoff
# entirely.
_CHUNK_PAGES = 60


def _partial_path(out_path: Path) -> Path:
    """Staging path used while a file is still being written. Writing here
    (not directly to out_path) and renaming only on success means a crash
    mid-run leaves no file at out_path — so a later re-run without --force
    correctly sees the PDF as not-yet-done, instead of silently treating a
    truncated file as complete."""
    return out_path.with_suffix(out_path.suffix + ".partial")


def _run_mineru_range(mineru_bin: Path, pdf_path: Path, first: int, last: int) -> str:
    """Run MinerU's vlm-engine over pages [first, last) (0-based, half-open)
    and return the resulting markdown text for just that range. MinerU
    writes into its own <stem>/vlm/<stem>.md layout (plus images/ and
    various json sidecars) inside a scratch directory; only the final .md's
    text is kept. Raises CalledProcessError on failure."""
    with tempfile.TemporaryDirectory() as scratch:
        subprocess.run(
            [
                str(mineru_bin),
                "-p", str(pdf_path),
                "-o", scratch,
                "-s", str(first),
                "-e", str(last - 1),
                "-b", "vlm-engine",
            ],
            check=True,
        )
        produced = Path(scratch) / pdf_path.stem / "vlm" / f"{pdf_path.stem}.md"
        return produced.read_text(encoding="utf-8")


def _extract_with_mineru(pdf_path: Path, out_path: Path, first: int, last: int, force: bool) -> int:
    """Run MinerU over pages [first, last) (0-based, half-open) in
    <= _CHUNK_PAGES windows, checkpointing each window's raw text to a cache
    dir next to out_path before concatenating everything into out_path.
    A crash loses at most one window's work (a few minutes), not the whole
    book — re-running the same command skips windows whose cache file
    already exists, same discipline as extract_entities.py's per-entity
    checkpointing.
    """
    partial = _partial_path(out_path)

    # Resolve mineru relative to the running interpreter rather than trusting
    # PATH — invoking this script via a venv's python (e.g. .venv/bin/python
    # scripts/ocr_ingest.py) does not put that venv's bin/ on PATH, so a bare
    # "mineru" lookup fails even though it's installed right next to it.
    # Reuse sys.executable's own suffix (".exe" on Windows, "" on Mac/Linux)
    # rather than hardcoding one — confirmed live that the un-suffixed guess
    # silently failed on Windows (pip installs console scripts as
    # "mineru.exe" there, not bare "mineru"), producing a WinError 2 from
    # subprocess.run once it fell through to the PATH-lookup fallback too
    # (the venv's Scripts/ dir isn't on PATH unless the venv is activated).
    interpreter_suffix = Path(sys.executable).suffix
    mineru_bin = Path(sys.executable).parent / f"mineru{interpreter_suffix}"
    if not mineru_bin.exists():
        mineru_bin = Path("mineru")  # fall back to PATH lookup

    cache_dir = out_path.parent / f".{pdf_path.stem}.ocr_chunks"
    if force and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    pos = first
    while pos < last:
        chunk_end = min(last, pos + _CHUNK_PAGES)
        chunk_path = cache_dir / f"{pos:04d}_{chunk_end:04d}.md"
        if chunk_path.exists():
            print(f"    pages {pos + 1}-{chunk_end}: cached, skipping")
        else:
            print(f"    pages {pos + 1}-{chunk_end}: extracting...")
            text = _run_mineru_range(mineru_bin, pdf_path, pos, chunk_end)
            chunk_partial = _partial_path(chunk_path)
            chunk_partial.write_text(text, encoding="utf-8")
            chunk_partial.replace(chunk_path)
        texts.append(chunk_path.read_text(encoding="utf-8"))
        pos = chunk_end

    combined = "\n\n".join(texts)
    partial.write_text(combined, encoding="utf-8")
    partial.replace(out_path)
    shutil.rmtree(cache_dir, ignore_errors=True)
    return len(combined)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PDFs to Markdown for the DM RAG pipeline.")
    ap.add_argument("--input",      default=DEFAULT_INPUT,  help="folder of source PDFs")
    ap.add_argument("--output",     default=DEFAULT_OUTPUT, help="folder for .md output")
    ap.add_argument("--file",       default=None,           help="process a single PDF instead of the whole folder")
    ap.add_argument("--pages",      type=int, default=None, help="max pages to process (default: all)")
    ap.add_argument("--start-page", type=int, default=1,    help="1-based page to start from (default: 1)")
    ap.add_argument("--force",      action="store_true",    help="re-process even if .md already exists")
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

    print(f"{len(pdfs)} PDF(s) to process\n")

    had_failure = False

    for pdf in pdfs:
        out_path = out_dir / f"{pdf.stem}.md"

        if out_path.exists() and not args.force:
            print(f"SKIP  {pdf.name}  (already done — use --force to redo)")
            continue

        doc   = fitz.open(pdf)
        total = len(doc)
        last  = min(total, first + args.pages) if args.pages else total
        doc.close()

        print(f"── {pdf.name}  ({total} pages, converting {last - first}) ──")

        try:
            chars = _extract_with_mineru(pdf, out_path, first, last, args.force)
        except subprocess.CalledProcessError as e:
            print(f"  FAILED: mineru exited with code {e.returncode}", file=sys.stderr)
            had_failure = True
            continue

        size_kb = out_path.stat().st_size // 1024
        print(f"  Wrote {out_path.name}  ({chars:,} chars, {size_kb} KB)")

        # Don't move the file when targeting a single page range — it may need
        # another pass for other pages.
        if not args.file:
            pdf.rename(done_dir / pdf.name)
            print(f"  Moved → done/")

        print()

    print("Done.")
    # A caller chaining this into a pipeline (e.g. an overnight ingest queue)
    # needs the exit code to actually reflect failure — a caught
    # CalledProcessError was silently swallowed here before, so a crashed
    # mineru run still reported success (exit 0) up the chain.
    if had_failure:
        sys.exit(1)


if __name__ == "__main__":
    main()
