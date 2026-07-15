# Research prompt — PDF map/image extraction → ASCII/grid recreation

Paste everything below into Claude Desktop (with web search / extended research enabled) as one prompt.

---

I'm building a local, self-hosted AI Dungeon Master web app for D&D 5e (Python/FastAPI backend, no cloud services by design). I need a deep-dive research pass on the current state of the art for automatically extracting maps out of published adventure PDFs and turning them into a symbolic/ASCII grid representation my app can render and reason over.

**Existing pipeline, for context:**
- `ocr_ingest.py`: PDF → Markdown. Tier 1 is native PDF text extraction; Tier 2 falls back to Apple Vision OCR (macOS-only) for scanned/garbled pages. This currently extracts **text only** — no images or figures are pulled out at all.
- Dungeon/battle maps are currently **100% hand-authored**: a DM manually writes out a `Location.grid` (a 2D array of symbols) plus a `legend` dict mapping symbols to meanings (wall, door, water, difficult terrain, furniture, etc.). A renderer (`map_render.py`) turns that grid into colored HTML cells (roguelike-style, monospace, small fixed palette keyed off legend keywords) for both a live combat map and a persistent "Maps" browser page. Fog-of-war is a separate reveal-radius overlay on top of the same grid.
- The whole system's guiding principle is "ground or abstain" — never invent something the source material doesn't actually support (this already governs RAG retrieval, loot generation, and world-building; a map extraction feature would need to follow the same rule rather than hallucinating plausible-looking dungeon layouts).

**The concrete problem that triggered this research:** I was reading through the published adventure *Out of the Abyss*, and at least one of its maps is drawn in an **isometric / angled perspective** rather than a clean top-down (orthographic) layout. Any naive "screenshot the map page → ask a model to describe a grid" approach will likely produce garbage or a misleading grid for that kind of art, and I suspect this isn't a one-off — TTRPG books mix top-down dungeon maps, isometric cutaways, and pure illustrative (non-mechanical) art on the same page or even in the same figure. I need this handled explicitly, not glossed over.

**Please research and report on:**

1. **PDF image/figure extraction.** Current best tools/libraries (e.g. PyMuPDF/fitz, pdfplumber, pdf2image + Poppler, pikepdf, commercial options like Adobe's PDF Extract API) for pulling embedded raster images and vector art out of a PDF at usable resolution. Include how to distinguish "this is a map" from "this is decorative art, a character portrait, a border, or a sidebar icon" on a page — is this typically solved with heuristics (image size/aspect ratio/position), a trained classifier, or a vision-LLM judgment call?

2. **Perspective/projection classification.** How well-solved is detecting whether an extracted map image is top-down/orthographic vs. isometric/angled/perspective art? Is there existing tooling or research for this specific classification, or does it realistically require a multimodal LLM to look at the image and judge? What's the practical failure mode of feeding an isometric map into a top-down grid-extraction pipeline (garbled grid vs. confidently-wrong grid vs. clean failure)?

3. **Map → structured grid conversion.** Current techniques for turning a top-down map image into a symbolic grid (walls/doors/floor/water/etc.):
   - Traditional computer vision (edge/line detection, grid-cell detection, flood-fill room segmentation)
   - Vision-LLM-based description (prompting a multimodal model to directly output a grid or room list)
   - Hybrid approaches (CV for geometry, LLM for semantic labeling)
   - Any existing open-source projects or papers specifically about TTRPG/battle-map digitization — there's a real hobbyist/OSS niche here (VTT map importers for Roll20/Foundry/Owlbear Rodeo, battle-map grid detectors, dungeon-generator-adjacent tools). Name specific projects if you find them, not just categories.

4. **Handling non-top-down maps specifically.** For isometric/angled art (the *Out of the Abyss* case) — and for purely illustrative, non-mechanical art — what's a realistic fallback? E.g., could a vision-LLM instead produce a textual room-connectivity description (rooms + doors + rough relative layout) rather than forcing a literal grid, when the source art doesn't support one? What does "abstain gracefully" look like in practice for this kind of pipeline?

5. **Local-only feasibility and cost.** Given the app's local-only stance (currently running a `Qwen3-30B-A3B` MoE checkpoint via vLLM-metal for text, `nomic-embed-text` via Ollama for embeddings, all on a single Mac), what parts of this pipeline could run fully locally vs. would realistically need a cloud vision API (e.g. GPT-4V/Claude/Gemini vision) for acceptable quality? This would run as a one-time offline batch job per adventure (not a runtime/per-request cost), so give a rough sense of compute/time/API cost at that usage pattern, not real-time constraints.

6. **Recent developments.** Anything published or released in roughly the last 12–18 months specifically about LLM/vision-model-assisted map digitization, document figure extraction, or TTRPG-specific tooling that's meaningfully better than what existed before.

**Conclude with:** a comparison table or summary of the realistic approaches (not just one recommendation), and a feasibility/complexity assessment specifically for a solo developer adding this as an offline batch-processing feature to an existing Python/FastAPI app with the grid/legend format described above — what's a reasonable MVP scope vs. what's a much bigger lift than it looks.
