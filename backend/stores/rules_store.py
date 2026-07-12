import re
from pathlib import Path

from langchain_chroma import Chroma
from pydantic import BaseModel

from backend.config import settings
from backend.llm import ollama_embeddings
from backend.rag.hybrid import BM25Index, reciprocal_rank_fusion
from backend.rag.reranker import LLMJudgeReranker, Reranker


class RuleChunk(BaseModel):
    book: str
    section: str
    content: str
    chunk_id: str = ""
    parent_chunk_id: str = ""


# A decorative drop-cap (the oversized first letter of a flavor-text
# paragraph, e.g. the "W" in "WIZARDS ARE DEFINED...") occasionally gets
# OCR'd/parsed as its own one-character "# W" section header instead of
# staying part of the paragraph it belongs to — confirmed live via
# eval_retrieval.py: PHB has 3 of these, Monster Manual 26, DMG 0
# (`grep -c '^# [A-Z]$'` against the indexed markdown). The resulting
# section is short and keyword-dense (it's the start of the real prose,
# just missing its first letter), which can out-rank the correctly-parsed
# section on BM25's length-normalized scoring — confirmed live: "What is
# the Wizard class like?" hybrid-ranked the bogus "W" section above the
# real "WIZARD CLASS FEATURES" section. A real OCR/chunking fix belongs
# upstream (the markdown header-detection step) and would need a reindex;
# this is the cheap retrieval-time mitigation — drop any single-letter
# section from the candidate pool before it can compete for a rank slot.
_DROP_CAP_SECTION_RE = re.compile(r"^[A-Za-z]$")


def _is_drop_cap_artifact(chunk: "RuleChunk") -> bool:
    return bool(_DROP_CAP_SECTION_RE.match(chunk.section))


class RulesStore:
    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        ollama_base_url: str = settings.ollama_base_url,
        reranker: Reranker | None = None,
    ) -> None:
        self._persist_dir = persist_dir
        self._ollama_base_url = ollama_base_url
        self._store: Chroma | None = None
        # LLMJudgeReranker (Ollama-based) rather than CrossEncoderReranker —
        # this is a low-throughput single-user app, so the extra Ollama
        # round-trip's latency is negligible next to the mechanics/narrator
        # calls already made every turn, and it avoids needing torch/
        # sentence-transformers in the container at all (real memory cost,
        # confirmed live: Docker Desktop's default memory allocation
        # couldn't even load the cross-encoder model without OOM-killing).
        self._reranker: Reranker = reranker if reranker is not None else LLMJudgeReranker()
        self._bm25: BM25Index | None = None
        self._bm25_loaded = False

    def load(self) -> None:
        """Open the existing ChromaDB collection. No-op if chroma_db doesn't exist yet."""
        if not Path(self._persist_dir).exists():
            return
        embeddings = ollama_embeddings(base_url=self._ollama_base_url)
        self._store = Chroma(
            collection_name="rules",
            embedding_function=embeddings,
            persist_directory=self._persist_dir,
        )

    def is_ready(self) -> bool:
        """False until build_index.py has been run and load() called. Note:
        this only checks that a Chroma collection exists — it does NOT
        detect whether that collection still uses the pre-Stage-0 schema
        (no granularity/chunk_id metadata). A collection built before the
        parent/child split will return empty results from search()/
        search_adventure_only() (their granularity filter matches nothing)
        until `make reindex-full` is run. There is no in-place Chroma schema
        migration — a full rebuild is required, same as build_index.py's own
        docstring states."""
        return self._store is not None

    def _get_bm25(self) -> BM25Index | None:
        """Lazy-loaded from data/bm25_rules.pkl (built by build_index.py).
        Returns None (dense-only search) if it hasn't been built yet — a
        graceful degrade, not a crash, since this file may not exist right
        after a fresh clone/migration."""
        if not self._bm25_loaded:
            path = str(Path(self._persist_dir).parent / "bm25_rules.pkl")
            self._bm25 = BM25Index.load(path)
            self._bm25_loaded = True
        return self._bm25

    @staticmethod
    def _build_where(books_in_play: list[str] | None, extra: dict | None = None) -> dict | None:
        if books_in_play is None:
            where = None
        elif not books_in_play:
            where = {"source_type": {"$eq": "core"}}
        else:
            where = {"$or": [
                {"source_type": {"$eq": "core"}},
                {"adventure": {"$in": books_in_play}},
            ]}
        if extra is None:
            return where
        if where is None:
            return extra
        return {"$and": [where, extra]}

    def _hydrate(self, chunk_ids: list[str]) -> dict[str, RuleChunk]:
        """Fetch content+metadata for a list of chunk_ids in one call,
        returning {chunk_id: RuleChunk}. Missing ids are simply absent from
        the result (e.g. a BM25-only hit whose id since disappeared from a
        stale pickle)."""
        if not chunk_ids or self._store is None:
            return {}
        raw = self._store._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        result: dict[str, RuleChunk] = {}
        for cid, doc, meta in zip(raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])):
            meta = meta or {}
            result[cid] = RuleChunk(
                book=meta.get("book", "Unknown"),
                section=meta.get("section", "Unknown"),
                content=doc or "",
                chunk_id=cid,
                parent_chunk_id=meta.get("parent_chunk_id", ""),
            )
        return result

    def _expand_to_parents(self, chunks: list[RuleChunk], k: int) -> list[RuleChunk]:
        """Replace each child chunk with its full parent section (one Chroma
        get() call for all needed parents), deduping if two surviving
        children share a parent — the "parent-document retrieval" pattern,
        achieved via existing metadata rather than a second docstore.

        `chunks` should be reranked-in-full (not pre-truncated to k) so this
        can backfill: walks the reranked order and stops once k DISTINCT
        parents are collected, rather than truncating to k children first
        and deduping after. Confirmed live (recall@k eval, 2026-07-08) that
        truncate-then-dedupe can silently shrink well below k — a short
        section (e.g. a class's flavor-text intro) split across several
        child sub-chunks can occupy most of the pre-dedup top-k on its own,
        crowding out a more relevant, differently-titled section (e.g. that
        same class's "CLASS FEATURES" section) that ranked just below the
        old cutoff. Backfilling from the fuller reranked order fixes that
        without changing the reranker itself."""
        parent_ids = [c.parent_chunk_id for c in chunks if c.parent_chunk_id]
        parents = self._hydrate(list(dict.fromkeys(parent_ids))) if parent_ids else {}

        results: list[RuleChunk] = []
        seen_parents: set[str] = set()
        for c in chunks:
            if len(results) >= k:
                break
            if c.parent_chunk_id and c.parent_chunk_id in parents:
                if c.parent_chunk_id in seen_parents:
                    continue
                seen_parents.add(c.parent_chunk_id)
                results.append(parents[c.parent_chunk_id])
            elif not c.parent_chunk_id:
                # No parent on record (e.g. this hit already was a parent,
                # or the index predates the parent/child split) — return it
                # as-is rather than dropping it.
                results.append(c)
        return results

    def search(
        self,
        query: str,
        k: int = 6,
        books_in_play: list[str] | None = None,
        wide_k: int = 30,
        use_reranker: bool = False,
    ) -> list[RuleChunk]:
        """Hybrid search: dense (Chroma) + BM25, fused via Reciprocal Rank
        Fusion, optionally reranked, then each surviving child chunk is
        expanded to its full parent section.

        Core books are always included. Pass books_in_play (list of adventure
        slugs from Campaign.books_in_play) to also search those adventures.
        None means no filter — searches everything (useful for admin/debug).

        use_reranker defaults to False — see design.md's Evolution section,
        2026-07-09: the reranker (LLMJudgeReranker) makes its own separate
        ChatOllama call, distinct from the embedding model this method's dense
        step already uses, so calling it here forces Ollama to evict one
        model and load the other on every single call — the exact embed<->chat
        swap already root-caused as the trigger for the whole-app MLX-runner
        freeze (world-prep's freeze, 2026-07-08; the same fix — search_rules/
        search_lore, the two live-gameplay callers — is applied here rather
        than only to the two call sites that got it back then). RRF-fused
        dense+BM25 is a legitimate hybrid retrieval result on its own even
        without a reranked reorder on top — this trades a bit of ranking
        precision for not freezing the app on every rules/lore lookup, which
        given how constantly this is called during live play (every combat
        start alone can trigger several) is the right side of that trade.
        Pass True explicitly for a lower-frequency, quality-sensitive caller
        that can tolerate the swap risk (e.g. scripts/eval_retrieval.py,
        which is specifically measuring reranked quality offline)."""
        if self._store is None:
            self.load()
        if not self._store:
            raise RuntimeError(
                "RulesStore is not ready. Run build_index.py first, "
                "then restart the app."
            )

        where = self._build_where(books_in_play, {"granularity": {"$eq": "child"}})

        dense_hits = self._store.similarity_search(query, k=wide_k, filter=where)
        dense_ids = [d.metadata.get("chunk_id") for d in dense_hits if d.metadata.get("chunk_id")]

        bm25 = self._get_bm25()
        bm25_ids = [cid for cid, _ in bm25.search(query, k=wide_k, where=where)] if bm25 else []

        fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:wide_k] if bm25_ids else dense_ids[:wide_k]
        if not fused_ids:
            return []

        hydrated = self._hydrate(fused_ids)
        candidates = [hydrated[cid] for cid in fused_ids if cid in hydrated and not _is_drop_cap_artifact(hydrated[cid])]

        if use_reranker:
            # Rerank the FULL candidate set (top_n=len(candidates), not k) —
            # _expand_to_parents needs the fuller order to backfill past a
            # dedup collapse; both Reranker implementations already compute
            # the complete ranking internally before slicing to top_n, so
            # this costs nothing extra (same single LLM/cross-encoder call
            # either way). See this method's docstring for why this is opt-in.
            candidates = self._reranker.rerank(query, candidates, top_n=len(candidates))
        return self._expand_to_parents(candidates, k)

    def search_adventure_only(self, query: str, adventure: str, k: int = 4) -> list[RuleChunk]:
        """Search only the given adventure's indexed text — no core rulebook
        fallback. Core books vastly outnumber a single adventure's chunks, so
        a mixed search() for generic worldbuilding queries tends to surface
        core DMG advice instead of the adventure's own named locations. Used
        by world-prep, which wants this adventure's own geography. Kept as
        plain dense similarity search (not the hybrid pipeline) — this is a
        narrower, lower-stakes lookup than search()'s general case.
        """
        if self._store is None:
            self.load()
        if not self._store:
            raise RuntimeError(
                "RulesStore is not ready. Run build_index.py first, "
                "then restart the app."
            )
        hits = self._store.similarity_search(
            query, k=k, filter={"$and": [{"adventure": {"$eq": adventure}}, {"granularity": {"$eq": "child"}}]}
        )
        return [
            RuleChunk(
                book=doc.metadata.get("book", "Unknown"),
                section=doc.metadata.get("section", "Unknown"),
                content=doc.page_content,
                chunk_id=doc.metadata.get("chunk_id", ""),
                parent_chunk_id=doc.metadata.get("parent_chunk_id", ""),
            )
            for doc in hits
        ]

    def search_adventure_literal(
        self,
        query: str,
        books_in_play: list[str],
        context_chars: int = 200,
        limit: int | None = None,
    ) -> list[RuleChunk]:
        """Literal, case-insensitive full-text search across every page of the
        given adventure book(s) — NOT semantic/vector search like search()/
        search_adventure_only(). Reads the raw source markdown directly
        (docs/source/adventures/{slug}/*.md), so it needs no index and finds
        every scattered mention of a name/place/item across the WHOLE book,
        which top-k similarity search can easily miss. `limit` caps the
        number of matches returned (None = unbounded, the original in-game
        tool's behavior); `context_chars` controls the window on each side of
        a match. Shared by the search_adventure_literal tool (backend/tools/
        rules.py) and world_prep.py's opening-scene seeding, which was the
        reason this got pulled out of the tool closure in the first place."""
        results = []
        for slug in books_in_play:
            for path in Path(f"docs/source/adventures/{slug}").glob("*.md"):
                text = path.read_text(encoding="utf-8")
                for m in re.finditer(re.escape(query), text, re.IGNORECASE):
                    start = max(0, m.start() - context_chars)
                    end = min(len(text), m.end() + context_chars)
                    results.append(RuleChunk(book=slug, section=path.stem, content=f"…{text[start:end]}…"))
                    if limit is not None and len(results) >= limit:
                        return results
        return results
