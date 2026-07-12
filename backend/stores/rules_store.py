import asyncio
import re
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.llm import ollama_embeddings
from backend.rag.reranker import LLMJudgeReranker, Reranker
from backend.rag.hybrid import reciprocal_rank_fusion
from backend.stores import tables as t


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


def _row_to_chunk(row) -> RuleChunk:
    return RuleChunk(
        book=row.book, section=row.section, content=row.content,
        chunk_id=row.chunk_id, parent_chunk_id=row.parent_chunk_id,
    )


class RulesStore:
    """Rules corpus (core rulebooks + adventures) — Postgres/pgvector +
    native full-text search, replacing ChromaDB (2026-07-12 migration, see
    design.md's Tech Stack table). Hybrid retrieval: dense (pgvector cosine
    KNN via the `embedding` column's HNSW index) + sparse (Postgres
    `tsvector`/GIN, always in sync with `content` since it's a generated
    column — no separate rebuild step, unlike ChromaDB-era BM25/FTS5
    sidecar files), fused via reciprocal_rank_fusion() (unchanged, storage-
    agnostic). Confirmed live: ChromaDB's own local vector index needed
    ~2.6GB resident just to query the 441k-row "rules" collection,
    independent of any keyword-index approach — this is why the fix is a
    storage-engine swap, not a keyword-index optimization."""

    def __init__(
        self,
        engine: AsyncEngine,
        ollama_base_url: str | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._engine = engine
        self._embeddings = ollama_embeddings(base_url=ollama_base_url) if ollama_base_url else ollama_embeddings()
        # LLMJudgeReranker (Ollama-based) rather than CrossEncoderReranker —
        # this is a low-throughput single-user app, so the extra Ollama
        # round-trip's latency is negligible next to the mechanics/narrator
        # calls already made every turn, and it avoids needing torch/
        # sentence-transformers in the container at all (real memory cost,
        # confirmed live: Docker Desktop's default memory allocation
        # couldn't even load the cross-encoder model without OOM-killing).
        self._reranker: Reranker = reranker if reranker is not None else LLMJudgeReranker()
        self._ready: bool | None = None

    async def is_ready(self) -> bool:
        """False until build_index.py has populated rule_chunks. Cached
        after the first True (a populated corpus never becomes empty again
        except via an explicit --wipe reindex, which is an offline operation
        this process wouldn't be live for)."""
        if self._ready:
            return True
        async with self._engine.connect() as conn:
            result = await conn.execute(select(t.rule_chunks.c.chunk_id).limit(1))
            self._ready = result.first() is not None
        return self._ready

    @staticmethod
    def _books_predicate(books_in_play: list[str] | None):
        """Mirrors LoreStore's _book_scope: None = no filter (search
        everything — admin/debug); [] = core books only; a list = core +
        those adventures. Returns None for "no predicate" (caller must
        handle that case, same as the old Chroma `where=None` convention)."""
        if books_in_play is None:
            return None
        if not books_in_play:
            return t.rule_chunks.c.source_type == "core"
        return or_(
            t.rule_chunks.c.source_type == "core",
            t.rule_chunks.c.adventure.in_(books_in_play),
        )

    async def _hydrate(self, chunk_ids: list[str]) -> dict[str, RuleChunk]:
        """Fetch content+metadata for a list of chunk_ids in one call,
        returning {chunk_id: RuleChunk}. Missing ids are simply absent from
        the result (e.g. a stale id from a source that's since changed)."""
        if not chunk_ids:
            return {}
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(
                    t.rule_chunks.c.chunk_id, t.rule_chunks.c.book, t.rule_chunks.c.section,
                    t.rule_chunks.c.content, t.rule_chunks.c.parent_chunk_id,
                ).where(t.rule_chunks.c.chunk_id.in_(chunk_ids))
            )
            return {row.chunk_id: _row_to_chunk(row) for row in result}

    async def _expand_to_parents(self, chunks: list[RuleChunk], k: int) -> list[RuleChunk]:
        """Replace each child chunk with its full parent section (one query
        for all needed parents), deduping if two surviving children share a
        parent — the "parent-document retrieval" pattern.

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
        parents = await self._hydrate(list(dict.fromkeys(parent_ids))) if parent_ids else {}

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
                # No parent on record (e.g. this hit already was a parent) —
                # return it as-is rather than dropping it.
                results.append(c)
        return results

    async def search(
        self,
        query: str,
        k: int = 6,
        books_in_play: list[str] | None = None,
        wide_k: int = 30,
        use_reranker: bool = False,
    ) -> list[RuleChunk]:
        """Hybrid search: dense (pgvector) + sparse (Postgres full-text),
        fused via Reciprocal Rank Fusion, optionally reranked, then each
        surviving child chunk is expanded to its full parent section.

        Core books are always included. Pass books_in_play (list of adventure
        slugs/display names from Campaign.books_in_play) to also search those
        adventures. None means no filter — searches everything (admin/debug).

        use_reranker defaults to False — see design.md's Evolution section,
        2026-07-09: the reranker (LLMJudgeReranker) makes its own separate
        chat call, distinct from the embedding model this method's dense step
        already uses; at the time, both ran through Ollama, so calling it
        here forced Ollama to evict one model and load the other on every
        single call — the exact embed<->chat swap already root-caused as a
        whole-app MLX-runner freeze trigger. That specific swap risk is now
        moot (2026-07-13, vllm-migration-plan.md): chat moved to a separate
        vLLM-metal server, so the reranker's call no longer touches Ollama
        (which now only serves embeddings) at all. Left as False for now
        anyway — the extra vLLM round-trip's latency on this hot path is
        still a real, independent reason, just a smaller one than an
        outright freeze risk; worth reconsidering as a real default-flip
        candidate now that the swap-freeze reason is gone, not done here.
        RRF-fused dense+sparse is a legitimate hybrid retrieval result on its
        own even without a reranked reorder on top. Pass True explicitly for
        a lower-frequency, quality-sensitive caller (e.g.
        scripts/eval_retrieval.py)."""
        if not await self.is_ready():
            raise RuntimeError(
                "RulesStore is not ready. Run build_index.py first, "
                "then restart the app."
            )

        books_pred = self._books_predicate(books_in_play)
        base = t.rule_chunks.c.granularity == "child"
        where_clause = and_(base, books_pred) if books_pred is not None else base

        query_vec = await asyncio.to_thread(self._embeddings.embed_query, query)

        async with self._engine.connect() as conn:
            dense_result = await conn.execute(
                select(t.rule_chunks.c.chunk_id)
                .where(where_clause)
                .order_by(t.rule_chunks.c.embedding.cosine_distance(query_vec))
                .limit(wide_k)
            )
            dense_ids = [row.chunk_id for row in dense_result]

            tsquery = func.plainto_tsquery("english", query)
            sparse_result = await conn.execute(
                select(t.rule_chunks.c.chunk_id)
                .where(and_(where_clause, t.rule_chunks.c.content_tsv.op("@@")(tsquery)))
                .order_by(func.ts_rank_cd(t.rule_chunks.c.content_tsv, tsquery).desc())
                .limit(wide_k)
            )
            bm25_ids = [row.chunk_id for row in sparse_result]

        fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids])[:wide_k] if bm25_ids else dense_ids[:wide_k]
        if not fused_ids:
            return []

        hydrated = await self._hydrate(fused_ids)
        candidates = [hydrated[cid] for cid in fused_ids if cid in hydrated and not _is_drop_cap_artifact(hydrated[cid])]

        if use_reranker:
            # Rerank the FULL candidate set (top_n=len(candidates), not k) —
            # _expand_to_parents needs the fuller order to backfill past a
            # dedup collapse; both Reranker implementations already compute
            # the complete ranking internally before slicing to top_n, so
            # this costs nothing extra (same single LLM/cross-encoder call
            # either way). See this method's docstring for why this is opt-in.
            candidates = await asyncio.to_thread(self._reranker.rerank, query, candidates, len(candidates))
        return await self._expand_to_parents(candidates, k)

    async def search_adventure_only(self, query: str, adventure: str, k: int = 4) -> list[RuleChunk]:
        """Search only the given adventure's indexed text — no core rulebook
        fallback. Core books vastly outnumber a single adventure's chunks, so
        a mixed search() for generic worldbuilding queries tends to surface
        core DMG advice instead of the adventure's own named locations. Used
        by world-prep, which wants this adventure's own geography. Plain
        dense similarity search (not the hybrid pipeline) — this is a
        narrower, lower-stakes lookup than search()'s general case."""
        if not await self.is_ready():
            raise RuntimeError(
                "RulesStore is not ready. Run build_index.py first, "
                "then restart the app."
            )
        query_vec = await asyncio.to_thread(self._embeddings.embed_query, query)
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(
                    t.rule_chunks.c.chunk_id, t.rule_chunks.c.book, t.rule_chunks.c.section,
                    t.rule_chunks.c.content, t.rule_chunks.c.parent_chunk_id,
                )
                .where(and_(t.rule_chunks.c.adventure == adventure, t.rule_chunks.c.granularity == "child"))
                .order_by(t.rule_chunks.c.embedding.cosine_distance(query_vec))
                .limit(k)
            )
            return [_row_to_chunk(row) for row in result]

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
        (docs/source/adventures/{slug}/*.md), so it needs no index/DB access
        at all and finds every scattered mention of a name/place/item across
        the WHOLE book, which top-k similarity search can easily miss.
        `limit` caps the number of matches returned (None = unbounded, the
        original in-game tool's behavior); `context_chars` controls the
        window on each side of a match. Shared by the search_adventure_literal
        tool (backend/tools/rules.py) and world_prep.py's opening-scene
        seeding, which was the reason this got pulled out of the tool closure
        in the first place. Deliberately synchronous — no I/O beyond local
        file reads, unlike every other method on this class."""
        results = []
        for slug in books_in_play:
            for path in Path(f"docs/source/adventures/{slug}").glob("*.md"):
                text_ = path.read_text(encoding="utf-8")
                for m in re.finditer(re.escape(query), text_, re.IGNORECASE):
                    start = max(0, m.start() - context_chars)
                    end = min(len(text_), m.end() + context_chars)
                    results.append(RuleChunk(book=slug, section=path.stem, content=f"…{text_[start:end]}…"))
                    if limit is not None and len(results) >= limit:
                        return results
        return results
