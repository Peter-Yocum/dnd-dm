#!/usr/bin/env python3
"""
eval_retrieval.py — hand-labeled recall@k harness for RulesStore.search().

Loads scripts/eval/retrieval_questions.json (a curated list of
{"query", "expected_book", "expected_section_contains"} entries, each
grounded in a real header confirmed present in the indexed source markdown —
see that file) and reports recall@k: for what fraction of questions did at
least one returned chunk match the expected book (substring, case-
insensitive) and section (substring, case-insensitive)?

--baseline runs a plain dense-only pgvector KNN query directly (no full-text/
RRF/rerank/parent-expansion) to emulate this project's pre-Stage-0 retrieval
behavior, for a real side-by-side comparison without needing to check out an
old branch. No RAGAS/DeepEval needed for this — plain Python arithmetic,
matching validate_source.py's existing no-framework precedent.

Usage:
    python eval_retrieval.py                # evaluate the current hybrid search()
    python eval_retrieval.py --baseline      # evaluate dense-only KNN only
    python eval_retrieval.py --k 8
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine

from backend.config import settings
from backend.stores import tables as t
from backend.stores.rules_store import RuleChunk, RulesStore

QUESTIONS_PATH = Path(__file__).parent / "eval" / "retrieval_questions.json"


def _matches(chunk: RuleChunk, expected_book: str, expected_section_contains: str) -> bool:
    return (
        expected_book.lower() in chunk.book.lower()
        and expected_section_contains.lower() in chunk.section.lower()
    )


async def _dense_only_search(store: RulesStore, engine, query: str, k: int) -> list[RuleChunk]:
    """Emulates pre-Stage-0 retrieval: plain pgvector KNN, no full-text
    fusion, no rerank, no parent expansion — reimplemented here rather than
    exposed on RulesStore itself since it's only ever needed for this
    historical comparison, not live gameplay."""
    query_vec = await asyncio.to_thread(store._embeddings.embed_query, query)
    async with engine.connect() as conn:
        result = await conn.execute(
            select(
                t.rule_chunks.c.chunk_id, t.rule_chunks.c.book, t.rule_chunks.c.section,
                t.rule_chunks.c.content, t.rule_chunks.c.parent_chunk_id,
            )
            .where(t.rule_chunks.c.granularity == "child")
            .order_by(t.rule_chunks.c.embedding.cosine_distance(query_vec))
            .limit(k)
        )
        return [
            RuleChunk(book=r.book, section=r.section, content=r.content, chunk_id=r.chunk_id, parent_chunk_id=r.parent_chunk_id)
            for r in result
        ]


async def run_eval(k: int, baseline: bool) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    engine = create_async_engine(settings.database_url)
    store = RulesStore(engine, settings.ollama_base_url)
    if not await store.is_ready():
        print("RulesStore is not ready — run build_index.py first.")
        sys.exit(1)

    hits = 0
    for q in questions:
        query = q["query"]
        if baseline:
            chunks = await _dense_only_search(store, engine, query, k)
        else:
            # use_reranker=True — this eval specifically measures reranked
            # quality; the live-gameplay callers (search_rules/search_lore)
            # default to False now (see RulesStore.search()'s docstring).
            chunks = await store.search(query, k=k, books_in_play=None, use_reranker=True)

        found = any(_matches(c, q["expected_book"], q["expected_section_contains"]) for c in chunks)
        print(f"  [{'OK  ' if found else 'MISS'}] {query}")
        if found:
            hits += 1

    total = len(questions)
    mode = "baseline (dense-only KNN)" if baseline else "hybrid search()"
    if total:
        print(f"\nRecall@{k} ({mode}): {hits}/{total} = {hits / total:.1%}")
    else:
        print("No questions loaded.")
    await engine.dispose()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6, help="top-k to evaluate")
    ap.add_argument("--baseline", action="store_true",
                     help="use dense-only KNN instead of the hybrid pipeline")
    args = ap.parse_args()
    asyncio.run(run_eval(args.k, args.baseline))


if __name__ == "__main__":
    main()
