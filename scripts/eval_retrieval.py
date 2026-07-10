#!/usr/bin/env python3
"""
eval_retrieval.py — hand-labeled recall@k harness for RulesStore.search().

Loads scripts/eval/retrieval_questions.json (a curated list of
{"query", "expected_book", "expected_section_contains"} entries, each
grounded in a real header confirmed present in the indexed source markdown —
see that file) and reports recall@k: for what fraction of questions did at
least one returned chunk match the expected book (substring, case-
insensitive) and section (substring, case-insensitive)?

--baseline runs a plain naive Chroma similarity_search(k) directly (no
hybrid/BM25/rerank/parent-expansion) to emulate this project's pre-Stage-0
retrieval behavior, for a real side-by-side comparison without needing to
check out an old branch. No RAGAS/DeepEval needed for this — plain Python
arithmetic, matching validate_source.py's existing no-framework precedent.

Usage:
    python eval_retrieval.py                # evaluate the current hybrid search()
    python eval_retrieval.py --baseline      # evaluate naive similarity_search only
    python eval_retrieval.py --k 8
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.stores.rules_store import RuleChunk, RulesStore

QUESTIONS_PATH = Path(__file__).parent / "eval" / "retrieval_questions.json"


def _matches(chunk: RuleChunk, expected_book: str, expected_section_contains: str) -> bool:
    return (
        expected_book.lower() in chunk.book.lower()
        and expected_section_contains.lower() in chunk.section.lower()
    )


def run_eval(k: int, baseline: bool) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    store = RulesStore()
    store.load()
    if not store.is_ready():
        print("RulesStore is not ready — run build_index.py first.")
        sys.exit(1)

    hits = 0
    for q in questions:
        query = q["query"]
        if baseline:
            docs = store._store.similarity_search(query, k=k)
            chunks = [
                RuleChunk(
                    book=d.metadata.get("book", "Unknown"),
                    section=d.metadata.get("section", "Unknown"),
                    content=d.page_content,
                )
                for d in docs
            ]
        else:
            # use_reranker=True — this eval specifically measures reranked
            # quality; the live-gameplay callers (search_rules/search_lore)
            # default to False now (see RulesStore.search()'s docstring).
            chunks = store.search(query, k=k, books_in_play=None, use_reranker=True)

        found = any(_matches(c, q["expected_book"], q["expected_section_contains"]) for c in chunks)
        print(f"  [{'OK  ' if found else 'MISS'}] {query}")
        if found:
            hits += 1

    total = len(questions)
    mode = "baseline (naive similarity_search)" if baseline else "hybrid search()"
    if total:
        print(f"\nRecall@{k} ({mode}): {hits}/{total} = {hits / total:.1%}")
    else:
        print("No questions loaded.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=6, help="top-k to evaluate")
    ap.add_argument("--baseline", action="store_true",
                     help="use naive similarity_search instead of the hybrid pipeline")
    args = ap.parse_args()
    run_eval(args.k, args.baseline)


if __name__ == "__main__":
    main()
