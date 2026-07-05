from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel

from backend.config import settings


class RuleChunk(BaseModel):
    book: str
    section: str
    content: str


class RulesStore:
    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        ollama_base_url: str = settings.ollama_base_url,
    ) -> None:
        self._persist_dir = persist_dir
        self._ollama_base_url = ollama_base_url
        self._store: Chroma | None = None

    def load(self) -> None:
        """Open the existing ChromaDB collection. No-op if chroma_db doesn't exist yet."""
        if not Path(self._persist_dir).exists():
            return
        embeddings = OllamaEmbeddings(
            base_url=self._ollama_base_url,
            model="nomic-embed-text",
        )
        self._store = Chroma(
            collection_name="rules",
            embedding_function=embeddings,
            persist_directory=self._persist_dir,
        )

    def is_ready(self) -> bool:
        """False until build_index.py has been run and load() called."""
        return self._store is not None

    def search(
        self,
        query: str,
        k: int = 4,
        books_in_play: list[str] | None = None,
    ) -> list[RuleChunk]:
        """Search indexed rulebooks.

        Core books are always included. Pass books_in_play (list of adventure
        slugs from Campaign.books_in_play) to also search those adventures.
        None means no filter — searches everything (useful for admin/debug).
        """
        if self._store is None:
            self.load()
        if not self._store:
            raise RuntimeError(
                "RulesStore is not ready. Run build_index.py first, "
                "then restart the app."
            )
        if books_in_play is None:
            where = None
        elif not books_in_play:
            where = {"source_type": {"$eq": "core"}}
        else:
            where = {"$or": [
                {"source_type": {"$eq": "core"}},
                {"adventure": {"$in": books_in_play}},
            ]}
        hits = self._store.similarity_search(query, k=k, filter=where)
        return [
            RuleChunk(
                book=doc.metadata.get("book", "Unknown"),
                section=doc.metadata.get("section", "Unknown"),
                content=doc.page_content,
            )
            for doc in hits
        ]

    def search_adventure_only(self, query: str, adventure: str, k: int = 4) -> list[RuleChunk]:
        """Search only the given adventure's indexed text — no core rulebook
        fallback. Core books vastly outnumber a single adventure's chunks, so
        a mixed search() for generic worldbuilding queries tends to surface
        core DMG advice instead of the adventure's own named locations. Used
        by world-prep, which wants this adventure's own geography.
        """
        if self._store is None:
            self.load()
        if not self._store:
            raise RuntimeError(
                "RulesStore is not ready. Run build_index.py first, "
                "then restart the app."
            )
        hits = self._store.similarity_search(
            query, k=k, filter={"adventure": {"$eq": adventure}}
        )
        return [
            RuleChunk(
                book=doc.metadata.get("book", "Unknown"),
                section=doc.metadata.get("section", "Unknown"),
                content=doc.page_content,
            )
            for doc in hits
        ]
