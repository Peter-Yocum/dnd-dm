"""
HistoryStore — ChromaDB collection for session chronicles.

Each ended session is embedded and stored here so the DM agent can do
semantic search across past events rather than blindly injecting all history
into every context window.
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from backend.config import settings


class HistoryStore:
    COLLECTION = "session_chronicles"

    def __init__(
        self,
        persist_dir: str = settings.chroma_persist_dir,
        ollama_base_url: str = settings.ollama_base_url,
    ) -> None:
        self._persist_dir = persist_dir
        self._ollama_base_url = ollama_base_url
        self._store: Chroma | None = None

    def _chroma(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self.COLLECTION,
                embedding_function=OllamaEmbeddings(
                    model="nomic-embed-text",
                    base_url=self._ollama_base_url,
                ),
                persist_directory=self._persist_dir,
            )
        return self._store

    def add_session(
        self,
        campaign_id: str,
        session_id: str,
        session_number: int,
        text: str,
    ) -> None:
        self._chroma().add_documents(
            [Document(
                page_content=text,
                metadata={
                    "campaign_id": campaign_id,
                    "session_id": session_id,
                    "session_number": session_number,
                },
            )],
            ids=[session_id],
        )

    def search(self, query: str, campaign_id: str, k: int = 3) -> list[Document]:
        try:
            return self._chroma().similarity_search(
                query,
                k=k,
                filter={"campaign_id": {"$eq": campaign_id}},
            )
        except Exception:
            return []
