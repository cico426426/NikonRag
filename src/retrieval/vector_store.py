from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class VectorIndex:
    def __init__(self, *, persist_directory: Path, embedding_model: str, api_key: str) -> None:
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, api_key=api_key)
        self.store = Chroma(
            collection_name="nikon_manuals",
            embedding_function=self.embeddings,
            persist_directory=str(persist_directory),
        )

    def add_documents(self, documents: list[Document]) -> None:
        if documents:
            self.store.add_documents(documents)

    def retrieve_with_scores(
        self,
        query: str,
        *,
        k: int,
        source_file: str | None = None,
    ) -> list[tuple[Document, float]]:
        filter_payload = {"source_file": source_file} if source_file else None
        return self.store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter_payload,
        )

    def list_source_files(self) -> list[str]:
        raw = self.store.get(include=["metadatas"])
        files = {m.get("source_file") for m in raw.get("metadatas", []) if m.get("source_file")}
        return sorted(files)
