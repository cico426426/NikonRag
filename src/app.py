from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.agent.rag_agent import RagAgent
from src.common.config import Settings, get_settings
from src.common.types import RagAnswer
from src.ingestion.pdf_ingestor import load_pdf_documents
from src.retrieval.vector_store import VectorIndex


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        load_dotenv()
        self.settings = settings or get_settings()
        self.settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.settings.uploads_dir.mkdir(parents=True, exist_ok=True)

        self.vector_index = VectorIndex(
            persist_directory=self.settings.chroma_dir,
            embedding_model=self.settings.gemini_embedding_model,
            api_key=self.settings.gemini_api_key,
        )
        self.agent = RagAgent(
            vector_index=self.vector_index,
            model=self.settings.gemini_model,
            api_key=self.settings.gemini_api_key,
            recall_k=self.settings.recall_k,
            rerank_top_n=self.settings.rerank_top_n,
            use_rerank=self.settings.use_rerank,
        )

    def ingest(self, pdf_path: str | Path) -> int:
        docs = load_pdf_documents(
            pdf_path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            bridge_window=self.settings.bridge_window,
            enable_image_descriptions=self.settings.enable_image_descriptions,
            image_max_per_page=self.settings.image_max_per_page,
            image_min_bytes=self.settings.image_min_bytes,
            image_description_model=self.settings.gemini_model,
            image_description_api_key=self.settings.gemini_api_key,
        )
        self.vector_index.add_documents(docs)
        return len(docs)

    def ask(self, question: str, *, source_file: str | None = None) -> RagAnswer:
        return self.agent.ask(question, source_file=source_file)

    def list_documents(self) -> list[str]:
        return self.vector_index.list_source_files()
