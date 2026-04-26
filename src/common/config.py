from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str
    chroma_dir: Path
    uploads_dir: Path
    default_pdf_path: Path | None
    chunk_size: int
    chunk_overlap: int
    bridge_window: int
    recall_k: int
    rerank_top_n: int
    use_rerank: bool


def _as_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    root = Path(__file__).resolve().parents[2]
    uploads = root / "data" / "uploads"
    default_pdf = os.getenv("DEFAULT_PDF_PATH")
    default_pdf_path = Path(default_pdf).resolve() if default_pdf else None

    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        chroma_dir=Path(os.getenv("CHROMA_DIR", str(root / "data" / "chroma"))).resolve(),
        uploads_dir=Path(os.getenv("UPLOADS_DIR", str(uploads))).resolve(),
        default_pdf_path=default_pdf_path,
        chunk_size=_as_int("CHUNK_SIZE", 900),
        chunk_overlap=_as_int("CHUNK_OVERLAP", 150),
        bridge_window=_as_int("BRIDGE_WINDOW", 250),
        recall_k=_as_int("RECALL_K", 30),
        rerank_top_n=_as_int("RERANK_TOP_N", 6),
        use_rerank=_as_bool("USE_RERANK", False),
    )
