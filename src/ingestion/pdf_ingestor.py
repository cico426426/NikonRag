from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks: list[str] = []
    for piece in splitter.split_text(text):
        normalized = _normalize_text(piece)
        if normalized:
            chunks.append(normalized)
    return chunks


def _chunk_id(source_file: str, start_page: int, end_page: int, idx: int, text: str) -> str:
    key = f"{source_file}|{start_page}|{end_page}|{idx}|{text[:120]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_pdf_documents(
    pdf_path: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    bridge_window: int,
) -> list[Document]:
    path = Path(pdf_path).resolve()
    page_chunks = pymupdf4llm.to_markdown(str(path), page_chunks=True)

    pages: list[tuple[int, str]] = []
    for page_chunk in page_chunks:
        page_number = int(page_chunk["metadata"].get("page_number", 1))
        pages.append((page_number, page_chunk.get("text", "")))

    docs: list[Document] = []
    filename = path.name

    for page_number, page_text in pages:
        pieces = _split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            cid = _chunk_id(str(path), page_number, page_number, idx, piece)
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": cid,
                        "source_file": str(path),
                        "source_filename": filename,
                        "page": page_number,
                        "start_page": page_number,
                        "end_page": page_number,
                        "chunk_type": "page",
                        "snippet_preview": piece[:220],
                    },
                )
            )

    for idx in range(len(pages) - 1):
        current_page, current_text = pages[idx]
        next_page, next_text = pages[idx + 1]
        left = _normalize_text(current_text)[-bridge_window:]
        right = _normalize_text(next_text)[:bridge_window]
        bridge_text = _normalize_text(f"{left}\n{right}")
        if not bridge_text:
            continue

        cid = _chunk_id(str(path), current_page, next_page, idx, bridge_text)
        docs.append(
            Document(
                page_content=bridge_text,
                metadata={
                    "doc_id": cid,
                    "source_file": str(path),
                    "source_filename": filename,
                    "page": current_page,
                    "start_page": current_page,
                    "end_page": next_page,
                    "chunk_type": "cross_page",
                    "snippet_preview": bridge_text[:220],
                },
            )
        )

    return docs
