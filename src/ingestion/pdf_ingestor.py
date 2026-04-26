from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

import fitz
import pymupdf4llm
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


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


def _mime_type_for_extension(ext: str) -> str:
    normalized = ext.lower().strip(".")
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "png":
        return "image/png"
    if normalized == "webp":
        return "image/webp"
    if normalized == "gif":
        return "image/gif"
    return "application/octet-stream"


def _extract_llm_text(raw_content: Any) -> str:
    if isinstance(raw_content, str):
        return raw_content.strip()
    if isinstance(raw_content, list):
        fragments: list[str] = []
        for item in raw_content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    fragments.append(text)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
        return " ".join(fragments).strip()
    return ""


def _build_image_caption_documents(
    *,
    pdf_path: Path,
    model: str,
    api_key: str,
    max_images_per_page: int,
    min_image_bytes: int,
) -> list[Document]:
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        api_key=api_key,
    )
    prompt = (
        "Describe this Nikon manual image for retrieval. "
        "Focus on controls, menu names, settings, and operation steps visible in the image. "
        "If text appears in the image, include key terms exactly. Keep it under 120 words."
    )

    docs: list[Document] = []
    pdf = fitz.open(str(pdf_path))
    filename = pdf_path.name
    try:
        for page_idx in range(pdf.page_count):
            page = pdf[page_idx]
            page_number = page_idx + 1
            images = page.get_images(full=True)
            if not images:
                continue

            kept = 0
            for image_idx, image_entry in enumerate(images, start=1):
                if kept >= max_images_per_page:
                    break

                xref = int(image_entry[0])
                image_payload = pdf.extract_image(xref)
                image_bytes = image_payload.get("image")
                if not isinstance(image_bytes, bytes) or len(image_bytes) < min_image_bytes:
                    continue

                ext = str(image_payload.get("ext", "png"))
                mime_type = _mime_type_for_extension(ext)
                image_data = base64.b64encode(image_bytes).decode("ascii")
                image_url = f"data:{mime_type};base64,{image_data}"
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
                )
                try:
                    response = llm.invoke([message])
                except Exception:
                    continue

                caption = _normalize_text(_extract_llm_text(getattr(response, "content", "")))
                if not caption:
                    continue

                kept += 1
                content = f"[image_caption] page={page_number} image={image_idx} {caption}"
                cid = _chunk_id(str(pdf_path), page_number, page_number, image_idx, f"{xref}:{content}")
                docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "doc_id": cid,
                            "source_file": str(pdf_path),
                            "source_filename": filename,
                            "page": page_number,
                            "start_page": page_number,
                            "end_page": page_number,
                            "chunk_type": "image_caption",
                            "image_xref": xref,
                            "image_ext": ext,
                            "snippet_preview": content[:220],
                        },
                    )
                )
    finally:
        pdf.close()

    return docs


def load_pdf_documents(
    pdf_path: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    bridge_window: int,
    enable_image_descriptions: bool,
    image_max_per_page: int,
    image_min_bytes: int,
    image_description_model: str,
    image_description_api_key: str,
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

    if enable_image_descriptions and image_description_api_key and image_max_per_page > 0:
        image_docs = _build_image_caption_documents(
            pdf_path=path,
            model=image_description_model,
            api_key=image_description_api_key,
            max_images_per_page=image_max_per_page,
            min_image_bytes=max(0, image_min_bytes),
        )
        docs.extend(image_docs)

    return docs
