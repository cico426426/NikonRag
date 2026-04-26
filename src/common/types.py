from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Citation:
    source_file: str
    source_filename: str
    page: int
    quote: str
    chunk_id: str
    chunk_type: str


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation]
