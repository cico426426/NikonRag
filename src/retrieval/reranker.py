from __future__ import annotations

import math
import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


class RerankItem(BaseModel):
    candidate_id: int = Field(ge=0)
    score: float


class RerankResponse(BaseModel):
    ranked: list[RerankItem]


@dataclass
class Candidate:
    document: Document
    vector_score: float


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\-\u4e00-\u9fff]+", text.lower()))


def _lexical_overlap(query: str, content: str) -> float:
    q = _tokens(query)
    c = _tokens(content)
    if not q or not c:
        return 0.0
    return len(q & c) / math.sqrt(len(q) * len(c))


class GeminiReranker:
    def __init__(self, *, model: str, api_key: str, timeout: float = 20.0) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            api_key=api_key,
            request_timeout=timeout,
        )
        self.structured = self.llm.with_structured_output(RerankResponse)

    def rerank(self, query: str, candidates: list[Candidate], *, keep_n: int) -> list[Candidate]:
        if not candidates:
            return []

        lines: list[str] = []
        for idx, candidate in enumerate(candidates):
            snippet = candidate.document.page_content[:480].replace("\n", " ")
            page = candidate.document.metadata.get("page")
            chunk_type = candidate.document.metadata.get("chunk_type", "page")
            lines.append(f"ID={idx} | page={page} | type={chunk_type} | text={snippet}")

        prompt = (
            "You are a retrieval reranker. "
            "Given a query and candidates, return ranked candidate IDs with relevance score (0-1). "
            "Focus on factual relevance and citation quality.\n\n"
            f"Query:\n{query}\n\nCandidates:\n" + "\n".join(lines)
        )

        try:
            result = self.structured.invoke(prompt)
            id_to_candidate = {idx: c for idx, c in enumerate(candidates)}
            ordered: list[Candidate] = []
            seen: set[int] = set()
            for item in sorted(result.ranked, key=lambda x: x.score, reverse=True):
                if item.candidate_id in seen:
                    continue
                candidate = id_to_candidate.get(item.candidate_id)
                if candidate is None:
                    continue
                seen.add(item.candidate_id)
                ordered.append(candidate)
                if len(ordered) >= keep_n:
                    break
            if ordered:
                return ordered
        except Exception:
            pass

        rescored = sorted(
            candidates,
            key=lambda c: (0.75 * _lexical_overlap(query, c.document.page_content)) + (0.25 * c.vector_score),
            reverse=True,
        )
        return rescored[:keep_n]
