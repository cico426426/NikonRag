from __future__ import annotations

import re
import threading
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.common.types import Citation, RagAnswer
from src.retrieval.reranker import Candidate, GeminiReranker
from src.retrieval.vector_store import VectorIndex


class RagAgent:
    def __init__(
        self,
        *,
        vector_index: VectorIndex,
        model: str,
        api_key: str,
        recall_k: int,
        rerank_top_n: int,
        use_rerank: bool,
    ) -> None:
        self.vector_index = vector_index
        self.answer_llm = ChatGoogleGenerativeAI(model=model, temperature=0.2, api_key=api_key)
        self.reranker = GeminiReranker(model=model, api_key=api_key) if use_rerank else None
        self.recall_k = recall_k
        self.rerank_top_n = rerank_top_n
        self.use_rerank = use_rerank
        self._ask_lock = threading.Lock()
        self._active_source_file: str | None = None
        self._latest_retrieved_docs: list[Document] = []

        self.retrieve_context_tool = StructuredTool.from_function(
            func=self.retrieve_context,
            name="retrieve_context",
            description="Retrieve Nikon manual context snippets for the query. Always use this before final answer.",
        )
        self.agent = create_agent(
            model=self.answer_llm,
            tools=[self.retrieve_context_tool],
            system_prompt=(
                "你是 Nikon 手冊助理。回答前必須先呼叫 retrieve_context 取得證據。"
                "你只能根據工具回傳內容作答，不可腦補。"
                "每個重點句盡量在句尾加 [C編號]。"
                "若證據不足，明確說明。"
            ),
        )

    def _build_context(self, docs: list[Document]) -> str:
        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            filename = doc.metadata.get("source_filename")
            chunk_type = doc.metadata.get("chunk_type", "page")
            content = doc.page_content.strip().replace("\n", " ")
            parts.append(
                f"[C{idx}] file={filename} page={page} type={chunk_type}\n"
                f"{content}"
            )
        return "\n\n".join(parts)

    def _collect_citations(self, docs: list[Document], answer: str) -> list[Citation]:
        cited_ids = self._extract_cited_ids(answer)
        if not cited_ids:
            cited_ids = set(range(1, min(len(docs), 3) + 1))

        citations: list[Citation] = []
        for cid in sorted(cited_ids):
            if cid < 1 or cid > len(docs):
                continue
            doc = docs[cid - 1]
            metadata = doc.metadata
            quote = doc.page_content[:260].replace("\n", " ")
            start_page = int(metadata.get("start_page", metadata.get("page", 1)))
            end_page = int(metadata.get("end_page", start_page))

            if start_page == end_page:
                citations.append(
                    Citation(
                        citation_id=cid,
                        source_file=str(metadata.get("source_file", "")),
                        source_filename=str(metadata.get("source_filename", Path(str(metadata.get("source_file", ""))).name)),
                        page=start_page,
                        quote=quote,
                        chunk_id=str(metadata.get("doc_id", "")),
                        chunk_type=str(metadata.get("chunk_type", "page")),
                    )
                )
            else:
                for page in [start_page, end_page]:
                    citations.append(
                        Citation(
                            citation_id=cid,
                            source_file=str(metadata.get("source_file", "")),
                            source_filename=str(metadata.get("source_filename", Path(str(metadata.get("source_file", ""))).name)),
                            page=page,
                            quote=quote,
                            chunk_id=str(metadata.get("doc_id", "")),
                            chunk_type="cross_page",
                        )
                    )
        return citations

    @staticmethod
    def _extract_cited_ids(answer: str) -> set[int]:
        # Support grouped citations like [C1, C2, C6], [C1、C2], and full-width brackets.
        ids = {int(n) for n in re.findall(r"[Cc]\s*(\d+)", answer)}
        return ids

    def _retrieve_documents(self, query: str, *, source_file: str | None) -> list[Document]:
        recalled = self.vector_index.retrieve_with_scores(query, k=self.recall_k, source_file=source_file)
        candidates = [Candidate(document=doc, vector_score=score) for doc, score in recalled]
        if self.use_rerank and self.reranker is not None:
            picked = self.reranker.rerank(query, candidates, keep_n=self.rerank_top_n)
        else:
            picked = candidates[: self.rerank_top_n]
        return [item.document for item in picked]

    def _extract_answer_text(self, result: dict) -> str:
        messages = result.get("messages")
        if not messages:
            return ""
        for message in reversed(messages):
            if getattr(message, "type", "") != "ai":
                continue
            text = getattr(message, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
            content = getattr(message, "content", None)
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""

    def retrieve_context(self, query: str) -> str:
        self._latest_retrieved_docs = self._retrieve_documents(query, source_file=self._active_source_file)
        if not self._latest_retrieved_docs:
            return "NO_EVIDENCE"
        return self._build_context(self._latest_retrieved_docs)

    def retrieve_documents(self, query: str, *, source_file: str | None = None) -> list[Document]:
        return self._retrieve_documents(query, source_file=source_file)

    def ask(self, question: str, *, source_file: str | None = None) -> RagAnswer:
        with self._ask_lock:
            self._active_source_file = source_file
            self._latest_retrieved_docs = []
            try:
                result = self.agent.invoke({"messages": [{"role": "user", "content": question}]})
                answer_text = self._extract_answer_text(result)
                retrieved_docs = self._latest_retrieved_docs
            finally:
                self._active_source_file = None
                self._latest_retrieved_docs = []

            if not retrieved_docs:
                return RagAnswer(answer="找不到足夠的文件證據，請改寫問題或指定手冊。", citations=[])
            if not answer_text:
                answer_text = "已檢索到文件片段，但無法產生有效回答。請改寫問題後重試。"

            citations = self._collect_citations(retrieved_docs, answer_text)
            return RagAnswer(answer=answer_text, citations=citations)
