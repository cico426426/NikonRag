import unittest

from langchain_core.documents import Document

from src.agent.rag_agent import RagAgent


def _doc(page: int, text: str = "chunk text") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_file": "/tmp/manual.pdf",
            "source_filename": "manual.pdf",
            "page": page,
            "doc_id": f"doc-{page}",
            "chunk_type": "page",
        },
    )


class CitationParsingTests(unittest.TestCase):
    def test_extract_grouped_citations(self) -> None:
        answer = "安裝鏡頭時請對齊標記 [C1, C2, C6]。"
        self.assertEqual(RagAgent._extract_cited_ids(answer), {1, 2, 6})

    def test_extract_citations_with_chinese_delimiters(self) -> None:
        answer = "請參考【C3、C4，C8】。"
        self.assertEqual(RagAgent._extract_cited_ids(answer), {3, 4, 8})

    def test_collect_citations_uses_fallback_when_no_marker(self) -> None:
        agent = RagAgent.__new__(RagAgent)
        docs = [_doc(1), _doc(2), _doc(3), _doc(4)]

        citations = agent._collect_citations(docs, "這句沒有任何引用標記。")

        self.assertEqual([c.page for c in citations], [1, 2, 3])

    def test_collect_citations_respects_grouped_markers(self) -> None:
        agent = RagAgent.__new__(RagAgent)
        docs = [_doc(10), _doc(20), _doc(30), _doc(40), _doc(50), _doc(60)]

        citations = agent._collect_citations(docs, "摘要 [C1, C2, C6]。")

        self.assertEqual([c.page for c in citations], [10, 20, 60])


if __name__ == "__main__":
    unittest.main()
