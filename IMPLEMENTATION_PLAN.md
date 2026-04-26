# Multimodal RAG Agent Implementation Plan

## Goal

Build a multimodal RAG agent for Nikon manuals using LangChain + Gemini, with:

- robust PDF ingestion (including cross-page context)
- retrieval with rerank
- answer-level citations (page + original snippet)
- clickable citation UX to open PDF at the referenced page

## Scope and Constraints

- Framework: LangChain
- Models: Gemini chat model + Gemini embedding model
- PDF parsing: PyMuPDF / PyMuPDF4LLM
- Frontend: Streamlit
- Citation requirement: each answer must include source file, page number, and original quote snippet
- UX requirement: citation click should navigate to PDF content (target page)

## Architecture

1. Ingestion Layer
- Parse PDF page-by-page and preserve page metadata.
- Generate standard page chunks.
- Generate cross-page bridge chunks (tail of page N + head of page N+1).
- Persist chunk metadata:
  - `doc_id`
  - `source_file`
  - `start_page`
  - `end_page`
  - `chunk_type` (`page` or `cross_page`)
  - `snippet_preview`

2. Index Layer
- Store chunks in Chroma.
- Use Gemini embeddings for vector indexing.
- Keep metadata filterable by `source_file` for single-manual mode.

3. Retrieval + Rerank Layer
- Stage 1: Vector recall from Chroma (high recall, e.g. top 20-40).
- Stage 2: Rerank candidate chunks to top-N context (e.g. top 5-8) before generation.
- Rerank strategy:
  - Primary: model-based rerank chain (LLM scores relevance for query-candidate pairs).
  - Fallback: similarity-score-based rerank if model rerank is unavailable or timeout.
- Deduplication:
  - Merge near-duplicate chunks by `(source_file, start_page, normalized_text_hash)`.
- Cross-page handling in rerank:
  - Prefer cross-page chunks when query likely spans transitions (steps/table split across pages).
  - Keep at least 1-2 single-page chunks to preserve exact-page citation precision.

4. Answer Generation Layer
- Use retrieved + reranked chunks as context.
- Enforce structured output:
  - `answer`
  - `citations[]`:
    - `source_file`
    - `page`
    - `quote` (verbatim snippet from chunk)
- Validation:
  - If answer sentence has no supporting citation, mark as uncertain or omit claim.

5. Frontend Layer (Streamlit)
- Left: chat UI (`st.chat_input`, chat history).
- Right: PDF viewer (`st.pdf`) for selected manual.
- Citation panel:
  - clickable source item (`file + page`)
  - on click, jump viewer to page (via PDF viewer page navigation approach)
  - show quote snippet under citation entry
- Query modes:
  - all indexed manuals
  - selected manual only (metadata filter)

## Retrieval and Rerank Details

1. Candidate Recall
- Retriever: Chroma similarity/MMR.
- Initial `k`: 20-40 (tunable by document size).

2. Rerank
- Input: user query + recalled candidates.
- Output: ranked list with relevance score.
- Keep top `N` for answer context (5-8 default).
- Include diversity guard:
  - avoid selecting all chunks from same page unless required.

3. Citation Reliability
- For each selected chunk, store precise page metadata.
- For cross-page chunks:
  - if quote range is attributable to one page, cite that page.
  - if ambiguous, create separate citations for both pages.

4. Fallback and Resilience
- If rerank service fails:
  - fall back to top vector results.
- If retrieved evidence is weak:
  - return "insufficient evidence" style response with best-effort citations.

## Milestones

1. Milestone A: Data and Chunking
- Implement page-level extraction and cross-page bridge chunks.
- Define metadata schema and test with sample Nikon PDF.

2. Milestone B: Index and Retrieval
- Build Chroma index pipeline with Gemini embeddings.
- Add retriever modes (all docs / single doc).

3. Milestone C: Rerank + Generation
- Add two-stage retrieval (recall + rerank).
- Add structured citation output and evidence checks.

4. Milestone D: Frontend
- Build Streamlit chat + PDF viewer split layout.
- Add clickable citations and page navigation UX.

5. Milestone E: Evaluation
- Build mini eval set with cross-page questions.
- Measure citation correctness and answer grounding quality.

## Initial Configuration (Suggested Defaults)

- Embedding model: `models/gemini-embedding-001`
- Generation model: `gemini-2.5-flash`
- Chunk size: 700-1000 chars
- Chunk overlap: 120-180 chars
- Cross-page bridge window: 200-300 chars (tail/head)
- Recall top-k: 30
- Rerank keep-N: 6

## Risks and Mitigations

1. Cross-page citation ambiguity
- Mitigation: dual citation for bridge chunks when page attribution is unclear.

2. Rerank latency
- Mitigation: bounded candidate count + timeout + fallback path.

3. OCR / PDF extraction noise
- Mitigation: use PyMuPDF layout-aware extraction and normalize whitespace.

4. Hallucination under weak retrieval
- Mitigation: citation-required output policy and insufficient-evidence response.

## Official Documentation References

- LangChain Retrieval:
  - https://docs.langchain.com/oss/python/langchain/retrieval
- LangChain `create_retrieval_chain`:
  - https://reference.langchain.com/python/langchain-classic/chains/retrieval/create_retrieval_chain
- LangChain Chroma integration:
  - https://docs.langchain.com/oss/python/integrations/vectorstores/chroma
- LangChain Google Generative AI integration:
  - https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- Gemini models:
  - https://ai.google.dev/gemini-api/docs/models
- Gemini embeddings:
  - https://ai.google.dev/gemini-api/docs/embeddings
- Gemini document processing:
  - https://ai.google.dev/gemini-api/docs/document-processing
- PyMuPDF4LLM API:
  - https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html
- PyMuPDF extraction notes:
  - https://pymupdf.readthedocs.io/en/latest/app1.html
- LangChain PyMuPDF4LLM loader:
  - https://docs.langchain.com/oss/python/integrations/document_loaders/pymupdf4llm
- Streamlit chat input:
  - https://docs.streamlit.io/develop/api-reference/chat/st.chat_input
- Streamlit PDF viewer:
  - https://docs.streamlit.io/develop/api-reference/media/st.pdf
- PDF.js viewer options:
  - https://github.com/mozilla/pdf.js/wiki/Viewer-options

## Method References

- LangChain multimodal / semi-structured RAG:
  - https://www.langchain.com/blog/semi-structured-multi-modal-rag
- Pinecone chunking strategies:
  - https://www.pinecone.io/learn/chunking-strategies/
- Cohere chunking strategies:
  - https://docs.cohere.com/page/chunking-strategies
