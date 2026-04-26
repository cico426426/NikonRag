# Nikon Manual RAG Assistant

以 Nikon 相機手冊為知識來源的 RAG 專案，支援 PDF 索引、語意檢索、引用追溯，以及 Streamlit 互動式問答介面。  
此專案重點是「可追溯的回答品質」與「跨頁內容檢索穩定性」。

## Portfolio Highlights

- 以 `pymupdf4llm` 解析 PDF，保留頁碼與來源檔 metadata。
- 使用 `RecursiveCharacterTextSplitter` 切塊，並補上 cross-page bridge chunks，降低段落跨頁斷裂造成的漏召回。
- Chroma 向量索引 + Gemini Embedding，支援 `source_file` 條件過濾。
- 兩階段檢索：向量召回 + 可選 Gemini rerank（失敗時 lexical/vector fallback）。
- 回答要求引用 `[C#]`，並輸出 `source_file / page / quote / chunk_type`。
- Streamlit 介面提供可點擊 citation 與對應頁面預覽，方便人工驗證答案來源。

## Demo Scenarios

- 問「如何格式化記憶卡？」可回傳步驟與頁碼。
- 問跨頁流程題時，能靠 cross-page chunk 提升檢索命中率。
- 可限定單一手冊查詢，避免多份文件內容混淆。

## Tech Stack

- LLM Framework: LangChain
- Model: Gemini (`langchain-google-genai`)
- Vector Store: Chroma (`langchain-chroma`)
- PDF Parsing: PyMuPDF / `pymupdf4llm`
- Frontend: Streamlit
- Language: Python 3.11+

## Project Structure

```text
src/
  agent/         # Agent orchestration and citation assembly
  ingestion/     # PDF loading and chunk generation
  retrieval/     # Vector index and reranker
  frontend/      # Streamlit UI
  common/        # Config and shared types
  app.py         # RagService facade (ingest / ask / list)
```

## How It Works

1. 讀取 PDF 並切成 page chunks。
2. 產生 cross-page bridge chunks（頁尾 + 下一頁頁首）。
3. 將 chunk 與 metadata 寫入 Chroma。
4. 問題進來後先做向量召回（top-k）。
5. 視設定執行 rerank，保留 top-n context。
6. LLM 生成繁中答案，並附上 citation。
7. 前端可直接點 citation 查看來源頁面。

## Quick Start

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install dotenv datasets langchain-chroma langchain-google-genai pymupdf4llm ragas "streamlit[pdf]" langsmith langchain-upstage
```

### 2) Environment

建立 `.env`：

```env
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
CHROMA_DIR=data/chroma
UPLOADS_DIR=data/uploads
DEFAULT_PDF_PATH=data/uploads/your_manual.pdf
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
BRIDGE_WINDOW=250
RECALL_K=30
RERANK_TOP_N=6
USE_RERANK=false
```

### 3) Download Manual (Official Source)

請勿將相機手冊 PDF 直接上傳到公開 GitHub。  
請從 Nikon 官方下載中心自行下載後放到 `data/uploads/`：

- Z 7II（繁體中文）: https://downloadcenter.nikonimglib.com/zh-tw/products/558/Z_7II.html
- Nikon Download Center 首頁: https://downloadcenter.nikonimglib.com/

例如下載後放入：

```text
data/uploads/Z7IIZ6IIRM_(Tc)10.pdf
```

### 4) Build Index

```bash
python3 -m src.build_index --pdf "data/uploads/your_manual.pdf"
```

### 5) Ask (CLI)

```bash
python3 -m src.query "如何格式化記憶卡？"
```

指定單一文件：

```bash
python3 -m src.query "如何設定自動對焦？" --source-file "/abs/path/to/manual.pdf"
```

### 6) Run UI

```bash
streamlit run src/frontend/streamlit_app.py
```

## Ragas Evaluation

準備評估資料（`jsonl` 或 `json`），每筆至少要有 `question`，建議加上 `reference`：

```json
{"question":"如何格式化記憶卡？","reference":"...正確答案..."}
```

執行評分：

```bash
python3 -m src.eval_ragas --eval-set data/eval/sample_eval.jsonl
```

可選參數：

- `--source-file`: 限定單一 PDF 來源
- `--out-json`: 摘要輸出（預設 `data/eval/ragas_summary.json`）
- `--out-csv`: 每題明細輸出（預設 `data/eval/ragas_details.csv`）

預設指標：

- `faithfulness`
- `answer_relevancy`
- `context_recall`（僅當每筆都有 `reference`）
- `context_precision_with_reference`（僅當每筆都有 `reference`）

## Engineering Notes

- 回答不只給文字，還要給可驗證引用，降低 hallucination 風險。
- bridge chunk 專門處理 PDF 常見的跨頁段落斷裂問題。
- reranker 有 fallback，提升可靠性與可用性。
- 目前 `RagAgent` 以 lock 保護共享狀態，避免多人同時請求時 citation 混線。

## Future Improvements

- 加入離線 evaluation（命中率、citation correctness、latency）。
- 將請求狀態改成完全 request-scoped，提升高併發吞吐。
- 增加 OCR/表格策略，提升掃描型 PDF 支援度。
- 導入 LangSmith trace 與評估指標儀表板。

## Author

`cico426426`  
NLP / LLM Application Engineer Portfolio Project
