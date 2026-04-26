from __future__ import annotations

import html
from pathlib import Path
import sys

import pymupdf
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import RagService
from src.common.types import Citation

st.set_page_config(page_title="Nikon RAG Agent", layout="wide")
st.markdown(
    """
    <style>
    :root {
      --bg-a: #f9fbff;
      --bg-b: #f0f5fc;
      --ink: #0b1f33;
      --muted: #3f556f;
      --line: #b8c8da;
      --card: #ffffff;
      --chip: #e9f1ff;
      --chip-hover: #d7e7ff;
    }
    .stApp {
      background:
        radial-gradient(circle at 5% 0%, #f0f7ff 0%, transparent 30%),
        radial-gradient(circle at 95% 10%, #eef3ff 0%, transparent 28%),
        linear-gradient(180deg, var(--bg-a) 0%, var(--bg-b) 100%);
      color: #0f2740;
    }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
      color: #0b1f33;
    }
    h1, h2, h3, h4 {
      color: #0b1f33 !important;
    }
    .block-container {
      padding-top: 2.6rem;
      padding-left: 1.2rem;
      padding-right: 1.2rem;
      max-width: 1650px;
    }
    .panel-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--card);
      padding: 1rem 1rem;
      margin-bottom: 0.9rem;
      box-shadow: 0 10px 20px rgba(12, 35, 62, 0.08);
    }
    .quick-jump-title {
      color: #1e3a5f;
      font-weight: 700;
      margin-bottom: 0.55rem;
    }
    .cite-snippet {
      font-size: 0.88rem;
      line-height: 1.45;
      color: #1f354e;
      margin: 0.25rem 0 0.95rem 0;
      white-space: normal;
      word-break: break-word;
    }
    .quote-box {
      border: 1px solid #c2d0df;
      border-radius: 10px;
      padding: 12px;
      background: #f4f8fd;
      font-size: 0.95rem;
      line-height: 1.5;
      white-space: pre-wrap;
      word-break: break-word;
      margin-bottom: 0.75rem;
    }
    div.stButton > button {
      border-radius: 999px;
      border: 1px solid #9ab0ca;
      background: var(--chip);
      color: var(--ink);
      font-weight: 600;
      min-height: 2.2rem;
      padding-left: 0.85rem;
      padding-right: 0.85rem;
    }
    div.stButton > button:hover {
      border-color: #6f8fb5;
      background: var(--chip-hover);
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #f7fbff 0%, #eef4fb 100%);
      border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * {
      color: #0b1f33 !important;
    }
    [data-testid="stTextInput"] input,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
      color: #0b1f33 !important;
      background: #ffffff !important;
      border: 1px solid #9fb4cc !important;
    }
    [data-testid="stSelectbox"] svg {
      fill: #4a6481 !important;
    }
    [data-testid="stChatInput"] {
      border-top: 1px solid #d3deea;
      background: rgba(247, 250, 255, 0.96);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_service() -> RagService:
    return RagService()


@st.cache_data(show_spinner=False)
def render_page_image(pdf_path: str, page_number: int, zoom: float = 1.6) -> bytes:
    doc = pymupdf.open(pdf_path)
    page = doc.load_page(max(page_number - 1, 0))
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return pix.tobytes("png")


def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / uploaded_file.name
    target.write_bytes(uploaded_file.getbuffer())
    return target


def citation_to_dict(c: Citation) -> dict[str, str | int]:
    return {
        "source_file": c.source_file,
        "source_filename": c.source_filename,
        "page": c.page,
        "quote": c.quote,
        "chunk_id": c.chunk_id,
        "chunk_type": c.chunk_type,
    }


def short_snippet(text: str, limit: int = 140) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rstrip() + " ..."


service = get_service()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None
if "scroll_to_pdf" not in st.session_state:
    st.session_state.scroll_to_pdf = False
if "latest_citations" not in st.session_state:
    st.session_state.latest_citations = []

st.markdown("### Nikon RAG")

with st.sidebar:
    st.subheader("檢索設定")
    indexed_files = service.list_documents()
    options = ["(全部文件)"] + indexed_files
    selected_source = st.selectbox("檢索範圍", options)

    with st.expander("索引與檔案管理", expanded=False):
        upload = st.file_uploader("上傳 PDF", type=["pdf"])
        if upload is not None:
            saved = save_uploaded_file(upload, service.settings.uploads_dir)
            st.success(f"已儲存：{saved.name}")

        local_pdfs = sorted(str(p.resolve()) for p in service.settings.uploads_dir.glob("*.pdf"))
        ingest_target = st.selectbox("建立索引文件", local_pdfs if local_pdfs else ["(尚無可用 PDF)"])
        if st.button("重建/新增索引", disabled=not local_pdfs):
            with st.spinner("索引中..."):
                count = service.ingest(ingest_target)
            st.success(f"完成，新增 {count} 個 chunks")

col_chat, col_pdf = st.columns([1.35, 0.65], gap="large")

with col_chat:
    st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
    with st.expander("快速來源跳轉（最新回應）", expanded=False):
        if st.session_state.latest_citations:
            quick_cols = st.columns(2, gap="medium")
            for cite_idx, c in enumerate(st.session_state.latest_citations, start=1):
                label = f"[{cite_idx}] {c['source_filename']} p.{c['page']}"
                col = quick_cols[(cite_idx - 1) % 2]
                with col:
                    if st.button(label, key=f"quick_cite_{cite_idx}_{c['chunk_id']}_{c['page']}"):
                        st.session_state.selected_citation = c
                        st.session_state.scroll_to_pdf = True
        else:
            st.caption("送出問題後，這裡會顯示最新回應的來源。")
    st.markdown("</div>", unsafe_allow_html=True)

    chat_history = st.container(height=560, border=True, autoscroll=True)
    with chat_history:
        for msg_idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                citations = msg.get("citations", [])
                if msg["role"] == "assistant" and citations:
                    with st.expander("來源", expanded=False):
                        for cite_idx, c in enumerate(citations, start=1):
                            label = f"[{cite_idx}] {c['source_filename']} p.{c['page']} ({c['chunk_type']})"
                            if st.button(label, key=f"msg_{msg_idx}_cite_{cite_idx}_{c['chunk_id']}_{c['page']}"):
                                st.session_state.selected_citation = c
                                st.session_state.scroll_to_pdf = True
                            st.markdown(
                                f"<div class='cite-snippet'>{html.escape(short_snippet(str(c['quote'])))}</div>",
                                unsafe_allow_html=True,
                            )

with col_pdf:
    st.markdown("<div id='pdf-viewer-anchor'></div>", unsafe_allow_html=True)
    st.subheader("PDF 檢視")
    selected_citation = st.session_state.selected_citation

    if selected_citation is not None:
        st.markdown(
            f"選取來源：`{selected_citation['source_filename']}` 第 `{selected_citation['page']}` 頁"
        )
        pdf_path = Path(str(selected_citation["source_file"]))
        tab_quote, tab_page = st.tabs(["引用原文", "對應頁預覽"])
        with tab_quote:
            with st.expander("顯示引用原文", expanded=True):
                st.markdown(
                    f"<div class='quote-box'>{html.escape(str(selected_citation['quote']))}</div>",
                    unsafe_allow_html=True,
                )
        with tab_page:
            if pdf_path.exists():
                try:
                    image_bytes = render_page_image(str(pdf_path), int(selected_citation["page"]), 1.7)
                    st.image(
                        image_bytes,
                        caption=f"頁面預覽：p.{selected_citation['page']}",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.warning(f"頁面預覽失敗：{exc}")
            else:
                st.warning("找不到來源 PDF 檔案。")
    elif selected_source != "(全部文件)":
        pdf_path = Path(selected_source)
        if pdf_path.exists():
            st.markdown(f"目前文件：`{pdf_path.name}`")
            st.pdf(pdf_path.read_bytes(), height=380)
        else:
            st.info("請先建立索引或上傳 PDF。")
    else:
        st.info("點擊左側答案的來源，即可查看對應 PDF 內容。")

question = st.chat_input("請輸入問題，例如：如何格式化記憶卡？")
if question and question.strip():
    clean_question = question.strip()
    st.session_state.messages.append({"role": "user", "content": clean_question})

    source_filter = None if selected_source == "(全部文件)" else selected_source
    with st.spinner("檢索與生成中..."):
        result = service.ask(clean_question, source_file=source_filter)

    citations = [citation_to_dict(c) for c in result.citations]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer,
            "citations": citations,
        }
    )
    if citations:
        st.session_state.selected_citation = citations[0]
        st.session_state.scroll_to_pdf = True
    st.session_state.latest_citations = citations
    st.rerun()

if st.session_state.scroll_to_pdf:
    components.html(
        """
        <script>
        const anchor = window.parent.document.getElementById("pdf-viewer-anchor");
        if (anchor) {
          anchor.scrollIntoView({ behavior: "smooth", block: "start" });
        }
        </script>
        """,
        height=0,
    )
    st.session_state.scroll_to_pdf = False
