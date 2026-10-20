"""Streamlit Web Application for the Research Intelligence Engine.

Demonstrates the complete end-to-end user workflow:
1. PDF ingestion and page-aware chunking.
2. Semantic vector and hybrid retrieval.
3. Evidence-grounded answer generation.
4. Verbatim citation verification with page provenance.
5. Strict abstention on insufficient or out-of-scope evidence.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

# Load .env variables (e.g. GEMINI_API_KEY, OPENAI_API_KEY)
load_dotenv(PROJECT_ROOT / ".env")

from src.ui.service import ResearchEngineService, IngestionReport
from src.generation.models import AnswerStatus, GeneratedAnswer


# ---------------------------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Research Intelligence Engine",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .status-answered {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    .status-partial {
        background-color: #FEF08A;
        color: #854D0E;
        border: 1px solid #FDE047;
    }
    .status-abstained {
        background-color: #E0E7FF;
        color: #3730A3;
        border: 1px solid #C7D2FE;
    }
    .status-conflict {
        background-color: #FCE7F3;
        color: #9D174D;
        border: 1px solid #FBCFE8;
    }
    .status-error {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #F8B4B4;
    }
    .citation-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .citation-header {
        font-weight: 600;
        font-size: 0.95rem;
        color: #0F172A;
        margin-bottom: 0.25rem;
    }
    .citation-meta {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .citation-excerpt {
        font-size: 0.9rem;
        font-style: italic;
        color: #334155;
        background-color: #FFFFFF;
        border: 1px dashed #CBD5E1;
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
    }
    .metric-pill {
        display: inline-block;
        background-color: #F1F5F9;
        color: #475569;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached Engine Service
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initializing Research Intelligence Engine...")
def get_service() -> ResearchEngineService:
    """Initialize and cache the engine service across interactions."""
    return ResearchEngineService(
        qdrant_path=PROJECT_ROOT / "data" / "qdrant",
        collection_name="research_corpus",
        chunks_file=PROJECT_ROOT / "data" / "chunked" / "corpus_chunks.json",
    )


service = get_service()
status_info = service.get_system_status()


# ---------------------------------------------------------------------------
# Sidebar: Controls & Corpus Overview
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🔬 Research Engine")
    st.markdown(
        "Grounded question answering with page-level citations over scientific literature."
    )
    st.divider()

    st.markdown("#### Provider & Model Settings")
    available_providers = ["auto", "gemini", "mock", "openai", "groq", "ollama"]
    
    # Auto-detect default provider choice
    gemini_ready = status_info["providers"]["gemini"]["configured"]
    default_provider_idx = 1 if gemini_ready else 0

    selected_provider = st.selectbox(
        "LLM Provider",
        options=available_providers,
        index=default_provider_idx,
        help="Select LLM provider. 'auto' selects Gemini if configured, else Mock.",
    )

    provider_status_text = "🟢 Gemini API Key active" if gemini_ready else "⚪ Gemini API Key not set"
    st.caption(provider_status_text)

    model_override = st.text_input(
        "Model Identifier (Optional)",
        value="",
        placeholder="e.g. gemini-flash-lite-latest",
        help="Leave blank to use provider default.",
    )
    chosen_model = model_override.strip() if model_override.strip() else None

    st.divider()
    st.markdown("#### Retrieval Settings")
    retrieval_mode = st.radio(
        "Retrieval Algorithm",
        options=["Hybrid (Dense + BM25 RRF)", "Dense Vector Only"],
        index=0,
        help="Hybrid combines dense embeddings and BM25 using Reciprocal Rank Fusion.",
    )
    resolved_retrieval_mode = "hybrid" if "Hybrid" in retrieval_mode else "dense"

    top_k_chunks = st.slider(
        "Top-k Context Chunks",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of retrieved passages provided to the synthesizer.",
    )

    st.divider()
    st.markdown("#### Indexed Corpus")
    st.metric(
        label="Total Indexed Chunks",
        value=status_info["total_points"],
        delta=f"{status_info['indexed_papers_count']} papers",
    )

    with st.expander("View Indexed Papers", expanded=False):
        for paper in status_info["papers"]:
            st.markdown(f"**{paper.document_id}**: {paper.title}")
            st.caption(f"Pages: {paper.total_pages} | Chunks: {paper.total_chunks}")


# ---------------------------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------------------------

tab_qa, tab_ingest, tab_health = st.tabs([
    "💬 Grounded Q&A",
    "📄 Paper Ingestion",
    "⚙️ System Status",
])


# ===========================================================================
# Tab 1: Grounded Q&A
# ===========================================================================

with tab_qa:
    st.markdown('<div class="main-header">Ask Research Questions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Every factual statement is synthesized from retrieved passages and backed by verified page-level citations.</div>',
        unsafe_allow_html=True,
    )

    # Sample query buttons
    st.markdown("**Sample Questions:**")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    default_q = ""
    if "current_question" not in st.session_state:
        st.session_state["current_question"] = "What generator model does RAG use?"

    with col_s1:
        if st.button("Q01: RAG Generator (P1)", use_container_width=True):
            st.session_state["current_question"] = "What generator model does RAG use?"
    with col_s2:
        if st.button("Q02: DPR Retrieval Accuracy (P2)", use_container_width=True):
            st.session_state["current_question"] = "What is the passage retrieval accuracy of DPR on Natural Questions?"
    with col_s3:
        if st.button("Q03: REALM Knowledge Retriever (P3)", use_container_width=True):
            st.session_state["current_question"] = "How does REALM pre-train its neural knowledge retriever?"
    with col_s4:
        if st.button("Out-of-Scope Query (Abstention)", use_container_width=True):
            st.session_state["current_question"] = "Who won the 2024 ICC Men's T20 World Cup?"

    # Query Input Form
    with st.form("qa_form", clear_on_submit=False):
        user_query = st.text_area(
            "Enter your research question:",
            value=st.session_state["current_question"],
            height=85,
            placeholder="e.g., What are the key evaluation metrics proposed by Ragas?",
        )
        submit_btn = st.form_submit_button("Generate Answer", type="primary", use_container_width=False)

    if submit_btn and user_query.strip():
        with st.spinner("Retrieving evidence and synthesizing grounded answer..."):
            ans: GeneratedAnswer = service.ask_question(
                question=user_query.strip(),
                provider_name=selected_provider,
                model_name=chosen_model,
                retrieval_mode=resolved_retrieval_mode,
                top_k=top_k_chunks,
            )

        st.divider()

        # Status badge rendering
        status_map = {
            AnswerStatus.ANSWERED: ("status-answered", "Answered Grounded"),
            AnswerStatus.PARTIALLY_ANSWERED: ("status-partial", "Partially Answered"),
            AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE: ("status-abstained", "Abstained: Insufficient Evidence"),
            AnswerStatus.ABSTAINED_OUT_OF_SCOPE: ("status-abstained", "Abstained: Out of Scope"),
            AnswerStatus.CONFLICTING_EVIDENCE: ("status-conflict", "Conflicting Evidence Detected"),
            AnswerStatus.ERROR: ("status-error", "Generation Error"),
        }
        css_cls, badge_label = status_map.get(ans.status, ("status-answered", str(ans.status.value)))

        st.markdown(
            f'<span class="status-badge {css_cls}">{badge_label}</span>',
            unsafe_allow_html=True,
        )

        # Answer text presentation
        st.markdown("### Answer")
        st.markdown(ans.answer)

        # Confidence Notes
        if ans.confidence_notes:
            st.info(f"**Evidence & Confidence Notes:** {ans.confidence_notes}")

        # Metadata Bar
        token_info = f"Prompt: {ans.prompt_tokens or 0} | Completion: {ans.completion_tokens or 0}" if ans.total_tokens else "N/A"
        st.markdown(
            f'<div style="margin: 0.8rem 0;">'
            f'<span class="metric-pill">Provider: {ans.provider_name}</span>'
            f'<span class="metric-pill">Model: {ans.model_name}</span>'
            f'<span class="metric-pill">Retrieval: {ans.retrieval_mode.upper()}</span>'
            f'<span class="metric-pill">Latency: {ans.latency_ms:.1f} ms</span>'
            f'<span class="metric-pill">Tokens: {token_info}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Verified Citations Section
        st.markdown(f"### Verified Citations ({len(ans.citations)})")
        if not ans.citations:
            if ans.status in (AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE, AnswerStatus.ABSTAINED_OUT_OF_SCOPE):
                st.caption("No citations attached because the system abstained from making unsupported factual claims.")
            else:
                st.caption("No verified citations attached to this response.")
        else:
            for idx, cite in enumerate(ans.citations, 1):
                printed_page_str = f" (Printed p. {cite.printed_page_number})" if cite.printed_page_number else ""
                st.markdown(
                    f"""
                    <div class="citation-card">
                        <div class="citation-header">[{idx}] {cite.document_title}</div>
                        <div class="citation-meta">Paper: <b>{cite.document_id}</b> &bull; Physical Page: <b>{cite.physical_page_number}</b>{printed_page_str} &bull; Chunk: <code>{cite.chunk_id}</code></div>
                        <div class="citation-excerpt">"{cite.excerpt}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Retrieved Context Passages (Collapsible)
        with st.expander(f"Inspect Retrieved Context Passages ({len(ans.retrieved_chunks)})", expanded=False):
            if not ans.retrieved_chunks:
                st.write("No context passages were retrieved.")
            else:
                for idx, chunk in enumerate(ans.retrieved_chunks, 1):
                    score_str = f"{chunk.score:.4f}" if chunk.score is not None else "N/A"
                    method_str = f" | Method: {chunk.retrieval_method}" if chunk.retrieval_method else ""
                    st.markdown(
                        f"**Passage [{idx}]** — `{chunk.chunk_id}` | Score: `{score_str}`{method_str} | Page {chunk.physical_page_number}"
                    )
                    st.caption(f"Document: {chunk.document_title} ({chunk.document_id})")
                    st.code(chunk.text, language="text")


# ===========================================================================
# Tab 2: Paper Ingestion & Indexing
# ===========================================================================

with tab_ingest:
    st.markdown('<div class="main-header">Ingest Research Papers</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Extract text page-by-page with physical page numbers, apply sentence-boundary chunking, and index vectors in Qdrant.</div>',
        unsafe_allow_html=True,
    )

    source_type = st.radio(
        "Ingestion Source",
        options=["Select Corpus Paper", "Upload New PDF"],
        horizontal=True,
    )

    papers_dir = PROJECT_ROOT / "evaluation" / "corpus" / "papers"
    available_files: List[Path] = sorted(list(papers_dir.glob("*.pdf"))) if papers_dir.is_dir() else []

    target_pdf_path: Optional[Path] = None
    default_doc_id = ""
    default_doc_title = ""

    if source_type == "Select Corpus Paper":
        if not available_files:
            st.warning(f"No PDF files found in {papers_dir}.")
        else:
            file_options = {f.name: f for f in available_files}
            selected_file_name = st.selectbox("Select Paper from Repository:", options=list(file_options.keys()))
            target_pdf_path = file_options[selected_file_name]
            default_doc_id = target_pdf_path.stem.split("_")[0] if "_" in target_pdf_path.stem else target_pdf_path.stem
            default_doc_title = target_pdf_path.stem.replace("_", " ")

    else:
        uploaded_file = st.file_uploader("Upload a Research Paper PDF:", type=["pdf"])
        if uploaded_file is not None:
            # Save uploaded bytes to a temporary file
            temp_dir = PROJECT_ROOT / "data" / "uploads"
            temp_dir.mkdir(parents=True, exist_ok=True)
            target_pdf_path = temp_dir / uploaded_file.name
            with open(target_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            default_doc_id = target_pdf_path.stem[:8].upper()
            default_doc_title = target_pdf_path.stem.replace("_", " ").title()

    col_id, col_title = st.columns([1, 3])
    with col_id:
        doc_id_input = st.text_input("Document Identifier:", value=default_doc_id, placeholder="e.g. P6")
    with col_title:
        doc_title_input = st.text_input("Paper Title:", value=default_doc_title, placeholder="e.g. Constitutional AI")

    dry_run_opt = st.checkbox("Dry Run (Validate extraction & chunking without writing to vector database)", value=False)

    if st.button("Ingest & Index Document", type="primary", disabled=target_pdf_path is None):
        with st.status("Executing ingestion pipeline...", expanded=True) as status:
            st.write("1. Reading PDF and resolving page numbers...")
            time_start = time.time()
            report: IngestionReport = service.ingest_paper(
                pdf_path=target_pdf_path,
                doc_id=doc_id_input.strip() or None,
                doc_title=doc_title_input.strip() or None,
                dry_run=dry_run_opt,
            )
            time_elapsed = time.time() - time_start

            if report.success:
                status.update(
                    label=f"Ingestion successful ({time_elapsed:.2f}s)!",
                    state="complete",
                    expanded=False,
                )
            else:
                status.update(
                    label="Ingestion failed!",
                    state="error",
                    expanded=True,
                )

        if report.success:
            st.success(
                f"Successfully ingested and indexed **{report.title}** (ID: `{report.document_id}`)."
            )

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Pages Extracted", report.pages_extracted)
            with col_m2:
                st.metric("Chunks Generated", report.chunks_created)
            with col_m3:
                st.metric("Total Collection Points", report.total_collection_points)

            if report.sample_chunks:
                st.markdown("#### Sample Generated Chunks")
                st.dataframe(
                    report.sample_chunks,
                    use_container_width=True,
                    column_config={
                        "chunk_id": "Chunk ID",
                        "page": "Page #",
                        "words": "Word Count",
                        "preview": "Text Preview",
                    },
                )
        else:
            st.error(f"Ingestion Error: {report.error_message}")


# ===========================================================================
# Tab 3: System Status & Diagnostics
# ===========================================================================

with tab_health:
    st.markdown('<div class="main-header">System Health & Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Inspect local vector store status, model configuration, and provider readiness.</div>',
        unsafe_allow_html=True,
    )

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        st.markdown("#### Vector Database (Qdrant)")
        st.json({
            "storage_path": str(service.qdrant_path),
            "collection_name": service.collection_name,
            "total_points_in_collection": status_info["total_points"],
            "indexed_papers_count": status_info["indexed_papers_count"],
        })

        st.markdown("#### Embedding Model")
        st.json({
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "normalize_embeddings": True,
            "device": "cpu/cuda auto",
        })

    with col_h2:
        st.markdown("#### LLM Providers Status")
        st.json(status_info["providers"])

        st.markdown("#### Retrieval Capabilities")
        st.json({
            "dense_vector": "Cosine similarity over 384d all-MiniLM-L6-v2 vectors",
            "bm25_keyword": "Okapi BM25 inverted index with compound-word tokenization",
            "hybrid_fusion": "Reciprocal Rank Fusion (RRF, k=60)",
            "reranker": "Cross-Encoder disabled (resource safety constraint)",
        })

