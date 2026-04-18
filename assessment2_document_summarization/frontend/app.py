"""
Document Summarization - Streamlit Frontend
Upload a PDF → get a structured executive summary + quality evaluation scores.
"""

import os
import sys
import json
import logging
import streamlit as st

# Add parent dirs to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.pdf_extractor import extract_full_content
from agents.agents import OrchestratorAgent
from evaluation.eval_metrics import evaluate_summary

logging.basicConfig(level=logging.INFO)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="DocSummarizer AI",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Agentic Document Summarizer")
st.markdown(
    "Upload a technical PDF and the multi-agent pipeline will extract key information "
    "and generate a structured executive summary."
)

# ──────────────────────────────────────────────
# Sidebar: Configuration
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Required for LLM summarization. Not stored.",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    st.divider()

    reference_summary = st.text_area(
        "Reference Summary (optional)",
        placeholder="Paste a ground-truth summary here to compute ROUGE scores...",
        height=150,
    )

    st.divider()
    st.markdown("**Agent Pipeline:**")
    st.markdown("1. 📥 Document Parser")
    st.markdown("2. 🔍 Retriever (ChromaDB)")
    st.markdown("3. ✍️ Summarizer (GPT-4o-mini)")
    st.markdown("4. 🧩 Aggregator")
    st.markdown("5. 📊 Evaluator (ROUGE + G-Eval)")

# ──────────────────────────────────────────────
# Main: Upload & Process
# ──────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type="pdf",
    help="Works best with technical documents under 50 pages.",
)

if uploaded_file and st.button("🚀 Generate Summary", type="primary"):

    if not os.getenv("OPENAI_API_KEY"):
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    pdf_bytes = uploaded_file.read()

    # ── Step 1: Extract Text ────────────────────
    with st.status("📥 Extracting text from PDF...", expanded=True) as status:
        try:
            raw_text = extract_full_content(pdf_bytes)
            st.write(f"✅ Extracted {len(raw_text):,} characters from {uploaded_file.name}")
            status.update(label="Extraction complete", state="complete")
        except Exception as e:
            st.error(f"PDF extraction failed: {e}")
            st.stop()

    # ── Step 2: Run Agent Pipeline ───────────────
    with st.status("🤖 Running multi-agent pipeline...", expanded=True) as status:
        try:
            st.write("→ Parsing document into chunks...")
            st.write("→ Indexing into ChromaDB...")
            st.write("→ Summarizing sections via LLM...")
            st.write("→ Aggregating executive summary...")

            orchestrator = OrchestratorAgent(persist_dir="./chroma_db_temp")
            result = orchestrator.run(raw_text)

            if result.error:
                st.error(f"Pipeline error: {result.error}")
                st.stop()

            status.update(label="Pipeline complete", state="complete")
        except Exception as e:
            st.error(f"Agent pipeline failed: {e}")
            st.stop()

    # ── Step 3: Display Summary ──────────────────
    st.success("✅ Summary generated!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Executive Summary")
        st.markdown(result.final_summary)

        st.subheader("📑 Section Summaries")
        with st.expander(f"View all {len(result.section_summaries)} section summaries"):
            for i, summary in enumerate(result.section_summaries):
                st.markdown(f"**Section {i+1}:**")
                st.markdown(summary)
                st.divider()

    with col2:
        st.subheader("📊 Document Stats")
        st.metric("Total Characters", f"{len(raw_text):,}")
        st.metric("Chunks Created", len(result.chunks))
        st.metric("Sections Summarized", len(result.section_summaries))

    # ── Step 4: Evaluation ───────────────────────
    st.subheader("🔬 Summary Quality Evaluation")

    with st.spinner("Running ROUGE + G-Eval scoring..."):
        try:
            eval_report = evaluate_summary(
                generated_summary=result.final_summary,
                source_text=raw_text,
                reference_summary=reference_summary if reference_summary.strip() else None,
            )

            col3, col4 = st.columns(2)

            with col3:
                st.markdown("**G-Eval Scores (LLM-as-Judge, 1-5)**")
                g = eval_report.get("g_eval", {})
                for metric, score in g.items():
                    stars = "⭐" * int(score)
                    st.markdown(f"- **{metric.replace('_', ' ').title()}**: {score}/5 {stars}")

            with col4:
                if isinstance(eval_report.get("rouge"), dict):
                    st.markdown("**ROUGE Scores (vs Reference)**")
                    r = eval_report["rouge"]
                    st.metric("ROUGE-1 F1", f"{r['rouge1_f1']:.3f}")
                    st.metric("ROUGE-2 F1", f"{r['rouge2_f1']:.3f}")
                    st.metric("ROUGE-L F1", f"{r['rougeL_f1']:.3f}")
                else:
                    st.info("Paste a reference summary in the sidebar to see ROUGE scores.")

        except Exception as e:
            st.warning(f"Evaluation failed: {e}")

    # ── Download ─────────────────────────────────
    st.download_button(
        label="⬇️ Download Summary as JSON",
        data=json.dumps({
            "filename": uploaded_file.name,
            "executive_summary": result.final_summary,
            "section_count": len(result.section_summaries),
        }, indent=2),
        file_name="summary.json",
        mime="application/json",
    )
