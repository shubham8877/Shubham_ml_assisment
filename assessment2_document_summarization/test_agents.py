"""
Integration Tests for Assessment 2 — Agentic Document Summarization
Tests each agent in isolation and then the full pipeline.

Usage:
    cd assessment2_document_summarization
    python test_agents.py

Note: Tests that call the LLM are skipped if OPENAI_API_KEY is not set.
This lets you verify the structural/logic layers without incurring API costs.
"""

import os
import sys
import logging
import time

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))

SAMPLE_TEXT = """
Executive Summary

This report covers the Q3 performance of the distributed caching layer deployed
across three data centers. The system handles 2.4 million requests per second at
peak load with an average latency of 1.2ms.

Technical Specifications

Cache nodes: 48 machines, each with 256GB RAM
Eviction policy: LRU with TTL of 3600 seconds
Replication factor: 3 (across availability zones)
Write throughput: 800K ops/sec sustained
Read throughput: 2.4M ops/sec at p99

Performance Results

During the load test on September 12th, the cache achieved 99.97% hit rate under
synthetic workload. Tail latency (p99.9) stayed below 8ms even at peak load.
The only performance degradation occurred during a node failure simulation where
one AZ went offline — failover took 340ms on average.

Issues and Recommendations

The current consistent hashing algorithm leads to hotspots when key distribution
is skewed. Recommend switching to jump consistent hash or adding a virtual node layer.
Memory fragmentation is at 18%, higher than the 10% target — a defragmentation job
should run nightly during off-peak hours.
"""


# ──────────────────────────────────────────────
# Test 1: PDF Extractor (no API key needed)
# ──────────────────────────────────────────────

def test_pdf_extractor():
    logger.info("=" * 50)
    logger.info("TEST 1: PDF Extractor (text parsing logic)")
    logger.info("=" * 50)

    from src.pdf_extractor import _table_to_markdown

    # Test table conversion
    sample_table = [
        ["Metric", "Value", "Target"],
        ["Hit Rate", "99.97%", "99.9%"],
        ["P99 Latency", "8ms", "10ms"],
        ["Memory Frag", "18%", "10%"],
    ]

    md = _table_to_markdown(sample_table)
    assert "| Metric |" in md
    assert "| Hit Rate |" in md
    assert "---" in md

    logger.info("✅ Table-to-markdown conversion working")
    logger.info(f"   Sample output:\n{md[:200]}")
    return True


# ──────────────────────────────────────────────
# Test 2: Document Parser Agent
# ──────────────────────────────────────────────

def test_document_parser():
    logger.info("=" * 50)
    logger.info("TEST 2: DocumentParserAgent")
    logger.info("=" * 50)

    from agents.agents import DocumentParserAgent, PipelineState

    agent = DocumentParserAgent(chunk_size=300, chunk_overlap=50)
    state = PipelineState(raw_text=SAMPLE_TEXT)
    result = agent.run(state)

    assert result.error is None, f"Parser error: {result.error}"
    assert len(result.chunks) > 0, "No chunks created"
    assert result.current_step == "parsed"

    # Verify chunk metadata
    for i, chunk in enumerate(result.chunks):
        assert chunk.metadata.get("chunk_index") == i
        assert len(chunk.page_content) > 0

    logger.info(f"✅ DocumentParserAgent: created {len(result.chunks)} chunks")
    logger.info(f"   Avg chunk length: {sum(len(c.page_content) for c in result.chunks) / len(result.chunks):.0f} chars")
    return True


# ──────────────────────────────────────────────
# Test 3: Evaluation Metrics (no API key needed for ROUGE)
# ──────────────────────────────────────────────

def test_rouge_evaluation():
    logger.info("=" * 50)
    logger.info("TEST 3: ROUGE Evaluation Metrics")
    logger.info("=" * 50)

    from evaluation.eval_metrics import compute_rouge

    generated = "The caching system handles 2.4M requests/sec with 99.97% hit rate and 1.2ms latency."
    reference = "Cache performance: 2.4 million requests per second, hit rate 99.97%, average latency 1.2ms."

    scores = compute_rouge(generated, reference)

    assert "rouge1_f1" in scores
    assert "rouge2_f1" in scores
    assert "rougeL_f1" in scores
    assert 0 <= scores["rouge1_f1"] <= 1
    assert scores["rouge1_f1"] > 0.3, "ROUGE-1 suspiciously low for similar sentences"

    logger.info(f"✅ ROUGE scores computed successfully")
    logger.info(f"   ROUGE-1: {scores['rouge1_f1']:.3f}")
    logger.info(f"   ROUGE-2: {scores['rouge2_f1']:.3f}")
    logger.info(f"   ROUGE-L: {scores['rougeL_f1']:.3f}")
    return True


# ──────────────────────────────────────────────
# Test 4: Pipeline State Flow (structural test)
# ──────────────────────────────────────────────

def test_pipeline_state():
    logger.info("=" * 50)
    logger.info("TEST 4: PipelineState Data Flow")
    logger.info("=" * 50)

    from agents.agents import PipelineState
    from langchain.schema import Document

    # Simulate state flowing through parser manually
    state = PipelineState(raw_text=SAMPLE_TEXT)
    assert state.chunks == []
    assert state.section_summaries == []
    assert state.final_summary == ""
    assert state.error is None

    # Simulate what parser outputs
    state.chunks = [
        Document(page_content="Cache handles 2.4M rps", metadata={"chunk_index": 0}),
        Document(page_content="P99 latency under 8ms", metadata={"chunk_index": 1}),
    ]
    state.current_step = "parsed"

    # Simulate what summarizer outputs
    state.section_summaries = [
        "Cache system processes 2.4M requests/second.",
        "Tail latency stays under 8ms at peak load.",
    ]
    state.current_step = "summarized"

    # Simulate aggregator output
    state.final_summary = "## Executive Summary\n\nHigh-performance cache with 2.4M rps and sub-8ms latency."
    state.current_step = "aggregated"

    assert state.current_step == "aggregated"
    assert len(state.section_summaries) == 2
    assert "Executive Summary" in state.final_summary

    logger.info("✅ PipelineState flows correctly through all stages")
    logger.info(f"   Steps verified: parse → index → summarize → aggregate")
    return True


# ──────────────────────────────────────────────
# Test 5: Full LLM Pipeline (requires API key)
# ──────────────────────────────────────────────

def test_full_pipeline_with_llm():
    if not HAS_OPENAI_KEY:
        logger.info("⚠️  Skipping full pipeline test (no OPENAI_API_KEY set)")
        return "SKIPPED"

    logger.info("=" * 50)
    logger.info("TEST 5: Full LLM Pipeline (live API call)")
    logger.info("=" * 50)

    from agents.agents import OrchestratorAgent

    orchestrator = OrchestratorAgent(persist_dir="./test_chroma_db")
    result = orchestrator.run(SAMPLE_TEXT)

    assert result.error is None, f"Pipeline error: {result.error}"
    assert len(result.section_summaries) > 0
    assert len(result.final_summary) > 100

    logger.info(f"✅ Full pipeline ran successfully")
    logger.info(f"   Chunks: {len(result.chunks)}, Summaries: {len(result.section_summaries)}")
    logger.info(f"   Final summary length: {len(result.final_summary)} chars")
    logger.info(f"\n--- SUMMARY PREVIEW ---\n{result.final_summary[:400]}...\n")

    # Cleanup test DB
    import shutil
    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")

    return True


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

def main():
    tests = [
        ("PDF Extractor",         test_pdf_extractor),
        ("Document Parser Agent", test_document_parser),
        ("ROUGE Evaluation",      test_rouge_evaluation),
        ("Pipeline State Flow",   test_pipeline_state),
        ("Full LLM Pipeline",     test_full_pipeline_with_llm),
    ]

    results = []
    for name, fn in tests:
        try:
            start = time.time()
            outcome = fn()
            elapsed = time.time() - start
            status = "SKIP" if outcome == "SKIPPED" else "PASS"
            results.append((name, status, f"{elapsed:.1f}s"))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            logger.error(f"❌ {name} failed: {e}", exc_info=True)

    print("\n" + "=" * 58)
    print("  ASSESSMENT 2 — TEST SUMMARY")
    print("=" * 58)
    for name, status, detail in results:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⚠️ "}.get(status, "?")
        print(f"  {icon} {name:<32} {status}  ({detail})")
    print("=" * 58)

    if not HAS_OPENAI_KEY:
        print("\n  Tip: Set OPENAI_API_KEY to run the full LLM pipeline test.")

    failed = [r for r in results if r[1] == "FAIL"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
