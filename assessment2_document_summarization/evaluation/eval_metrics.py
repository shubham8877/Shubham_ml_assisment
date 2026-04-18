"""
Summary Quality Evaluation
Measures summary quality using ROUGE metrics and a simple G-Eval style
LLM-based grading rubric.

ROUGE scores measure lexical overlap between generated and reference summaries.
G-Eval uses an LLM as a judge to score on dimensions like coherence, relevance,
and factual consistency.
"""

import logging
from typing import Optional
from rouge_score import rouge_scorer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# ROUGE Evaluation
# ──────────────────────────────────────────────

def compute_rouge(
    generated_summary: str,
    reference_summary: str,
) -> dict:
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.
    Returns F1 scores for each metric.

    ROUGE-1: unigram overlap
    ROUGE-2: bigram overlap (captures phrase-level quality)
    ROUGE-L: longest common subsequence (captures fluency)
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    scores = scorer.score(reference_summary, generated_summary)

    return {
        "rouge1_f1": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f1": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f1": round(scores["rougeL"].fmeasure, 4),
    }


# ──────────────────────────────────────────────
# G-Eval (LLM-as-Judge)
# ──────────────────────────────────────────────

G_EVAL_PROMPT = """You are evaluating the quality of an AI-generated document summary.
Score the summary on each dimension from 1 (poor) to 5 (excellent).

Dimensions:
1. **Coherence** - Is the summary logically structured and easy to follow?
2. **Relevance** - Does it focus on the most important information from the source?
3. **Factual Consistency** - Does it avoid hallucinations or unsupported claims?
4. **Conciseness** - Is it appropriately brief without losing key information?

Source Document (first 2000 chars):
{source}

Generated Summary:
{summary}

Respond ONLY in this exact JSON format (no markdown, no explanation):
{{"coherence": X, "relevance": X, "factual_consistency": X, "conciseness": X, "overall": X}}
"""


def compute_g_eval(
    generated_summary: str,
    source_text: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Uses an LLM to evaluate summary quality across 4 dimensions.
    Returns scores from 1-5 for each dimension plus an overall score.
    """
    import json

    llm = ChatOpenAI(model=model, temperature=0)

    prompt = G_EVAL_PROMPT.format(
        source=source_text[:2000],
        summary=generated_summary[:1500],
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        scores = json.loads(response.content.strip())
        return {k: float(v) for k, v in scores.items()}
    except Exception as e:
        logger.error(f"G-Eval failed: {e}")
        return {
            "coherence": 0,
            "relevance": 0,
            "factual_consistency": 0,
            "conciseness": 0,
            "overall": 0,
        }


# ──────────────────────────────────────────────
# Combined Report
# ──────────────────────────────────────────────

def evaluate_summary(
    generated_summary: str,
    source_text: str,
    reference_summary: Optional[str] = None,
) -> dict:
    """
    Runs both ROUGE (if reference available) and G-Eval.
    Returns a unified evaluation report.
    """
    report = {}

    if reference_summary:
        rouge = compute_rouge(generated_summary, reference_summary)
        report["rouge"] = rouge
        logger.info(f"ROUGE scores: {rouge}")
    else:
        report["rouge"] = "No reference summary provided"
        logger.info("Skipping ROUGE (no reference summary)")

    g_eval = compute_g_eval(generated_summary, source_text)
    report["g_eval"] = g_eval
    logger.info(f"G-Eval scores: {g_eval}")

    return report
