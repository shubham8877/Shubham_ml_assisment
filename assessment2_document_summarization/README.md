# Assessment 2: Agentic Workflow for Document Summarization

A multi-agent pipeline that autonomously processes large PDF documents and generates structured executive summaries, built with **LangGraph** and **ChromaDB**.

## Multi-Agent Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            OrchestratorAgent             │
                    │         (LangGraph StateGraph)           │
                    └──────────────┬──────────────────────────┘
                                   │
              ┌────────────────────▼──────────────────────────┐
              │                                               │
     ┌────────▼─────────┐                         ┌──────────▼────────┐
     │ DocumentParser   │                         │  RetrieverAgent   │
     │ Agent            │                         │  (ChromaDB)       │
     │                  │                         │                   │
     │ - Splits PDF     │──────── chunks ────────▶│ - Embeds chunks   │
     │   into ~1000     │                         │ - Stores in vec   │
     │   char chunks    │                         │   database        │
     │ - Handles layout │                         │ - Similarity      │
     └──────────────────┘                         │   search          │
                                                  └──────────┬────────┘
                                                             │
                                                  ┌──────────▼────────┐
                                                  │  SummarizerAgent  │
                                                  │  (GPT-4o-mini)    │
                                                  │                   │
                                                  │ - Summarizes each │
                                                  │   chunk via LLM   │
                                                  │ - Domain-aware    │
                                                  │   prompt          │
                                                  └──────────┬────────┘
                                                             │
                                                  ┌──────────▼────────┐
                                                  │  AggregatorAgent  │
                                                  │                   │
                                                  │ - Combines all    │
                                                  │   summaries       │
                                                  │ - Structured      │
                                                  │   exec summary    │
                                                  └──────────┬────────┘
                                                             │
                                                  ┌──────────▼────────┐
                                                  │  EvaluatorAgent   │
                                                  │                   │
                                                  │ - ROUGE scores    │
                                                  │ - G-Eval (LLM     │
                                                  │   as judge)       │
                                                  └───────────────────┘
```

**Shared State** (PipelineState dataclass) is passed between every agent, ensuring no data is lost between steps.

## Project Structure

```
assessment2_document_summarization/
├── agents/
│   └── agents.py           # All 5 agent class definitions + orchestrator
├── src/
│   └── pdf_extractor.py    # PyMuPDF text + pdfplumber table extraction
├── frontend/
│   └── app.py              # Streamlit UI
├── evaluation/
│   └── eval_metrics.py     # ROUGE + G-Eval scoring
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="your-key-here"
# OR create a .env file with: OPENAI_API_KEY=your-key-here
```

### 3. Run the Streamlit app
```bash
streamlit run frontend/app.py
```
Then open `http://localhost:8501` in your browser.

### 4. Using the pipeline directly (without UI)

```python
from src.pdf_extractor import extract_full_content
from agents.agents import OrchestratorAgent
from evaluation.eval_metrics import evaluate_summary

# Read PDF
with open("my_document.pdf", "rb") as f:
    pdf_bytes = f.read()

# Extract text
raw_text = extract_full_content(pdf_bytes)

# Run pipeline
orchestrator = OrchestratorAgent()
result = orchestrator.run(raw_text)

print(result.final_summary)

# Evaluate
scores = evaluate_summary(result.final_summary, raw_text)
print(scores)
```

## Design Decisions

**Why LangGraph over CrewAI?**
LangGraph gives explicit control over the state machine — you can see exactly what data flows between each agent. CrewAI abstracts this away, which is convenient but makes debugging harder in production.

**Why ChromaDB?**
It's local (no external service needed), has a simple Python API, and persists to disk. For production, this would swap to Pinecone or Weaviate.

**Why two-stage summarization (chunk → aggregate)?**
Direct summarization of large documents hits LLM context limits. The map-reduce pattern (summarize each chunk, then summarize the summaries) is more reliable and cheaper.

**Why G-Eval in addition to ROUGE?**
ROUGE measures word overlap, not meaning. A summary can have high ROUGE but be incoherent. G-Eval's LLM judge catches semantic quality issues that ROUGE misses.
