# ML Engineer Assessment Submissions

Completed assessments: **Assessment 1** (Real-Time Anomaly Detection) and **Assessment 2** (Agentic Document Summarization).

## Repository Structure

```
ml-assessments/
├── assessment1_anomaly_detection/   # Fraud detection pipeline
│   ├── src/
│   │   ├── stream_simulator.py      # Synthetic transaction stream
│   │   ├── features.py              # Feature engineering
│   │   ├── train.py                 # Model training
│   │   └── app.py                   # FastAPI inference server
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   ├── reports/
│   │   └── evaluate.py              # Performance benchmarking
│   ├── requirements.txt
│   └── README.md
│
└── assessment2_document_summarization/   # Agentic PDF summarizer
    ├── agents/
    │   └── agents.py                # 5-agent LangGraph pipeline
    ├── src/
    │   └── pdf_extractor.py         # PDF text + table extraction
    ├── frontend/
    │   └── app.py                   # Streamlit UI
    ├── evaluation/
    │   └── eval_metrics.py          # ROUGE + G-Eval
    ├── requirements.txt
    └── README.md
```

## Quick Navigation

| Assessment | Key Tech | README |
|---|---|---|
| 1 — Anomaly Detection | Isolation Forest, Autoencoder, FastAPI, Docker | [README](./assessment1_anomaly_detection/README.md) |
| 2 — Document Summarization | LangGraph, ChromaDB, GPT-4o-mini, Streamlit | [README](./assessment2_document_summarization/README.md) |

---

Each assessment folder has its own `README.md` with detailed setup and run instructions.
