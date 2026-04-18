"""
Multi-Agent Document Summarization System
Built with LangGraph-style orchestration using explicit agent classes.

Agents:
  1. DocumentParserAgent  - Splits PDF into chunks, extracts tables
  2. RetrieverAgent       - Queries ChromaDB for relevant context
  3. SummarizerAgent      - Generates section summaries via LLM
  4. AggregatorAgent      - Combines section summaries into executive summary
  5. OrchestratorAgent    - Manages the workflow and routes between agents
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Shared State (passed between all agents)
# ──────────────────────────────────────────────

@dataclass
class PipelineState:
    """
    Central state object threaded through every node in the graph.
    LangGraph reads/writes this at each step.
    """
    raw_text: str = ""
    chunks: list[Document] = field(default_factory=list)
    section_summaries: list[str] = field(default_factory=list)
    final_summary: str = ""
    key_specs: dict = field(default_factory=dict)
    error: Optional[str] = None
    current_step: str = "start"


# ──────────────────────────────────────────────
# Agent 1: Document Parser
# ──────────────────────────────────────────────

class DocumentParserAgent:
    """
    Responsible for splitting raw PDF text into overlapping chunks.
    Overlap ensures that context spanning chunk boundaries isn't lost.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "],
        )

    def run(self, state: PipelineState) -> PipelineState:
        logger.info("[DocumentParserAgent] Splitting text into chunks...")

        if not state.raw_text:
            state.error = "No text to parse"
            return state

        chunks = self.splitter.create_documents(
            [state.raw_text],
            metadatas=[{"source": "uploaded_pdf", "chunk_index": i}
                       for i in range(1000)]  # rough upper bound
        )

        # Re-attach correct chunk indices after splitting
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        state.chunks = chunks
        state.current_step = "parsed"
        logger.info(f"[DocumentParserAgent] Created {len(chunks)} chunks.")
        return state


# ──────────────────────────────────────────────
# Agent 2: Retriever (ChromaDB)
# ──────────────────────────────────────────────

class RetrieverAgent:
    """
    Indexes document chunks into ChromaDB and retrieves relevant
    context for a given query. Using local Chroma (no server needed).
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None

    def index(self, chunks: list[Document]) -> None:
        """Embeds and stores all chunks in ChromaDB."""
        logger.info(f"[RetrieverAgent] Indexing {len(chunks)} chunks into ChromaDB...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
        )
        logger.info("[RetrieverAgent] Indexing complete.")

    def query(self, question: str, k: int = 4) -> list[Document]:
        """Returns top-k most relevant chunks for a given question."""
        if not self.vectorstore:
            raise RuntimeError("Index not built. Call .index() first.")
        return self.vectorstore.similarity_search(question, k=k)

    def run(self, state: PipelineState) -> PipelineState:
        self.index(state.chunks)
        state.current_step = "indexed"
        return state


# ──────────────────────────────────────────────
# Agent 3: Summarizer
# ──────────────────────────────────────────────

class SummarizerAgent:
    """
    Generates summaries for each document chunk using GPT.
    Processes chunks in batches to stay within rate limits.
    """

    SYSTEM_PROMPT = """You are a technical document analyst. 
    Your job is to extract key information from a document chunk and summarize it clearly.
    Focus on: technical specifications, metrics, key findings, and action items.
    Be concise but complete. Do not add information not present in the chunk."""

    def __init__(self, model: str = "llama-3.1-8b-instant", batch_size: int = 5):
        self.llm = ChatGroq(model=model, temperature=0)
        self.batch_size = batch_size

    def _summarize_chunk(self, chunk: Document) -> str:
        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Summarize this section:\n\n{chunk.page_content}"),
        ]
        response = self.llm.invoke(messages)
        return response.content

    def run(self, state: PipelineState) -> PipelineState:
        logger.info(f"[SummarizerAgent] Summarizing {len(state.chunks)} chunks...")
        summaries = []

        for i, chunk in enumerate(state.chunks):
            try:
                summary = self._summarize_chunk(chunk)
                summaries.append(summary)
                if (i + 1) % 5 == 0:
                    logger.info(f"  Summarized {i+1}/{len(state.chunks)} chunks")
            except Exception as e:
                logger.warning(f"  Chunk {i} failed: {e}. Using truncated text.")
                summaries.append(chunk.page_content[:300] + "...")

        state.section_summaries = summaries
        state.current_step = "summarized"
        logger.info("[SummarizerAgent] All chunks summarized.")
        return state


# ──────────────────────────────────────────────
# Agent 4: Aggregator
# ──────────────────────────────────────────────

class AggregatorAgent:
    """
    Takes all section summaries and produces a single structured
    executive summary with clearly labeled sections.
    """

    SYSTEM_PROMPT = """You are an expert technical writer creating an executive summary.
    Given a list of section summaries from a technical document, produce a structured report with:

    1. **Document Overview** (2-3 sentences on what this document is about)
    2. **Key Technical Specifications** (bullet points with concrete numbers/specs)
    3. **Main Findings & Insights** (the most important takeaways)
    4. **Recommended Actions** (if applicable, based on the document content)
    5. **Open Questions / Gaps** (what information is missing or unclear)

    Be precise, professional, and avoid vague statements."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    def run(self, state: PipelineState) -> PipelineState:
        logger.info("[AggregatorAgent] Generating executive summary...")

        combined = "\n\n---\n\n".join(
            f"Section {i+1}:\n{s}"
            for i, s in enumerate(state.section_summaries)
        )

        messages = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=f"Here are the section summaries:\n\n{combined}"),
        ]

        response = self.llm.invoke(messages)
        state.final_summary = response.content
        state.current_step = "aggregated"
        logger.info("[AggregatorAgent] Executive summary generated.")
        return state


# ──────────────────────────────────────────────
# Agent 5: Orchestrator (LangGraph Workflow)
# ──────────────────────────────────────────────

class OrchestratorAgent:
    """
    Builds and runs the LangGraph state machine that connects all agents.
    Graph: parse → index → summarize → aggregate → END
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.parser = DocumentParserAgent()
        self.retriever = RetrieverAgent(persist_dir=persist_dir)
        self.summarizer = SummarizerAgent()
        self.aggregator = AggregatorAgent()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(PipelineState)

        # Register each agent as a node
        workflow.add_node("parse", self.parser.run)
        workflow.add_node("index", self.retriever.run)
        workflow.add_node("summarize", self.summarizer.run)
        workflow.add_node("aggregate", self.aggregator.run)

        # Define the linear flow
        workflow.set_entry_point("parse")
        workflow.add_edge("parse", "index")
        workflow.add_edge("index", "summarize")
        workflow.add_edge("summarize", "aggregate")
        workflow.add_edge("aggregate", END)

        return workflow.compile()

    def run(self, raw_text: str) -> PipelineState:
        """Entry point: takes raw PDF text, returns final PipelineState."""
        initial_state = PipelineState(raw_text=raw_text)
        final_state = self.graph.invoke(initial_state)
        return final_state
