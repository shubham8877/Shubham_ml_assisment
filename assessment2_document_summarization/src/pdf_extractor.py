"""
PDF Text & Table Extractor
Uses PyMuPDF (fitz) for text extraction and pdfplumber for table detection.
Falls back gracefully when tables can't be parsed.
"""

import io
import logging
from typing import Optional
import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts all text from a PDF file using PyMuPDF.
    Handles multi-column layouts by sorting text blocks by vertical position.
    """
    text_parts = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total_pages = len(doc)
        logger.info(f"Extracting text from {total_pages}-page PDF...")

        for page_num, page in enumerate(doc):
            # Extract text blocks with their bounding boxes
            blocks = page.get_text("blocks")

            # Sort by vertical position (top to bottom), then horizontal (left to right)
            blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))  # group rows by ~20pt

            page_text = f"\n--- Page {page_num + 1} ---\n"
            page_text += "\n".join(block[4] for block in blocks if block[4].strip())
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)
    logger.info(f"Extracted {len(full_text):,} characters of text.")
    return full_text


def extract_tables_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extracts tables from PDF using pdfplumber.
    Returns a list of dicts with page number and table data as markdown.
    """
    tables = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages):
            try:
                page_tables = page.extract_tables()
                for table in page_tables:
                    if table and len(table) > 1:
                        md_table = _table_to_markdown(table)
                        tables.append({
                            "page": page_num + 1,
                            "markdown": md_table,
                        })
            except Exception as e:
                logger.warning(f"Table extraction failed on page {page_num + 1}: {e}")

    logger.info(f"Extracted {len(tables)} tables from PDF.")
    return tables


def _table_to_markdown(table: list[list]) -> str:
    """Converts a pdfplumber table (list of lists) to markdown format."""
    if not table:
        return ""

    # Clean up None values
    cleaned = [[str(cell or "").strip() for cell in row] for row in table]

    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]

    return "\n".join([header, separator] + rows)


def extract_full_content(pdf_bytes: bytes) -> str:
    """
    Combines extracted text and tables into a single string
    that gets fed into the agent pipeline.
    """
    text = extract_text_from_pdf(pdf_bytes)
    tables = extract_tables_from_pdf(pdf_bytes)

    if tables:
        table_section = "\n\n## Extracted Tables\n\n"
        for t in tables:
            table_section += f"**Page {t['page']}:**\n{t['markdown']}\n\n"
        return text + table_section

    return text
