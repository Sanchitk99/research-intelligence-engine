"""Pytest fixtures for generating synthetic test PDFs."""

import pytest
import pymupdf
from pathlib import Path


@pytest.fixture
def synthetic_multipage_pdf(tmp_path: Path) -> Path:
    """Create a 3-page synthetic PDF with known text on each page."""
    pdf_path = tmp_path / "synthetic_multipage.pdf"
    doc = pymupdf.open()

    page_texts = [
        "First page content discussing retrieval augmented generation techniques.",
        "Second page detailing dense bi-encoder architectures and hard negative mining.",
        "Third page evaluating exact match accuracy across benchmarks.",
    ]

    for text in page_texts:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 72), text)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def synthetic_empty_and_whitespace_pdf(tmp_path: Path) -> Path:
    """Create a 3-page synthetic PDF containing valid text, an empty page, and whitespace."""
    pdf_path = tmp_path / "synthetic_empty_pages.pdf"
    doc = pymupdf.open()

    # Page 0: Valid text
    page0 = doc.new_page(width=595, height=842)
    page0.insert_text((50, 72), "Non-empty content on first page.")

    # Page 1: Completely empty page (no operations)
    doc.new_page(width=595, height=842)

    # Page 2: Whitespace only
    page2 = doc.new_page(width=595, height=842)
    page2.insert_text((50, 72), "   \n\t   \n  ")

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Create an unreadable, corrupted file disguised as a PDF."""
    corrupt_path = tmp_path / "corrupt_document.pdf"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_PDF_HEADER_JUST_GARBAGE_BYTES_1234567890\x00\xff\xfe")
    return corrupt_path

