"""Unit and integration tests for PDF Ingestion Module.

Covers:
- Multipage PDF text extraction.
- Zero-based and one-based physical page numbering.
- Printed-page mapping when configured or absent.
- Missing PDF files.
- Invalid or unreadable PDFs.
- Empty or whitespace-only pages.
- Preservation of document IDs and source metadata.
- Batch ingestion of registered corpus.
- Preservation of verbatim text without rewriting.
"""

import json
from pathlib import Path
import pytest

from src.ingestion.models import DocumentMetadata, DocumentPage
from src.ingestion.loader import PDFLoader, PDFLoadError
from src.ingestion.pagination import PaginationResolver
from src.ingestion.registry import load_paper_registry, RegistryValidationError
from src.ingestion.batch import batch_ingest


class TestPDFLoaderUnit:
    """Unit tests using synthetic PDFs."""

    def test_multipage_pdf_extraction(self, synthetic_multipage_pdf: Path):
        """Test extraction of all pages from a multipage PDF."""
        loader = PDFLoader()
        result = loader.load_document(synthetic_multipage_pdf)

        assert result.success is True
        assert result.total_pages == 3
        assert len(result.pages) == 3
        assert result.empty_pages_count == 0

        # Verbatim text preservation
        assert "retrieval augmented generation" in result.pages[0].text
        assert "dense bi-encoder architectures" in result.pages[1].text
        assert "exact match accuracy" in result.pages[2].text

    def test_physical_page_numbering(self, synthetic_multipage_pdf: Path):
        """Verify correct zero-based physical index and one-based physical page number."""
        loader = PDFLoader()
        result = loader.load_document(synthetic_multipage_pdf)

        assert result.success is True
        for idx, page_record in enumerate(result.pages):
            assert page_record.page.physical_page_index == idx
            assert page_record.page.physical_page_number == idx + 1

    def test_printed_page_mapping_configured(self, synthetic_multipage_pdf: Path):
        """Verify printed-page mapping behavior when configured vs. unconfigured."""
        loader = PDFLoader()

        # Case 1: Verified offset mapping (e.g., physical_index_i = printed_page_i_plus_1)
        meta_with_offset = DocumentMetadata(
            id="DOC-1",
            title="Offset Doc",
            filename=synthetic_multipage_pdf.name,
            pagination_mapping="physical_index_i = printed_page_i_plus_1",
        )
        res1 = loader.load_document(synthetic_multipage_pdf, metadata=meta_with_offset)
        assert res1.pages[0].page.printed_page_number == 1
        assert res1.pages[1].page.printed_page_number == 2
        assert res1.pages[2].page.printed_page_number == 3

        # Case 2: Explicit offset override (e.g. +100)
        res2 = loader.load_document(
            synthetic_multipage_pdf,
            metadata=meta_with_offset,
            explicit_offset=100,
        )
        assert res2.pages[0].page.printed_page_number == 100
        assert res2.pages[1].page.printed_page_number == 101
        assert res2.pages[2].page.printed_page_number == 102

        # Case 3: Preprint format with no printed numbers -> printed_page_number is None
        meta_preprint = DocumentMetadata(
            id="DOC-2",
            title="Preprint Doc",
            filename=synthetic_multipage_pdf.name,
            pagination_mapping="physical_index_i = physical_page_i_plus_1 (preprint format)",
        )
        res3 = loader.load_document(synthetic_multipage_pdf, metadata=meta_preprint)
        assert res3.pages[0].page.printed_page_number is None
        assert res3.pages[1].page.printed_page_number is None
        assert res3.pages[2].page.printed_page_number is None

        # Case 4: No mapping provided -> printed_page_number is None
        meta_none = DocumentMetadata(
            id="DOC-3",
            title="No Mapping Doc",
            filename=synthetic_multipage_pdf.name,
        )
        res4 = loader.load_document(synthetic_multipage_pdf, metadata=meta_none)
        assert res4.pages[0].page.printed_page_number is None

    def test_missing_pdf_file(self, tmp_path: Path):
        """Verify graceful failure and error reporting when PDF file is missing."""
        missing_path = tmp_path / "non_existent_file.pdf"

        # Default safe mode (returns failed result)
        loader_safe = PDFLoader(raise_on_error=False)
        result = loader_safe.load_document(missing_path)

        assert result.success is False
        assert result.total_pages == 0
        assert "not found" in result.error_message.lower()

        # Strict mode (raises FileNotFoundError)
        loader_strict = PDFLoader(raise_on_error=True)
        with pytest.raises(FileNotFoundError):
            loader_strict.load_document(missing_path)

    def test_invalid_or_unreadable_pdf(self, corrupt_pdf: Path):
        """Verify graceful failure when PDF is corrupted or malformed."""
        # Default safe mode
        loader_safe = PDFLoader(raise_on_error=False)
        result = loader_safe.load_document(corrupt_pdf)

        assert result.success is False
        assert result.total_pages == 0
        assert "malformed or unreadable" in result.error_message.lower()

        # Strict mode
        loader_strict = PDFLoader(raise_on_error=True)
        with pytest.raises(PDFLoadError):
            loader_strict.load_document(corrupt_pdf)

    def test_empty_and_whitespace_only_pages(self, synthetic_empty_and_whitespace_pdf: Path):
        """Verify handling of empty and whitespace-only pages without ingestion crash."""
        loader = PDFLoader()
        result = loader.load_document(synthetic_empty_and_whitespace_pdf)

        assert result.success is True
        assert result.total_pages == 3
        assert result.empty_pages_count == 2

        # Page 0: Valid text
        assert result.pages[0].page.is_empty is False
        assert result.pages[0].page.char_count > 0
        assert result.pages[0].page.word_count > 0

        # Page 1: Truly empty page
        assert result.pages[1].page.is_empty is True
        assert result.pages[1].page.char_count == 0
        assert result.pages[1].page.word_count == 0
        assert result.pages[1].text == ""

        # Page 2: Whitespace-only page
        assert result.pages[2].page.is_empty is True
        assert result.pages[2].page.word_count == 0
        assert result.pages[2].text.strip() == ""

    def test_metadata_preservation(self, synthetic_multipage_pdf: Path):
        """Verify document identity and metadata are preserved across all page records."""
        loader = PDFLoader()
        metadata = DocumentMetadata(
            id="P99",
            title="Preserved Metadata Title",
            short_name="PMT",
            authors=["Alice Specialist", "Bob Researcher"],
            year=2024,
            venue="ICLR 2024",
            arxiv_id="2401.99999",
            doi="10.1234/test.2024",
            filename=synthetic_multipage_pdf.name,
        )

        result = loader.load_document(synthetic_multipage_pdf, metadata=metadata)
        assert result.success is True
        assert result.document_id == "P99"

        for page_record in result.pages:
            assert page_record.document.id == "P99"
            assert page_record.document.title == "Preserved Metadata Title"
            assert page_record.document.short_name == "PMT"
            assert page_record.document.authors == ["Alice Specialist", "Bob Researcher"]
            assert page_record.document.year == 2024
            assert page_record.document.venue == "ICLR 2024"
            assert page_record.document.arxiv_id == "2401.99999"
            assert page_record.document.doi == "10.1234/test.2024"


class TestPaginationResolver:
    """Unit tests for PaginationResolver."""

    def test_resolve_offset_string(self):
        assert PaginationResolver.resolve_printed_page(0, "physical_index_i = printed_page_i_plus_1") == 1
        assert PaginationResolver.resolve_printed_page(10, "physical_index_i = printed_page_i_plus_1") == 11
        assert PaginationResolver.resolve_printed_page(0, "printed_page_i_plus_5") == 5

    def test_resolve_preprint_format(self):
        assert PaginationResolver.resolve_printed_page(0, "physical_index_i = physical_page_i_plus_1 (preprint format)") is None
        assert PaginationResolver.resolve_printed_page(5, "no printed page numbers") is None

    def test_resolve_explicit_map(self):
        custom_map = {0: 10, 1: 20}
        assert PaginationResolver.resolve_printed_page(0, explicit_map=custom_map) == 10
        assert PaginationResolver.resolve_printed_page(1, explicit_map=custom_map) == 20
        assert PaginationResolver.resolve_printed_page(2, explicit_map=custom_map) is None


class TestBatchIngestionCorpus:
    """Integration tests on the actual evaluation corpus."""

    @pytest.fixture
    def corpus_paths(self):
        project_root = Path.cwd()
        registry_path = project_root / "evaluation" / "corpus" / "metadata" / "paper_registry.json"
        pdf_dir = project_root / "evaluation" / "corpus" / "papers"
        return registry_path, pdf_dir

    def test_load_paper_registry_validation(self, corpus_paths):
        registry_path, _ = corpus_paths
        papers = load_paper_registry(registry_path)

        assert len(papers) == 5
        paper_ids = [p.id for p in papers]
        assert paper_ids == ["P1", "P2", "P3", "P4", "P5"]

        for paper in papers:
            assert paper.id is not None
            assert paper.title is not None
            assert paper.filename is not None
            assert paper.total_pages is not None

    def test_successful_batch_ingestion_corpus(self, corpus_paths, tmp_path: Path):
        """Test batch ingestion of all 5 papers from the real evaluation corpus."""
        registry_path, pdf_dir = corpus_paths
        output_file = tmp_path / "test_corpus_pages.json"

        summary = batch_ingest(
            registry_path=registry_path,
            pdf_dir=pdf_dir,
            output_path=output_file,
        )

        assert summary.total_documents == 5
        assert summary.successful_documents == 5
        assert summary.failed_documents == 0
        assert summary.total_pages_extracted == 82  # 19 + 13 + 12 + 30 + 8
        assert summary.total_empty_pages == 0

        # Verify per-paper page counts and pagination
        res_by_id = {r.document_id: r for r in summary.results}
        assert res_by_id["P1"].total_pages == 19
        assert res_by_id["P2"].total_pages == 13
        assert res_by_id["P3"].total_pages == 12
        assert res_by_id["P4"].total_pages == 30
        assert res_by_id["P5"].total_pages == 8

        # P1 printed page numbers start at 1
        assert res_by_id["P1"].pages[0].page.printed_page_number == 1
        assert res_by_id["P1"].pages[18].page.printed_page_number == 19

        # P2, P3, P5 are preprint format -> printed_page_number is None
        assert all(p.page.printed_page_number is None for p in res_by_id["P2"].pages)
        assert all(p.page.printed_page_number is None for p in res_by_id["P3"].pages)
        assert all(p.page.printed_page_number is None for p in res_by_id["P5"].pages)

        # P4 printed page numbers start at 1
        assert res_by_id["P4"].pages[0].page.printed_page_number == 1
        assert res_by_id["P4"].pages[29].page.printed_page_number == 30

        # Output file verification
        assert output_file.is_file()
        with open(output_file, "r", encoding="utf-8") as f:
            saved_pages = json.load(f)
        assert len(saved_pages) == 82
        first_page = saved_pages[0]
        assert "document" in first_page
        assert "page" in first_page
        assert "text" in first_page
        assert first_page["document"]["id"] == "P1"

    def test_batch_ingestion_with_missing_document(self, tmp_path: Path):
        """Verify that batch ingestion reports missing documents without halting."""
        # Create a mock registry with 1 valid synthetic PDF and 1 missing PDF
        pdf_path = tmp_path / "valid.pdf"
        import pymupdf
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Valid doc")
        doc.save(str(pdf_path))
        doc.close()

        mock_registry = [
            {"id": "DOC1", "title": "Valid Doc", "filename": "valid.pdf"},
            {"id": "DOC2", "title": "Missing Doc", "filename": "missing.pdf"},
        ]
        reg_file = tmp_path / "mock_registry.json"
        with open(reg_file, "w", encoding="utf-8") as f:
            json.dump(mock_registry, f)

        summary = batch_ingest(
            registry_path=reg_file,
            pdf_dir=tmp_path,
        )

        assert summary.total_documents == 2
        assert summary.successful_documents == 1
        assert summary.failed_documents == 1
        assert summary.total_pages_extracted == 1

        failed_res = [r for r in summary.results if not r.success][0]
        assert failed_res.document_id == "DOC2"
        assert "not found" in failed_res.error_message.lower()

