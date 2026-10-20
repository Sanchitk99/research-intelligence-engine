"""PDF Text Extraction Loader using PyMuPDF.

Extracts text page-by-page while preserving document provenance, physical
and printed page numbering, and verbatim page content.
"""

from pathlib import Path
from typing import Optional, Union, List
import pymupdf

from src.ingestion.models import (
    DocumentMetadata,
    PageMetadata,
    DocumentPage,
    DocumentIngestionResult,
)
from src.ingestion.pagination import PaginationResolver


class PDFLoadError(Exception):
    """Raised when a PDF file cannot be opened or parsed."""
    pass


class PDFLoader:
    """Safely extracts text and page metadata from PDF documents using PyMuPDF."""

    def __init__(self, raise_on_error: bool = False):
        """Initialize the loader.

        Args:
            raise_on_error: If True, raises exceptions on file/parsing errors.
                If False (default), catches errors and returns a failed
                DocumentIngestionResult.
        """
        self.raise_on_error = raise_on_error

    def load_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[DocumentMetadata] = None,
        explicit_offset: Optional[int] = None,
    ) -> DocumentIngestionResult:
        """Safely extract all pages and metadata from a PDF file.

        Args:
            file_path: Path to the target PDF file.
            metadata: Provenance metadata for the document. If omitted,
                a basic DocumentMetadata is constructed from the filename.
            explicit_offset: Optional manual override for printed page offset.

        Returns:
            DocumentIngestionResult containing all extracted DocumentPage
            records or error details.
        """
        path = Path(file_path).resolve()

        if metadata is None:
            metadata = DocumentMetadata(
                id=path.stem,
                title=path.stem.replace("_", " "),
                filename=path.name,
            )

        # 1. Validate file existence
        if not path.is_file():
            err_msg = f"PDF file not found: {path}"
            if self.raise_on_error:
                raise FileNotFoundError(err_msg)
            return DocumentIngestionResult(
                document_id=metadata.id,
                file_path=str(path),
                success=False,
                error_message=err_msg,
            )

        # 2. Open PDF and extract text safely
        doc: Optional[pymupdf.Document] = None
        try:
            doc = pymupdf.open(str(path))

            if doc.is_encrypted:
                err_msg = f"Encrypted PDF cannot be read without password: {path}"
                if self.raise_on_error:
                    raise PDFLoadError(err_msg)
                return DocumentIngestionResult(
                    document_id=metadata.id,
                    file_path=str(path),
                    success=False,
                    error_message=err_msg,
                )

            total_pages = len(doc)
            pages: List[DocumentPage] = []
            empty_count = 0

            mapping_spec = metadata.pagination_mapping

            for physical_idx in range(total_pages):
                page_obj = doc[physical_idx]
                raw_text = page_obj.get_text("text")

                is_empty = len(raw_text.strip()) == 0
                if is_empty:
                    empty_count += 1

                printed_page = PaginationResolver.resolve_printed_page(
                    physical_page_index=physical_idx,
                    mapping_spec=mapping_spec,
                    explicit_offset=explicit_offset,
                )

                page_meta = PageMetadata(
                    physical_page_index=physical_idx,
                    physical_page_number=physical_idx + 1,
                    printed_page_number=printed_page,
                    is_empty=is_empty,
                    char_count=len(raw_text),
                    word_count=len(raw_text.split()),
                )

                doc_page = DocumentPage(
                    document=metadata,
                    page=page_meta,
                    text=raw_text,  # Verbatim text preserved without rewriting
                )
                pages.append(doc_page)

            return DocumentIngestionResult(
                document_id=metadata.id,
                file_path=str(path),
                success=True,
                total_pages=total_pages,
                empty_pages_count=empty_count,
                pages=pages,
            )

        except (pymupdf.FileDataError, pymupdf.EmptyFileError) as e:
            err_msg = f"Malformed or unreadable PDF ({type(e).__name__}): {e}"
            if self.raise_on_error:
                raise PDFLoadError(err_msg) from e
            return DocumentIngestionResult(
                document_id=metadata.id,
                file_path=str(path),
                success=False,
                error_message=err_msg,
            )
        except Exception as e:
            err_msg = f"Unexpected error during PDF ingestion ({type(e).__name__}): {e}"
            if self.raise_on_error:
                raise PDFLoadError(err_msg) from e
            return DocumentIngestionResult(
                document_id=metadata.id,
                file_path=str(path),
                success=False,
                error_message=err_msg,
            )
        finally:
            if doc is not None:
                doc.close()

