"""Text Chunker with sentence and paragraph boundary preservation.

Splits extracted page text into retrieval-sized chunks while maintaining
structural boundaries, verbatim text integrity, exact character offsets,
and complete document provenance.
"""

import re
from typing import List, Tuple, Optional

from src.ingestion.models import DocumentPage
from src.chunking.models import ChunkingConfig, ChunkMetadata, DocumentChunk


class TextChunker:
    """Configurable, deterministic text chunker with boundary preservation."""

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize the chunker with target configuration.

        Args:
            config: ChunkingConfig specifying chunk_size_chars,
                chunk_overlap_chars, and min_chunk_size_chars.
        """
        self.config = config or ChunkingConfig()

    # Common scientific abbreviations and titles that should not trigger false sentence splits
    PROTECTED_ABBREVIATIONS = {
        "fig", "figs", "figure", "figures",
        "tab", "tabs", "table", "tables",
        "eq", "eqs", "equation", "equations",
        "e.g", "i.e", "et al", "al", "etc",
        "dr", "prof", "vs", "cf",
        "ref", "refs", "sec", "secs", "section",
        "vol", "no", "pp", "p",
    }

    def _is_abbreviation_or_initial(self, text: str, split_pos: int) -> bool:
        """Check if a split position immediately follows a protected abbreviation or single initial."""
        preceding = text[:split_pos]
        match = re.search(r"([A-Za-z0-9.]+)\Z", preceding)
        if not match:
            return False
        word = match.group(1).rstrip(".").lower()
        # Single uppercase initial (e.g. 'J.' in 'J. Smith')
        raw_word = match.group(1)
        if len(raw_word.rstrip(".")) == 1 and raw_word[0].isupper():
            return True
        return word in self.PROTECTED_ABBREVIATIONS

    def segment_text(self, text: str) -> List[Tuple[int, int, str]]:
        """Split text into atomic units (sentences, headings, or word blocks).

        Guarantees that every resulting unit has character length <= self.config.chunk_size_chars.

        Args:
            text: Source page text.

        Returns:
            List of (start_char_idx, end_char_idx, segment_text) tuples.
        """
        n = len(text)
        if not text.strip():
            return []

        # Split on paragraph breaks (\n\n+), sentence boundaries ([.?!]\s+), or line breaks (\n)
        split_pattern = re.compile(r"\n\n+|(?<=[.?!])\s+|\n")

        raw_spans: List[Tuple[int, int]] = []
        last = 0
        for m in split_pattern.finditer(text):
            start = m.start()
            end = m.end()

            # Skip false sentence splits caused by abbreviations or single-letter initials
            if self._is_abbreviation_or_initial(text, start):
                continue

            if start > last:
                raw_spans.append((last, start))
            last = end
        if last < n:
            raw_spans.append((last, n))

        atomic_units: List[Tuple[int, int, str]] = []
        max_size = self.config.chunk_size_chars

        for s_start, s_end in raw_spans:
            segment = text[s_start:s_end]
            if not segment.strip():
                continue

            if len(segment) <= max_size:
                atomic_units.append((s_start, s_end, segment))
            else:
                # Subdivide oversized segment (e.g. giant run-on sentence, URL, or table line)
                sub_curr = s_start
                while sub_curr < s_end:
                    sub_max = min(sub_curr + max_size, s_end)
                    if sub_max == s_end:
                        atomic_units.append((sub_curr, s_end, text[sub_curr:s_end]))
                        break

                    # Look for word boundary (space) within the oversized segment
                    space_idx = text.rfind(" ", sub_curr, sub_max)
                    if space_idx > sub_curr:
                        split_at = space_idx + 1
                    else:
                        # Hard character fallback if no space exists (e.g. long unbroken token/URL)
                        split_at = sub_max

                    atomic_units.append((sub_curr, split_at, text[sub_curr:split_at]))
                    sub_curr = split_at

        return atomic_units

    def chunk_text(self, text: str) -> List[Tuple[int, int, str]]:
        """Chunk a text string into overlapping spans respecting boundaries.

        Args:
            text: Verbatim extracted text from a single page.

        Returns:
            List of (start_char_idx, end_char_idx, chunk_text) tuples, where
            text[start_char_idx:end_char_idx] == chunk_text.
        """
        if not text or not text.strip():
            return []

        units = self.segment_text(text)
        if not units:
            return []

        # If all units combined fit in one chunk, return single trimmed chunk
        total_span = units[-1][1] - units[0][0]
        if total_span <= self.config.chunk_size_chars:
            start = units[0][0]
            end = units[-1][1]
            return [(start, end, text[start:end])]

        chunks: List[Tuple[int, int, str]] = []
        i = 0
        n_units = len(units)

        while i < n_units:
            curr_units = [units[i]]
            j = i + 1

            while j < n_units:
                candidate_end = units[j][1]
                chunk_start = curr_units[0][0]
                span_len = candidate_end - chunk_start

                if span_len <= self.config.chunk_size_chars:
                    curr_units.append(units[j])
                    j += 1
                else:
                    break

            chunk_start = curr_units[0][0]
            chunk_end = curr_units[-1][1]
            chunk_str = text[chunk_start:chunk_end]
            chunks.append((chunk_start, chunk_end, chunk_str))

            if j >= n_units:
                break

            # Calculate exact character overlap for next chunk
            if self.config.chunk_overlap_chars <= 0:
                i = j
            else:
                chunk_A_end = curr_units[-1][1]
                best_k = None

                # Search backwards from the second-to-last unit in curr_units down to unit 1
                # to find the unit that yields the largest character overlap <= chunk_overlap_chars
                for k in range(len(curr_units) - 1, 0, -1):
                    ov = chunk_A_end - curr_units[k][0]
                    if ov <= self.config.chunk_overlap_chars:
                        best_k = k
                    else:
                        break

                if best_k is not None:
                    next_i = i + best_k
                elif len(curr_units) > 1:
                    # Documented sentence-boundary exception: when even the single last unit
                    # exceeds chunk_overlap_chars, include that single last unit as overlap
                    # to preserve sentence integrity, rather than cutting mid-sentence.
                    next_i = i + (len(curr_units) - 1)
                else:
                    next_i = j

                # Strict monotonic progress checks to prevent duplicate chunks or loops
                if next_i <= i:
                    next_i = i + 1
                if next_i >= j:
                    next_i = j

                i = next_i

        # Trailing fragment handling: if the final chunk is smaller than min_chunk_size_chars
        if len(chunks) > 1 and len(chunks[-1][2]) < self.config.min_chunk_size_chars:
            last_st, last_en, _ = chunks[-1]
            prev_st, _, _ = chunks[-2]

            # Case A: If merging with previous chunk fits in chunk_size_chars
            if last_en - prev_st <= self.config.chunk_size_chars:
                chunks[-2] = (prev_st, last_en, text[prev_st:last_en])
                chunks.pop()
            else:
                # Case B: Expand last chunk backwards to an earlier unit boundary
                # so its length is >= min_chunk_size_chars (and <= chunk_size_chars)
                for u in reversed(units):
                    if u[0] < last_st:
                        cand_len = last_en - u[0]
                        if cand_len <= self.config.chunk_size_chars:
                            chunks[-1] = (u[0], last_en, text[u[0]:last_en])
                            if cand_len >= self.config.min_chunk_size_chars:
                                break

        return chunks

    def chunk_page(
        self,
        page: DocumentPage,
        chunk_index_in_doc_start: int = 0,
    ) -> List[DocumentChunk]:
        """Convert a DocumentPage into a list of DocumentChunks with full provenance.

        Args:
            page: Source DocumentPage from Task 3.1.
            chunk_index_in_doc_start: Cumulative chunk index within the document.

        Returns:
            List of DocumentChunk objects with verified metadata and verbatim text.
        """
        # Safely handle empty or whitespace-only pages
        if page.page.is_empty or not page.text or not page.text.strip():
            return []

        raw_chunks = self.chunk_text(page.text)
        document_chunks: List[DocumentChunk] = []

        doc_id = page.document.id
        doc_title = page.document.title
        phys_idx = page.page.physical_page_index
        phys_num = page.page.physical_page_number
        printed_num = page.page.printed_page_number

        for page_chunk_idx, (start_idx, end_idx, chunk_text) in enumerate(raw_chunks):
            # Deterministic, stable chunk ID: e.g. P1_p01_c01
            chunk_id = f"{doc_id}_p{phys_num:02d}_c{page_chunk_idx + 1:02d}"

            meta = ChunkMetadata(
                chunk_id=chunk_id,
                document_id=doc_id,
                document_title=doc_title,
                physical_page_index=phys_idx,
                physical_page_number=phys_num,
                printed_page_number=printed_num,
                chunk_index_in_page=page_chunk_idx,
                chunk_index_in_doc=chunk_index_in_doc_start + page_chunk_idx,
                start_char_idx=start_idx,
                end_char_idx=end_idx,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
            )

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                metadata=meta,
                text=chunk_text,
            )
            document_chunks.append(chunk)

        return document_chunks

    def chunk_document_pages(
        self,
        pages: List[DocumentPage],
    ) -> List[DocumentChunk]:
        """Chunk an ordered sequence of DocumentPages for one or more documents.

        Args:
            pages: List of DocumentPage records.

        Returns:
            List of all generated DocumentChunk records.
        """
        all_chunks: List[DocumentChunk] = []
        doc_chunk_counters: dict[str, int] = {}

        for page in pages:
            doc_id = page.document.id
            current_doc_idx = doc_chunk_counters.get(doc_id, 0)

            page_chunks = self.chunk_page(
                page=page,
                chunk_index_in_doc_start=current_doc_idx,
            )

            all_chunks.extend(page_chunks)
            doc_chunk_counters[doc_id] = current_doc_idx + len(page_chunks)

        return all_chunks

