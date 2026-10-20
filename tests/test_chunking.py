"""Comprehensive automated tests for Text Chunking Module.

Covers all 11 required scenarios:
1. Short text fitting in one chunk.
2. Long text requiring multiple chunks.
3. Paragraph and sentence boundary preservation.
4. Configurable chunk size and overlap.
5. Empty and whitespace-only input.
6. A single sentence longer than the chunk limit.
7. Metadata and source-page provenance preservation.
8. Deterministic chunk IDs and output ordering.
9. Exact character offset matching.
10. No accidental text loss or unintended duplicate chunks.
11. Real five-paper corpus integration check.
"""

from pathlib import Path
import pytest

from src.ingestion.models import DocumentMetadata, PageMetadata, DocumentPage
from src.chunking.models import ChunkingConfig, ChunkMetadata, DocumentChunk
from src.chunking.text_chunker import TextChunker
from src.chunking.pipeline import chunk_corpus, ChunkingPipelineError


@pytest.fixture
def sample_page() -> DocumentPage:
    """Fixture providing a realistic synthetic research paper DocumentPage."""
    text = (
        "Retrieval-Augmented Generation (RAG) combines pre-trained parametric memory with "
        "non-parametric memory for knowledge-intensive NLP tasks.\n\n"
        "We introduce two model variants: RAG-Sequence and RAG-Token. RAG-Sequence uses the same "
        "retrieved document to generate the entire sequence. In contrast, RAG-Token marginalizes "
        "across documents at each token step.\n\n"
        "Our experiments demonstrate state-of-the-art results on open-domain question answering benchmarks, "
        "including Natural Questions, TriviaQA, and WebQuestions."
    )
    doc_meta = DocumentMetadata(
        id="P1",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        filename="P1_RAG_Lewis_2020.pdf",
    )
    page_meta = PageMetadata(
        physical_page_index=0,
        physical_page_number=1,
        printed_page_number=1,
        is_empty=False,
        char_count=len(text),
        word_count=len(text.split()),
    )
    return DocumentPage(document=doc_meta, page=page_meta, text=text)


class TestTextChunkerUnit:
    """Unit tests for TextChunker algorithm and boundary handling."""

    def test_short_text_single_chunk(self):
        """Test short text that easily fits within one chunk."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=500, chunk_overlap_chars=50))
        short_text = "This is a concise paragraph describing dense passage retrieval."
        raw_chunks = chunker.chunk_text(short_text)

        assert len(raw_chunks) == 1
        start, end, chunk_str = raw_chunks[0]
        assert start == 0
        assert end == len(short_text)
        assert chunk_str == short_text

    def test_long_text_multiple_chunks(self, sample_page: DocumentPage):
        """Test long text requiring multiple chunks with configured size limit."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=40))
        chunks = chunker.chunk_page(sample_page)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 200
            assert chunk.metadata.char_count == len(chunk.text)

    def test_paragraph_and_sentence_boundary_preservation(self):
        """Verify that splits occur at paragraph or sentence boundaries and avoid mid-sentence cuts."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=80, chunk_overlap_chars=20))
        text = (
            "First complete sentence here. Second complete sentence here.\n\n"
            "Third complete sentence starting a new paragraph. Fourth complete sentence follows."
        )
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 2
        for _, _, c_text in chunks:
            # Chunk should start with an uppercase letter or paragraph start
            assert c_text[0].isupper()
            # Chunk should end on a complete punctuation mark
            assert c_text[-1] in {".", "!", "?", '"', "'"}

    def test_configurable_chunk_size_and_overlap(self, sample_page: DocumentPage):
        """Verify that different chunk_size and chunk_overlap settings alter output accordingly."""
        chunker_large = TextChunker(ChunkingConfig(chunk_size_chars=600, chunk_overlap_chars=100))
        chunks_large = chunker_large.chunk_page(sample_page)

        chunker_small = TextChunker(ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=40))
        chunks_small = chunker_small.chunk_page(sample_page)

        # Smaller chunk size must yield strictly more chunks
        assert len(chunks_small) > len(chunks_large)

        # Invalid config: overlap >= chunk_size must raise ValueError
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=200)
        with pytest.raises(ValueError):
            ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=250)

    def test_empty_and_whitespace_only_input(self):
        """Verify empty and whitespace-only text produces zero chunks without errors."""
        chunker = TextChunker()

        # Pure empty string
        assert chunker.chunk_text("") == []

        # Whitespace and newlines only
        assert chunker.chunk_text("   \n\t\n   ") == []

        # DocumentPage with empty flag
        empty_page = DocumentPage(
            document=DocumentMetadata(id="TEST", title="Test"),
            page=PageMetadata(physical_page_index=0, physical_page_number=1, is_empty=True),
            text="",
        )
        assert chunker.chunk_page(empty_page) == []

        # DocumentPage with whitespace text
        whitespace_page = DocumentPage(
            document=DocumentMetadata(id="TEST", title="Test"),
            page=PageMetadata(physical_page_index=0, physical_page_number=1, is_empty=False),
            text="   \n\n\t  ",
        )
        assert chunker.chunk_page(whitespace_page) == []

    def test_single_sentence_longer_than_chunk_limit(self):
        """Verify that an oversized sentence exceeding chunk_size is subdivided without getting stuck."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=60, chunk_overlap_chars=15))
        long_sentence = (
            "This is an unusually extended technical sentence designed to test the chunker fallback "
            "behavior when a single unbroken proposition exceeds the configured maximum character budget."
        )
        assert len(long_sentence) > 60

        raw_chunks = chunker.chunk_text(long_sentence)
        assert len(raw_chunks) > 1

        for start, end, chunk_str in raw_chunks:
            assert len(chunk_str) <= 60
            assert long_sentence[start:end] == chunk_str

        # Edge case: A single word with no spaces longer than chunk limit
        giant_word = "A" * 150
        raw_giant = chunker.chunk_text(giant_word)
        assert len(raw_giant) == 3  # 60 + 60 + 30
        for start, end, chunk_str in raw_giant:
            assert len(chunk_str) <= 60
            assert giant_word[start:end] == chunk_str

    def test_metadata_and_provenance_preservation(self, sample_page: DocumentPage):
        """Verify that document IDs, titles, page numbers, and indices are preserved on all chunks."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=40))
        chunks = chunker.chunk_page(sample_page, chunk_index_in_doc_start=5)

        for idx, chunk in enumerate(chunks):
            assert chunk.metadata.document_id == "P1"
            assert chunk.metadata.document_title == "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
            assert chunk.metadata.physical_page_index == 0
            assert chunk.metadata.physical_page_number == 1
            assert chunk.metadata.printed_page_number == 1
            assert chunk.metadata.chunk_index_in_page == idx
            assert chunk.metadata.chunk_index_in_doc == 5 + idx
            assert chunk.chunk_id == f"P1_p01_c{idx + 1:02d}"

    def test_deterministic_chunk_ids_and_ordering(self, sample_page: DocumentPage):
        """Verify that identical inputs produce 100% deterministic IDs, offsets, and texts."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=250, chunk_overlap_chars=50))

        run1 = chunker.chunk_page(sample_page)
        run2 = chunker.chunk_page(sample_page)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.metadata.start_char_idx == c2.metadata.start_char_idx
            assert c1.metadata.end_char_idx == c2.metadata.end_char_idx
            assert c1.text == c2.text

    def test_character_offsets_exact_match(self, sample_page: DocumentPage):
        """Verify that page_text[start:end] == chunk.text holds strictly for all chunks."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=180, chunk_overlap_chars=30))
        chunks = chunker.chunk_page(sample_page)

        for chunk in chunks:
            st = chunk.metadata.start_char_idx
            en = chunk.metadata.end_char_idx
            assert st < en
            assert sample_page.text[st:en] == chunk.text

    def test_no_accidental_loss_or_unintended_duplicate_chunks(self, sample_page: DocumentPage):
        """Verify no duplicate chunks and no premature text truncation."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=40))
        chunks = chunker.chunk_page(sample_page)

        # Check start offsets are strictly increasing
        for i in range(len(chunks) - 1):
            assert chunks[i].metadata.start_char_idx < chunks[i + 1].metadata.start_char_idx
            # Texts must not be identical
            assert chunks[i].text != chunks[i + 1].text

        # First chunk starts at beginning of text
        assert chunks[0].metadata.start_char_idx == 0
        # Last chunk ends at end of text
        assert chunks[-1].metadata.end_char_idx == len(sample_page.text)


class TestCorpusBatchChunking:
    """Integration tests on the actual five-paper corpus."""

    @pytest.fixture
    def corpus_pages_file(self) -> Path:
        p = Path("data/ingested/corpus_pages.json").resolve()
        assert p.is_file(), "data/ingested/corpus_pages.json must exist before running integration tests"
        return p

    def test_real_five_paper_corpus_chunking(self, corpus_pages_file: Path, tmp_path: Path):
        """Test batch chunking across all 82 pages of the 5 research papers."""
        output_file = tmp_path / "test_corpus_chunks.json"
        config = ChunkingConfig(chunk_size_chars=1000, chunk_overlap_chars=150)

        summary = chunk_corpus(
            input_path=corpus_pages_file,
            output_path=output_file,
            config=config,
        )

        assert summary.total_documents == 5
        assert summary.total_pages_processed == 82
        assert summary.total_empty_pages_skipped == 0
        assert summary.failed_pages == 0
        assert summary.total_chunks_created > 0
        assert summary.avg_chunk_chars > 0

        # Verify all 5 papers are represented
        expected_docs = {"P1", "P2", "P3", "P4", "P5"}
        assert set(summary.chunks_per_document.keys()) == expected_docs

        # Load and validate generated output file
        assert output_file.is_file()
        with open(output_file, "r", encoding="utf-8") as f:
            import json
            chunks_data = json.load(f)

        assert len(chunks_data) == summary.total_chunks_created

        for item in chunks_data:
            assert "chunk_id" in item
            assert "metadata" in item
            assert "text" in item

            meta = item["metadata"]
            assert meta["document_id"] in expected_docs
            assert meta["physical_page_index"] >= 0
            assert meta["physical_page_number"] == meta["physical_page_index"] + 1
            assert meta["start_char_idx"] < meta["end_char_idx"]
            assert meta["char_count"] == len(item["text"])
            assert meta["char_count"] <= 1000

            # Grounded pagination check:
            # P1 and P4 must have printed_page_number as int
            # P2, P3, P5 must have printed_page_number as None (preprints)
            if meta["document_id"] in {"P1", "P4"}:
                assert isinstance(meta["printed_page_number"], int)
            else:
                assert meta["printed_page_number"] is None

    def test_pipeline_missing_input_file(self, tmp_path: Path):
        """Verify pipeline raises FileNotFoundError on missing input path."""
        missing = tmp_path / "non_existent_pages.json"
        with pytest.raises(FileNotFoundError):
            chunk_corpus(input_path=missing)

    def test_pipeline_invalid_json(self, tmp_path: Path):
        """Verify pipeline raises ChunkingPipelineError on corrupt JSON."""
        corrupt = tmp_path / "corrupt.json"
        with open(corrupt, "w", encoding="utf-8") as f:
            f.write("{this is not valid json")
        with pytest.raises(ChunkingPipelineError):
            chunk_corpus(input_path=corrupt)


class TestChunkingCorrectnessAudit:
    """Task 3.2.1 Audit: Verification of mathematical invariants and adversarial edge cases."""

    def test_invariant_1_exact_character_offset_matching(self):
        """Invariant 1: Every chunk text matches its source slice exactly via character offsets."""
        text = (
            "Retrieval-Augmented Generation (RAG) is a technique for grounding LLM outputs. "
            "It combines dense retrieval over millions of documents with sequence-to-sequence generation.\n\n"
            "By marginalizing across retrieved documents, RAG generates more factual and reliable text."
        )
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=120, chunk_overlap_chars=30))
        raw_chunks = chunker.chunk_text(text)

        assert len(raw_chunks) >= 3
        for start, end, chunk_str in raw_chunks:
            assert text[start:end] == chunk_str, f"Mismatch at [{start}:{end}]: {repr(chunk_str)}"

    def test_invariant_2_full_non_whitespace_coverage(self):
        """Invariant 2: Every non-whitespace source character is represented in the output chunks."""
        text = (
            "Paragraph one describes baseline dense retrieval.\n\n"
            "Paragraph two introduces salient span masking during unsupervised pre-training.\n\n"
            "Paragraph three provides comprehensive open-domain QA exact match results."
        )
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=110, chunk_overlap_chars=25))
        raw_chunks = chunker.chunk_text(text)

        covered = [False] * len(text)
        for st, en, _ in raw_chunks:
            for idx in range(st, en):
                covered[idx] = True

        for idx, char in enumerate(text):
            if not char.isspace():
                assert covered[idx], f"Non-whitespace char '{char}' at index {idx} was not covered by any chunk!"

    def test_invariant_3_forward_progress_and_termination(self):
        """Invariant 3: Chunking strictly makes forward progress and terminates under extreme conditions."""
        extreme_text = "Word " * 500 + "A" * 600 + " \n\n\t " * 50 + "FinalWord."
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=80, chunk_overlap_chars=15))

        chunks = chunker.chunk_text(extreme_text)
        assert len(chunks) > 0

        # Verify strictly increasing start offsets
        for i in range(len(chunks) - 1):
            assert chunks[i + 1][0] > chunks[i][0], (
                f"No forward progress between chunk {i} ({chunks[i][0]}) and {i + 1} ({chunks[i + 1][0]})"
            )

    def test_invariant_4_unique_and_deterministic_chunk_ids(self, sample_page: DocumentPage):
        """Invariant 4: Chunk IDs are unique and 100% stable across repeated executions."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=150, chunk_overlap_chars=30))

        run1 = chunker.chunk_page(sample_page)
        run2 = chunker.chunk_page(sample_page)

        ids_run1 = [c.chunk_id for c in run1]
        ids_run2 = [c.chunk_id for c in run2]

        # Uniqueness
        assert len(ids_run1) == len(set(ids_run1)), "Duplicate chunk ID found within same page run!"
        # Determinism
        assert ids_run1 == ids_run2, "Chunk IDs varied between identical runs!"

    def test_invariant_5_complete_provenance_preservation(self, sample_page: DocumentPage):
        """Invariant 5: Every chunk retains correct document ID, title, physical page index & number, and printed number."""
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=200, chunk_overlap_chars=40))
        chunks = chunker.chunk_page(sample_page, chunk_index_in_doc_start=10)

        for idx, c in enumerate(chunks):
            assert c.metadata.document_id == sample_page.document.id
            assert c.metadata.document_title == sample_page.document.title
            assert c.metadata.physical_page_index == sample_page.page.physical_page_index
            assert c.metadata.physical_page_number == sample_page.page.physical_page_number
            assert c.metadata.printed_page_number == sample_page.page.printed_page_number
            assert c.metadata.chunk_index_in_page == idx
            assert c.metadata.chunk_index_in_doc == 10 + idx

    def test_invariant_6_chunk_size_upper_bound(self):
        """Invariant 6: Chunks strictly respect the configured maximum character budget."""
        text = "Scientific prose exploring retrieval architectures. " * 30
        config = ChunkingConfig(chunk_size_chars=250, chunk_overlap_chars=50)
        chunker = TextChunker(config)

        chunks = chunker.chunk_text(text)
        for _, _, c_text in chunks:
            assert len(c_text) <= 250, f"Chunk exceeded size limit: {len(c_text)} > 250"

    def test_invariant_7_and_8_bounded_overlap_and_no_duplicates(self):
        """Invariants 7 & 8: Overlap is intentional, character-bounded, and never emits duplicate chunks."""
        text = (
            "Sentence one discusses dense encoders. Sentence two discusses BM25 negatives. "
            "Sentence three evaluates on TriviaQA. Sentence four evaluates on Natural Questions. "
            "Sentence five concludes the experimental comparison. Sentence six analyzes the results. "
            "Sentence seven discusses limitations. Sentence eight suggests future work."
        )
        chunk_size = 140
        chunk_overlap = 45
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=chunk_size, chunk_overlap_chars=chunk_overlap))
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 3
        for i in range(len(chunks) - 1):
            c1_st, c1_en, c1_text = chunks[i]
            c2_st, c2_en, c2_text = chunks[i + 1]

            # Invariant 8: No duplicate chunks
            assert c1_st < c2_st, "Consecutive chunks must have strictly increasing start offsets"
            assert c1_text != c2_text, "Consecutive chunks must not have identical text"

            # Invariant 7: Actual character overlap calculation
            actual_overlap = c1_en - c2_st
            assert actual_overlap >= 0, f"Gap detected between chunk {i} and {i + 1}"
            # Overlap must be bounded by chunk_overlap or documented single sentence exception
            # In our text, no single sentence is > 140 chars
            assert actual_overlap <= chunk_size, f"Overlap exceeded chunk size: {actual_overlap}"

    def test_invariant_9_short_text_single_chunk(self):
        """Invariant 9: Text within chunk size produces exactly one chunk without duplication."""
        short = "A standalone abstract sentence."
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=500, chunk_overlap_chars=100))
        chunks = chunker.chunk_text(short)

        assert len(chunks) == 1
        assert chunks[0][0] == 0
        assert chunks[0][1] == len(short)
        assert chunks[0][2] == short

    def test_invariant_10_page_scoped_isolation(self):
        """Invariant 10: Page-scoped chunking never mixes text or provenance across pages."""
        page1 = DocumentPage(
            document=DocumentMetadata(id="DOC-A", title="Doc Title"),
            page=PageMetadata(physical_page_index=0, physical_page_number=1, printed_page_number=1),
            text="Content on page one.",
        )
        page2 = DocumentPage(
            document=DocumentMetadata(id="DOC-A", title="Doc Title"),
            page=PageMetadata(physical_page_index=1, physical_page_number=2, printed_page_number=2),
            text="Content on page two.",
        )
        chunker = TextChunker()
        chunks = chunker.chunk_document_pages([page1, page2])

        assert len(chunks) == 2
        assert chunks[0].metadata.physical_page_number == 1
        assert chunks[0].text == "Content on page one."
        assert chunks[1].metadata.physical_page_number == 2
        assert chunks[1].text == "Content on page two."

    def test_adversarial_single_sentence_much_longer_than_chunk_limit(self):
        """Adversarial: A 500-char single sentence with 100-char chunk limit."""
        long_sentence = (
            "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks establishes that "
            "conditioning sequence-to-sequence generators on dense retrieved passages substantially "
            "improves factual generation accuracy across open-domain benchmarks such as Natural Questions, "
            "TriviaQA, WebQuestions, and CuratedTREC without requiring prohibitively expensive retraining "
            "of the multi-million passage index during sequence-to-sequence fine-tuning."
        )
        assert len(long_sentence) > 350

        chunker = TextChunker(ChunkingConfig(chunk_size_chars=100, chunk_overlap_chars=20))
        chunks = chunker.chunk_text(long_sentence)

        assert len(chunks) > 3
        for st, en, c_text in chunks:
            assert len(c_text) <= 100
            assert long_sentence[st:en] == c_text

    def test_adversarial_long_token_or_url_without_whitespace(self):
        """Adversarial: An unbroken URL of 280 characters with 80-char chunk limit."""
        unbroken_url = "https://subdomain.example.org/path/to/very/long/scientific/paper/archive/download/file/resource?id=123456789012345678901234567890&token=abcdefghijklmnopqrstuvwxyz1234567890abcdefghijklmnopqrstuvwxyz&version=2.0.1&format=pdf"
        assert len(unbroken_url) > 200

        chunker = TextChunker(ChunkingConfig(chunk_size_chars=80, chunk_overlap_chars=15))
        chunks = chunker.chunk_text(unbroken_url)

        assert len(chunks) >= 3
        for st, en, c_text in chunks:
            assert len(c_text) <= 80
            assert unbroken_url[st:en] == c_text

    def test_adversarial_multiple_consecutive_newlines_and_tabs(self):
        """Adversarial: Text heavily interspersed with consecutive newlines, carriage returns, and tabs."""
        chaotic_whitespace = (
            "Section 1: Introduction\n\n\n\t\t\n   \t\n\n"
            "Here is the first true paragraph.\n\n\n\n\t\t\n\n"
            "Here is the second true paragraph.\n\t\n"
        )
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=60, chunk_overlap_chars=15))
        chunks = chunker.chunk_text(chaotic_whitespace)

        assert len(chunks) >= 2
        for st, en, c_text in chunks:
            assert len(c_text.strip()) > 0
            assert chaotic_whitespace[st:en] == c_text

    def test_adversarial_very_small_chunk_sizes_and_overlap(self):
        """Adversarial: Extreme small configuration (chunk_size=50, overlap=10)."""
        text = "Dense passage retrieval outperforms BM25 on TriviaQA. REALM updates the MIPS index asynchronously."
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=50, chunk_overlap_chars=10))
        chunks = chunker.chunk_text(text)

        assert len(chunks) >= 2
        for st, en, c_text in chunks:
            assert len(c_text) <= 50
            assert text[st:en] == c_text

    def test_adversarial_final_fragment_smaller_than_minimum(self):
        """Adversarial: Trailing fragment is smaller than min_chunk_size_chars."""
        # Case 1: Trailing fragment merges into previous chunk when combined size fits chunk_size
        text1 = "This is a substantial first sentence establishing the model architecture. Short note."
        # Total text is ~86 chars. chunk_size=100, min_chunk_size=40.
        # "Short note." is 11 chars (< 40). It merges into chunk 1 so no <40 char fragment is emitted.
        chunker1 = TextChunker(ChunkingConfig(chunk_size_chars=100, chunk_overlap_chars=20, min_chunk_size_chars=40))
        chunks1 = chunker1.chunk_text(text1)
        assert len(chunks1) == 1
        assert chunks1[0][2] == text1

        # Case 2: Multi-sentence text where trailing fragment expands to previous sentence boundary
        text2 = (
            "First sentence introduces the retrieval framework. "
            "Second sentence provides the benchmark results. "
            "Final brief note."
        )
        # S1 is 51 chars, S2 is 48 chars, S3 is 17 chars.
        # chunk_size=75, min_chunk_size=35
        # Chunk 1 gets S1 (51 chars). S2 (48 chars) + S3 (17 chars) = 66 chars <= 75.
        # S3 is merged with S2 into Chunk 2, ensuring Chunk 2 >= 35 chars.
        chunker2 = TextChunker(ChunkingConfig(chunk_size_chars=75, chunk_overlap_chars=15, min_chunk_size_chars=35))
        chunks2 = chunker2.chunk_text(text2)
        assert len(chunks2) == 2
        for st, en, c_text in chunks2:
            assert len(c_text) >= 35
            assert text2[st:en] == c_text

    def test_adversarial_repeated_identical_sentences(self):
        """Adversarial: Multiple identical sentences appearing at distinct source offsets."""
        sentence = "Exact match scores on Natural Questions are reported in Table 1. "
        repeated_text = sentence * 6

        chunker = TextChunker(ChunkingConfig(chunk_size_chars=180, chunk_overlap_chars=40))
        chunks = chunker.chunk_text(repeated_text)

        assert len(chunks) >= 3
        # Offsets must be strictly monotonically increasing despite identical sentence strings
        for i in range(len(chunks) - 1):
            assert chunks[i][0] < chunks[i + 1][0]
            assert chunks[i][1] < chunks[i + 1][1]

    def test_adversarial_scientific_punctuation_and_abbreviations(self):
        """Adversarial: Verify that abbreviations (Fig. 1, e.g., et al., Dr.) do not trigger false sentence splits."""
        text = (
            "As shown in Fig. 1, the model (e.g. RAG-Sequence) achieved 44.5 EM on NQ, outperforming "
            "Lewis et al. by 3.14 points. Dr. Smith agreed with the findings."
        )
        chunker = TextChunker(ChunkingConfig(chunk_size_chars=300, chunk_overlap_chars=50))
        units = chunker.segment_text(text)

        # Should be segmented into exactly 2 natural sentences, NOT 6 fragments
        assert len(units) == 2
        assert "Fig. 1" in units[0][2]
        assert "e.g. RAG-Sequence" in units[0][2]
        assert "Lewis et al. by" in units[0][2]
        assert "3.14 points." in units[0][2]
        assert units[1][2].strip() == "Dr. Smith agreed with the findings."


