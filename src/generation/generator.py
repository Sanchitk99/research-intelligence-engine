"""Core AnswerGenerator integrating retrieval, LLM completion, and citation provenance validation."""

import json
import re
import time
import unicodedata
from typing import Optional, List, Dict, Any, Tuple

from src.retrieval.models import RetrievedChunk, RetrievalResult
from src.retrieval.retriever import DenseRetriever
from src.generation.models import (
    AnswerStatus,
    Citation,
    GenerationConfig,
    GeneratedAnswer,
)
from src.generation.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    ProviderAuthError,
    ProviderTimeoutError,
    RateLimitError,
)
from src.generation.providers.factory import get_llm_provider
from src.generation.prompts import build_generation_messages


class AnswerGenerator:
    """Coordinates evidence retrieval, prompt formatting, LLM synthesis, and citation verification."""

    def __init__(
        self,
        retriever: Optional[Any] = None,
        provider: Optional[BaseLLMProvider] = None,
        config: Optional[GenerationConfig] = None,
        retrieval_mode: Optional[str] = None,
    ) -> None:
        """Initialize AnswerGenerator.

        Args:
            retriever: Optional DenseRetriever or HybridRetriever instance. If provided,
                it is directly used regardless of retrieval_mode (dependency injection).
                If None, lazily initialized according to retrieval_mode.
            provider: Optional BaseLLMProvider instance. If None, loaded via get_llm_provider.
            config: GenerationConfig settings.
            retrieval_mode: Optional retrieval mode ('dense' or 'hybrid').
                Overrides config.retrieval_mode if specified. Defaults to 'dense'.
        """
        self.config = config or GenerationConfig()
        self._retriever = retriever

        # Determine retrieval mode: explicit param > config > auto-detect from retriever > default 'dense'
        if retrieval_mode is not None:
            mode = retrieval_mode.lower().strip()
        elif retriever is not None:
            from src.retrieval.hybrid import HybridRetriever
            mode = "hybrid" if isinstance(retriever, HybridRetriever) else "dense"
        else:
            mode = getattr(self.config, "retrieval_mode", "dense").lower().strip()

        if mode not in ("dense", "hybrid"):
            raise ValueError(f"Invalid retrieval mode: '{mode}'. Must be 'dense' or 'hybrid'.")
        self.retrieval_mode = mode

        self.provider = provider or get_llm_provider(
            provider_type=self.config.provider_type,
            model_name=self.config.model_name,
        )

    @property
    def retriever(self) -> Any:
        """Lazily initialize retriever (DenseRetriever or HybridRetriever) when needed."""
        if self._retriever is None:
            if self.retrieval_mode == "hybrid":
                from src.retrieval.hybrid import HybridRetriever
                self._retriever = HybridRetriever()
            else:
                self._retriever = DenseRetriever()
        return self._retriever

    def close(self) -> None:
        """Close vector store resources if retriever was initialized."""
        if self._retriever is not None and hasattr(self._retriever, "close"):
            self._retriever.close()

    def __enter__(self) -> "AnswerGenerator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @staticmethod
    def _clean_json_text(raw_text: Optional[str]) -> str:
        """Strip markdown code fence blocks if returned by the LLM."""
        text = (raw_text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _normalize_for_matching(text: str) -> str:
        """Normalize text for excerpt grounding checks.

        Handles:
        - Unicode ligatures (fi, fl, ffi, ffl, ft, st) via NFKD decomposition.
        - Unicode dashes, hyphens, and minus signs (-, –, —, −).
        - Unicode single and double quotes, primes, and apostrophes.
        - PDF line-break hyphenation (e.g. 'regres-\nsion' -> 'regression').
        - Remaining hyphens at linebreaks (e.g. '1D-\nCNN' -> '1D-CNN').
        - Collapsing whitespace.
        - Lowercasing for case-insensitive comparison.
        """
        if not text:
            return ""
        # 1. Unicode decomposition (decomposes ligatures like fi, fl)
        t = unicodedata.normalize("NFKD", text)
        # 2. Normalize unicode dashes and hyphens to ASCII hyphen
        t = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", t)
        # 3. Normalize unicode single and double quotes
        t = re.sub(r"[\u2018\u2019\u201A\u201B\u2032]", "'", t)
        t = re.sub(r"[\u201C\u201D\u201E\u201F\u2033]", '"', t)
        # 4. Handle hyphenated line breaks: 'regres-\nsion' -> 'regression'
        t = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", t)
        # 5. Handle remaining hyphens at line breaks: '1D-\nCNN' -> '1D-CNN'
        t = re.sub(r"-\s*\n\s*", "-", t)
        # 6. Normalize whitespace and lowercase
        return " ".join(t.split()).lower()

    @classmethod
    def _is_excerpt_grounded(cls, raw_excerpt: str, chunk_text: str) -> bool:
        """Check if raw excerpt is grounded in chunk text under robust normalization."""
        if not raw_excerpt or not chunk_text:
            return False

        # 1. Fast direct check
        clean_chunk_raw = " ".join(chunk_text.split()).lower()
        clean_exc_raw = " ".join(raw_excerpt.strip(".\"'\n").split()).lower()
        if clean_exc_raw and clean_exc_raw in clean_chunk_raw:
            return True

        # 2. Robust normalized check (ligatures, linebreaks, punctuation)
        clean_chunk = cls._normalize_for_matching(chunk_text)
        stripped_exc = raw_excerpt.strip(".\"'\n \t\r\u201c\u201d\u2018\u2019\u2026")
        clean_exc = cls._normalize_for_matching(stripped_exc)

        if not clean_exc:
            return False

        if clean_exc in clean_chunk:
            return True

        # 3. Check for ellipsis-separated fragments (e.g. 'part1 ... part2')
        fragments = [
            f.strip() for f in re.split(r"\.{3,}|\u2026|\[\.\.\.\]", clean_exc)
            if len(f.strip()) >= 8
        ]
        if len(fragments) > 1:
            pos = 0
            all_found = True
            for frag in fragments:
                idx = clean_chunk.find(frag, pos)
                if idx == -1:
                    all_found = False
                    break
                pos = idx + len(frag)
            if all_found:
                return True

        return False

    def _validate_and_build_citations(
        self,
        raw_citations: Any,
        chunks_map: Dict[str, RetrievedChunk],
    ) -> Tuple[List[Citation], List[str]]:
        """Validate LLM-emitted citations against the retrieved chunks and build Citation instances.

        Enforces that:
        1. raw_citations must be a list of citation dictionaries.
        2. The chunk_id must exist in the actual retrieved chunks provided to the LLM.
        3. Source document metadata (id, title, page numbers) is populated from verified chunks,
           never fabricated by the LLM.
        4. Excerpt is checked against the source chunk text. If absent or invalid, it is reconciled
           with a grounded verbatim excerpt directly from the chunk.

        Args:
            raw_citations: Any object from model output intended as citation list.
            chunks_map: Mapping of chunk_id to RetrievedChunk.

        Returns:
            Tuple of (verified_citations, validation_warnings).
        """
        if not isinstance(raw_citations, list):
            return [], [f"Expected citations to be a list, got {type(raw_citations).__name__}"]

        verified: List[Citation] = []
        seen_chunk_ids = set()
        warnings: List[str] = []

        for i, raw_cite in enumerate(raw_citations):
            if not isinstance(raw_cite, dict):
                warnings.append(f"Citation {i} is not a dictionary.")
                continue

            raw_chunk_id = str(raw_cite.get("chunk_id", "")).strip()
            if not raw_chunk_id:
                warnings.append(f"Citation {i} is missing 'chunk_id'.")
                continue

            # Case-insensitive resolution against retrieved context chunks
            matched_chunk = None
            if raw_chunk_id in chunks_map:
                matched_chunk = chunks_map[raw_chunk_id]
            else:
                for cid, c in chunks_map.items():
                    if cid.lower() == raw_chunk_id.lower():
                        matched_chunk = c
                        break

            # If chunk is not in retrieved context, reject hallucinated citation
            if matched_chunk is None:
                warnings.append(f"Rejected hallucinated chunk_id '{raw_chunk_id}' not in retrieved context.")
                continue

            # Prevent duplicate citations for identical chunks
            if matched_chunk.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(matched_chunk.chunk_id)

            raw_excerpt = str(raw_cite.get("excerpt", "")).strip()

            if self._is_excerpt_grounded(raw_excerpt, matched_chunk.text):
                excerpt = raw_excerpt
            else:
                if raw_excerpt:
                    warnings.append(
                        f"Citation excerpt for '{matched_chunk.chunk_id}' was not found in passage text; "
                        "using verified passage prefix."
                    )
                excerpt = matched_chunk.text[:120].strip() + "..."

            verified.append(
                Citation(
                    document_id=matched_chunk.document_id,
                    document_title=matched_chunk.document_title,
                    physical_page_number=matched_chunk.physical_page_number,
                    printed_page_number=matched_chunk.printed_page_number,
                    chunk_id=matched_chunk.chunk_id,
                    excerpt=excerpt,
                )
            )

        return verified, warnings

    def _parse_model_output(
        self,
        content: str,
        chunks_map: Dict[str, RetrievedChunk],
    ) -> Tuple[str, AnswerStatus, List[Citation], Optional[str]]:
        """Parse raw model content into answer text, status, validated citations, and notes.

        Enforces strict response validation:
        1. Strips markdown fences.
        2. Verifies JSON structure and strictly validates required schema fields:
           - 'status' must exist and match AnswerStatus enum.
           - 'answer' must exist and be non-empty string.
           - 'citations' must be a list if provided.
        3. Plain text fallback:
           - Rejects arbitrary plain text without verified chunk IDs as AnswerStatus.ERROR.
           - Handles plain text with explicit abstention or conflicting evidence keywords.
           - Allows plain text only if it explicitly cites verified chunks present in context.
        """
        cleaned = self._clean_json_text(content)
        if not cleaned:
            return (
                "Empty response received from LLM provider.",
                AnswerStatus.ERROR,
                [],
                "Provider returned empty or whitespace-only response content.",
            )

        parsed_data = None
        json_parse_error = None

        try:
            parsed_data = json.loads(cleaned)
        except json.JSONDecodeError as err:
            json_parse_error = str(err)
            # Attempt to extract JSON object substring within outer braces
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    parsed_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed_data = None

        # --- Case 1: Structured JSON Response ---
        if isinstance(parsed_data, dict):
            # 1. Validate required 'status' field
            if "status" not in parsed_data:
                return (
                    "",
                    AnswerStatus.ERROR,
                    [],
                    "Schema validation failed: Missing required 'status' field in model JSON response.",
                )

            raw_status = str(parsed_data.get("status", "")).lower().strip()
            try:
                status = AnswerStatus(raw_status)
            except ValueError:
                return (
                    "",
                    AnswerStatus.ERROR,
                    [],
                    f"Schema validation failed: Invalid 'status' value '{raw_status}'. "
                    f"Must be one of: {[s.value for s in AnswerStatus]}",
                )

            # 2. Validate required 'answer' field
            if "answer" not in parsed_data or not isinstance(parsed_data["answer"], str):
                return (
                    "",
                    AnswerStatus.ERROR,
                    [],
                    "Schema validation failed: Missing or non-string 'answer' field in model JSON response.",
                )

            answer_text = parsed_data["answer"].strip()
            if not answer_text:
                return (
                    "",
                    AnswerStatus.ERROR,
                    [],
                    "Schema validation failed: 'answer' field in model JSON response cannot be empty.",
                )

            # 3. Validate 'citations' field
            raw_citations = parsed_data.get("citations", [])
            if not isinstance(raw_citations, list):
                return (
                    "",
                    AnswerStatus.ERROR,
                    [],
                    f"Schema validation failed: 'citations' field must be a list, got {type(raw_citations).__name__}.",
                )

            citations, cite_warnings = self._validate_and_build_citations(raw_citations, chunks_map)
            notes = parsed_data.get("confidence_notes")
            if cite_warnings:
                warning_str = " | ".join(cite_warnings)
                notes = f"{notes} ({warning_str})" if notes else warning_str

            return answer_text, status, citations, notes

        # --- Case 2: Plain Text Fallback (Non-JSON) ---
        lower_ans = cleaned.lower()

        # Check for abstention keywords in plain-text
        if any(w in lower_ans for w in ["does not contain", "cannot be answered", "no information", "not mentioned in", "insufficient evidence"]):
            return (
                cleaned,
                AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE,
                [],
                f"Parsed plain-text abstention (JSON parse error: {json_parse_error}).",
            )
        elif "out of scope" in lower_ans or "not covered in the provided" in lower_ans:
            return (
                cleaned,
                AnswerStatus.ABSTAINED_OUT_OF_SCOPE,
                [],
                f"Parsed plain-text out-of-scope abstention (JSON parse error: {json_parse_error}).",
            )
        elif "conflict" in lower_ans or "contradict" in lower_ans:
            return (
                cleaned,
                AnswerStatus.CONFLICTING_EVIDENCE,
                [],
                f"Parsed plain-text conflicting evidence (JSON parse error: {json_parse_error}).",
            )

        # Check if plain text contains explicit citations to verified chunks
        found_cids = re.findall(r"P\d+_p\d{2}_c\d{2}", cleaned)
        citations: List[Citation] = []
        for cid in found_cids:
            if cid in chunks_map and cid not in [c.chunk_id for c in citations]:
                c = chunks_map[cid]
                citations.append(
                    Citation(
                        document_id=c.document_id,
                        document_title=c.document_title,
                        physical_page_number=c.physical_page_number,
                        printed_page_number=c.printed_page_number,
                        chunk_id=c.chunk_id,
                        excerpt=c.text[:120].strip() + "...",
                    )
                )

        if citations:
            return (
                cleaned,
                AnswerStatus.ANSWERED,
                citations,
                f"Parsed from unstructured plain text response containing verified citations (JSON parse error: {json_parse_error}).",
            )

        # Arbitrary plain text with no valid JSON, no valid citations, and no abstention keywords
        # is REJECTED rather than silently treated as ANSWERED
        return (
            cleaned,
            AnswerStatus.ERROR,
            [],
            f"Strict response validation rejected unstructured plain text: "
            f"Response is not valid JSON and contains no verified chunk citations (JSON parse error: {json_parse_error}).",
        )

    def generate(
        self,
        question: str,
        retrieval_result: Optional[RetrievalResult] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> GeneratedAnswer:
        """Generate a cited, evidence-grounded answer for a research question.

        Args:
            question: User research question.
            retrieval_result: Optional pre-computed RetrievalResult. If None, retrieval is executed.
            top_k: Number of chunks to retrieve if retrieval is performed.
            min_score: Minimum similarity threshold.

        Returns:
            GeneratedAnswer with synthesized text, status, and verified citations.
        """
        start_time = time.perf_counter()
        q_cleaned = question.strip()
        if not q_cleaned:
            return GeneratedAnswer(
                question=question,
                answer="Question cannot be empty or whitespace-only.",
                status=AnswerStatus.ERROR,
                retrieved_chunks=[],
                model_name=self.provider.default_model,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=0.0,
            )

        # 1. Execute retrieval if not pre-provided
        if retrieval_result is None:
            k = top_k or self.config.top_k
            score_th = min_score if min_score is not None else self.config.min_score
            retrieval_result = self.retriever.retrieve(
                query=q_cleaned,
                top_k=k,
                min_score=score_th,
            )

        chunks = retrieval_result.chunks
        chunks_map = {c.chunk_id: c for c in chunks}

        # 2. Check for empty evidence (immediate abstention without calling LLM)
        if not chunks:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer="The Research Intelligence Engine could not find any relevant passages in the indexed research corpus for this question.",
                status=AnswerStatus.ABSTAINED_INSUFFICIENT_EVIDENCE,
                citations=[],
                retrieved_chunks=[],
                confidence_notes="Retrieval returned 0 chunks (or no chunks satisfied the similarity threshold).",
                model_name=self.provider.default_model,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
            )

        # 3. Format prompt messages
        messages = build_generation_messages(
            question=q_cleaned,
            chunks=chunks,
            system_prompt_override=None,
        )

        # 4. Call LLM provider with comprehensive error handling
        llm_response: Optional[LLMResponse] = None
        try:
            generation_kwargs = {
                "messages": messages,
                "model": self.config.model_name,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "timeout": self.config.timeout_seconds,
                "response_format": "json_object",
            }
            completion_limit = (
                self.config.max_completion_tokens
                if self.config.max_completion_tokens is not None
                else getattr(self.provider, "max_completion_tokens", None)
            )
            if completion_limit is not None:
                generation_kwargs["max_completion_tokens"] = completion_limit
            llm_response = self.provider.generate(
                **generation_kwargs,
            )
        except ProviderTimeoutError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer=f"Answer generation timed out after {self.config.timeout_seconds}s.",
                status=AnswerStatus.ERROR,
                citations=[],
                retrieved_chunks=chunks,
                confidence_notes=str(e),
                model_name=self.config.model_name,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
            )
        except ProviderAuthError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer="LLM provider authentication failed. Please check your API key configuration.",
                status=AnswerStatus.ERROR,
                citations=[],
                retrieved_chunks=chunks,
                confidence_notes=str(e),
                model_name=self.config.model_name,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
            )
        except RateLimitError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer="LLM provider rate limit exceeded. Please wait before retrying.",
                status=AnswerStatus.ERROR,
                citations=[],
                retrieved_chunks=chunks,
                confidence_notes=str(e),
                model_name=self.config.model_name,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
            )
        except ProviderError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer=f"LLM provider error encountered: {e}",
                status=AnswerStatus.ERROR,
                citations=[],
                retrieved_chunks=chunks,
                confidence_notes=str(e),
                model_name=self.config.model_name,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return GeneratedAnswer(
                question=q_cleaned,
                answer=f"Unexpected generation error: {e}",
                status=AnswerStatus.ERROR,
                citations=[],
                retrieved_chunks=chunks,
                confidence_notes=str(e),
                model_name=self.config.model_name,
                provider_name=self.provider.name,
                retrieval_mode=self.retrieval_mode,
                latency_ms=round(elapsed_ms, 2),
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 5. Parse output and validate citations
        answer_text, status, citations, notes = self._parse_model_output(
            content=llm_response.content,
            chunks_map=chunks_map,
        )

        if status == AnswerStatus.ERROR and not answer_text:
            answer_text = notes or "Model response failed strict schema validation."

        return GeneratedAnswer(
            question=q_cleaned,
            answer=answer_text,
            status=status,
            citations=citations,
            retrieved_chunks=chunks,
            confidence_notes=notes,
            model_name=llm_response.model,
            provider_name=self.provider.name,
            retrieval_mode=self.retrieval_mode,
            latency_ms=round(elapsed_ms, 2),
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        )

