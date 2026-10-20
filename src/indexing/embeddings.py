"""Embedding generation wrappers for the Research Intelligence Engine.

Provides SentenceTransformerEmbedder for local dense embeddings and MockEmbedder
for fast, offline, deterministic testing.
"""

from typing import List, Optional, Protocol, runtime_checkable
import numpy as np

from src.indexing.models import EmbeddingError


@runtime_checkable
class BaseEmbedder(Protocol):
    """Protocol defining the embedding generator interface."""

    @property
    def model_name(self) -> str:
        """Name or identifier of the embedding model."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the generated embedding vectors."""
        ...

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a normalized dense vector."""
        ...

    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embed a batch of text strings into normalized dense vectors."""
        ...


class SentenceTransformerEmbedder:
    """Production embedder wrapping sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize SentenceTransformer embedder with lazy model loading.

        Args:
            model_name: HuggingFace model identifier.
            device: Optional compute device ('cpu', 'cuda', etc.).
            normalize_embeddings: Whether to apply L2 normalization to vectors.
        """
        self._model_name = model_name
        self._device = device
        self._normalize_embeddings = normalize_embeddings
        self._model = None
        self._embedding_dim: Optional[int] = None

    def _load_model(self):
        """Load the SentenceTransformer model once on demand."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name, device=self._device)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._embedding_dim = self._model.get_embedding_dimension()
                else:
                    self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                raise EmbeddingError(
                    f"Failed to load embedding model '{self._model_name}': {e}"
                ) from e
        return self._model

    @property
    def model_name(self) -> str:
        """Name of the embedding model."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Vector dimensionality of the model."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim  # type: ignore

    def embed_text(self, text: str) -> List[float]:
        """Generate a dense embedding vector for a single text string.

        Args:
            text: Input text string to embed.

        Returns:
            List of floats representing the dense embedding vector.

        Raises:
            EmbeddingError: If text is empty or embedding fails.
        """
        if not isinstance(text, str):
            raise EmbeddingError(f"Expected str input for text, got {type(text).__name__}")
        if not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text")

        model = self._load_model()
        try:
            vec = model.encode(
                text,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as e:
            raise EmbeddingError(f"Embedding generation failed: {e}") from e

        vec_arr = np.asarray(vec, dtype=np.float32)

        # Validate finiteness and shape
        if not np.isfinite(vec_arr).all():
            raise EmbeddingError("Embedding vector contains non-finite values (NaN or Inf)")

        expected_dim = self.embedding_dim
        if vec_arr.shape != (expected_dim,):
            raise EmbeddingError(
                f"Embedding vector shape mismatch: expected ({expected_dim},), got {vec_arr.shape}"
            )

        return vec_arr.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """Generate dense embeddings for a batch of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            List of embedding vectors, one for each input text.

        Raises:
            EmbeddingError: If any input text is invalid or embedding fails.
        """
        if not texts:
            return []

        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise EmbeddingError(f"Item at index {idx} is not a string: {type(t).__name__}")
            if not t.strip():
                raise EmbeddingError(f"Item at index {idx} is empty or whitespace-only")

        model = self._load_model()
        try:
            vecs = model.encode(
                texts,
                batch_size=batch_size or 32,
                normalize_embeddings=self._normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as e:
            raise EmbeddingError(f"Batch embedding generation failed: {e}") from e

        vecs_arr = np.asarray(vecs, dtype=np.float32)

        # Validate finiteness and batch shape
        if not np.isfinite(vecs_arr).all():
            raise EmbeddingError("Batch embedding contains non-finite values (NaN or Inf)")

        expected_shape = (len(texts), self.embedding_dim)
        if vecs_arr.shape != expected_shape:
            raise EmbeddingError(
                f"Batch embedding shape mismatch: expected {expected_shape}, got {vecs_arr.shape}"
            )

        return [row.tolist() for row in vecs_arr]


class MockEmbedder:
    """Deterministic, offline embedder for unit and integration testing.

    Generates stable, pseudo-random, L2-normalized vectors derived from input text
    hashes without downloading weights or requiring GPUs.
    """

    def __init__(
        self,
        model_name: str = "mock-minilm-l6-v2",
        embedding_dim: int = 384,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize MockEmbedder with fixed dimensionality.

        Args:
            model_name: Identifier for mock model.
            embedding_dim: Dimensionality of simulated vectors (default 384).
            normalize_embeddings: Whether to L2-normalize simulated vectors.
        """
        self._model_name = model_name
        self._embedding_dim = embedding_dim
        self._normalize_embeddings = normalize_embeddings

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _generate_vector(self, text: str) -> np.ndarray:
        """Generate a deterministic pseudo-vector from text hash."""
        # Use hashlib for deterministic seed across platforms
        import hashlib
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = np.random.default_rng(h % (2**32))
        vec = rng.standard_normal(self._embedding_dim).astype(np.float32)

        if self._normalize_embeddings:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise EmbeddingError(f"Expected str input for text, got {type(text).__name__}")
        if not text.strip():
            raise EmbeddingError("Cannot embed empty or whitespace-only text")

        vec = self._generate_vector(text)
        if not np.isfinite(vec).all():
            raise EmbeddingError("Vector contains non-finite values")
        return vec.tolist()

    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        if not texts:
            return []

        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise EmbeddingError(f"Item at index {idx} is not a string: {type(t).__name__}")
            if not t.strip():
                raise EmbeddingError(f"Item at index {idx} is empty or whitespace-only")

        results = [self._generate_vector(t).tolist() for t in texts]
        return results

