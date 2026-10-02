"""Embeddings from a sentence-transformers model."""

import logging
import os
from collections.abc import Sequence

import numpy as np

from digsite.embedding.base import Vector, Vectors

logger = logging.getLogger(__name__)

# Models of these families were trained with a marker in front of each text.
_PREFIXES = {"e5": ("query: ", "passage: ")}


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model as an `Embedder`.

    Args:
        model_name: A model from the Hugging Face hub or a local path. It is
            downloaded on first use.
        batch_size: Texts sent to the model at a time.
    """

    def __init__(self, model_name: str, *, batch_size: int = 32) -> None:
        # Windows users without developer mode get this warning on every load.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        # Imported here: it pulls in PyTorch, which takes seconds to load and
        # most commands never need.
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
        logger.info("loading embedding model %s", model_name)
        try:
            # A model that was downloaded before is loaded without asking the hub
            # whether it changed: faster, and it works with no network.
            self._model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except OSError:
            logger.info("downloading embedding model %s", model_name)
            self._model = SentenceTransformer(model_name, device="cpu")
        self._name = model_name
        self._batch_size = batch_size
        self._query_prefix, self._passage_prefix = next(
            (prefixes for family, prefixes in _PREFIXES.items() if family in model_name.lower()),
            ("", ""),
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError(f"{model_name} does not report the size of its embeddings")
        self._dimension = int(dimension)

    @property
    def name(self) -> str:
        return self._name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        return self._encode([self._passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._encode([self._query_prefix + text])[0]
        return vector

    def _encode(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
