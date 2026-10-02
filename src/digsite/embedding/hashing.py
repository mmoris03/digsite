"""A model-free embedder, for tests and for running without downloading a model."""

import hashlib
from collections.abc import Sequence

import numpy as np

from digsite.embedding.base import Vector, Vectors
from digsite.text import tokenize


class HashingEmbedder:
    """Embeds a text as a hashed bag of its words.

    Each word adds +1 or -1 to one component chosen by its hash. Two texts are
    close when they share words, so this captures no meaning: it exists so that
    everything built on embeddings can run, deterministically and instantly,
    without a neural model.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 1:
            raise ValueError("An embedding needs at least one dimension")
        self._dimension = dimension

    @property
    def name(self) -> str:
        return f"hashing-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)

    def _embed(self, text: str) -> Vector:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            sign = 1.0 if digest & 1 else -1.0
            vector[(digest >> 1) % self._dimension] += sign
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector
