"""The interface of an embedding model."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

type Vector = NDArray[np.float32]
type Vectors = NDArray[np.float32]


class Embedder(Protocol):
    """Turns text into vectors whose closeness reflects closeness in meaning.

    Every vector has unit length, so the dot product of two of them is their
    cosine similarity. Passages and queries are embedded by different methods
    because some models expect them to be marked differently.
    """

    @property
    def name(self) -> str:
        """Identifies the model. Vectors from different names cannot be compared."""
        ...

    @property
    def dimension(self) -> int:
        """Number of components of each vector."""
        ...

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        """Embed texts to be searched. Returns one row per text."""
        ...

    def embed_query(self, text: str) -> Vector:
        """Embed a search query."""
        ...
