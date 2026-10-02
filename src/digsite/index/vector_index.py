"""The vector index: nearest neighbours of a query among the stored embeddings."""

import numpy as np
from numpy.typing import NDArray


class VectorIndex[K]:
    """Exact nearest-neighbour search over unit-length vectors.

    Every query is compared with every stored vector: one matrix product. That
    is exact and, for corpora of up to a few hundred thousand passages, fast
    enough not to need an approximate index.

    Args:
        keys: The item each row of `vectors` belongs to.
        vectors: One unit-length embedding per row.
    """

    def __init__(self, keys: list[K], vectors: NDArray[np.float32]) -> None:
        if vectors.ndim != 2 or vectors.shape[0] != len(keys):
            raise ValueError(
                f"Expected one vector per key: {len(keys)} keys, vectors of shape {vectors.shape}"
            )
        self._keys = keys
        self._vectors = vectors

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def dimension(self) -> int:
        return int(self._vectors.shape[1])

    def search(self, query: NDArray[np.float32], limit: int = 10) -> list[tuple[K, float]]:
        """Return the items most similar to the query vector.

        Args:
            query: A unit-length vector of the index's dimension.
            limit: Maximum number of results.

        Returns:
            (key, cosine similarity) pairs by decreasing similarity; ties keep
            index order.
        """
        if limit <= 0 or not self._keys:
            return []
        if query.shape != (self.dimension,):
            raise ValueError(
                f"Query has shape {query.shape}, the index holds vectors of {self.dimension}"
            )
        scores = self._vectors @ query
        candidates = np.arange(len(scores))
        if len(scores) > limit:
            best = np.argpartition(-scores, limit - 1)[:limit]
            candidates = np.flatnonzero(scores >= scores[best].min())
        order = np.lexsort((candidates, -scores[candidates]))[:limit]
        return [(self._keys[position], float(scores[position])) for position in candidates[order]]
