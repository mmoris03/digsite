"""Semantic search: queries and items compared by the meaning of their text."""

from digsite.embedding import Embedder
from digsite.index.vector_index import VectorIndex
from digsite.search.retriever import Hit


class DenseSearcher[K]:
    """Answers text queries by nearest neighbours in embedding space.

    Args:
        index: The vectors to search.
        embedder: The model those vectors were computed with.
    """

    def __init__(self, index: VectorIndex[K], embedder: Embedder) -> None:
        if len(index) and index.dimension != embedder.dimension:
            raise ValueError(
                f"The index holds vectors of {index.dimension} dimensions "
                f"and {embedder.name} produces {embedder.dimension}"
            )
        self._index = index
        self._embedder = embedder

    def search(self, query: str, limit: int = 10) -> list[Hit[K]]:
        """Return the items closest in meaning to the query, closest first."""
        vector = self._embedder.embed_query(query)
        return [Hit(key, score) for key, score in self._index.search(vector, limit)]
