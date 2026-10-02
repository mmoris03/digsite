"""Searching for several wordings of one question at once."""

from collections.abc import Callable, Hashable, Sequence

from digsite.search.fusion import reciprocal_rank_fusion
from digsite.search.retriever import Hit, Retriever


def search_queries[K: Hashable](
    retriever: Retriever[K], queries: Sequence[str], limit: int = 10
) -> list[Hit[K]]:
    """Search for every query and fuse the rankings by reciprocal rank.

    A question can be worded in ways that find different things: as typed, as
    its key terms, in another language. An item that several wordings find
    comes before one that only one of them finds.

    Args:
        retriever: What to search with.
        queries: The wordings to search for. With only one, its ranking is
            returned as it is.
        limit: Results asked of the retriever for each query, and at most
            returned.
    """
    rankings = [retriever.search(query, limit) for query in queries]
    if len(rankings) == 1:
        return rankings[0]
    fused = reciprocal_rank_fusion([[hit.key for hit in ranking] for ranking in rankings])
    return fused[:limit]


class MultiQueryRetriever[K: Hashable]:
    """A retriever that searches for a query and for other wordings of it.

    Args:
        retriever: What to search with.
        rewrite: Returns the wordings to search for, given the query. They
            should include the query itself.
    """

    def __init__(self, retriever: Retriever[K], rewrite: Callable[[str], Sequence[str]]) -> None:
        self._retriever = retriever
        self._rewrite = rewrite

    def search(self, query: str, limit: int = 10) -> list[Hit[K]]:
        """Return what the wordings of the query find, fused, best first."""
        return search_queries(self._retriever, self._rewrite(query), limit)
