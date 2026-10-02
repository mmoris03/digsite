"""Hybrid search: several retrievers asked the same query, their rankings fused."""

from collections.abc import Callable, Hashable, Sequence

from digsite.search.fusion import DEFAULT_RANK_CONSTANT, reciprocal_rank_fusion
from digsite.search.retriever import Hit, Retriever

DEFAULT_DEPTH = 1000


class HybridSearcher[K: Hashable]:
    """Answers a query with the fusion of what several retrievers answer.

    Lexical search finds the exact terms of the query; semantic search finds
    passages that mean the same in other words. Each misses what the other
    finds, and an item that both rank high is a safer bet than the favourite of
    either.

    Optionally, a ranking that does not depend on the query, such as the
    importance of each item according to the links it receives, takes part in
    the fusion as a prior. It only reorders what the retrievers found: it never
    brings in an item that none of them returned.

    Args:
        retrievers: The retrievers to combine. They must index the same items.
        weights: How much each retriever counts. All the same by default.
        rank_constant: See `reciprocal_rank_fusion`.
        depth: Results asked of each retriever. An item that one retriever
            ranks low can still win if the other ranks it high, so this has to
            be much larger than the number of results wanted. It does not grow
            with that number: the fused ranking is the same however many
            results are asked for, and a search never returns more than the
            retrievers found at this depth.
        prior: Returns the position of an item, from 1, in the
            query-independent ranking.
        prior_weight: How much the prior counts, a retriever counting 1 by
            default. 0 ignores it.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever[K]],
        *,
        weights: Sequence[float] | None = None,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        depth: int = DEFAULT_DEPTH,
        prior: Callable[[K], int] | None = None,
        prior_weight: float = 0.0,
    ) -> None:
        if not retrievers:
            raise ValueError("Hybrid search needs at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(f"Expected one weight per retriever: {len(retrievers)} retrievers")
        if prior_weight < 0:
            raise ValueError("The weight of the prior must not be negative")
        if prior_weight > 0 and prior is None:
            raise ValueError("A prior weight needs a prior")
        if depth < 1:
            raise ValueError("The depth must be at least 1")
        self._retrievers = list(retrievers)
        self._weights = None if weights is None else list(weights)
        self._rank_constant = rank_constant
        self._depth = depth
        self._prior = prior
        self._prior_weight = prior_weight

    def search(self, query: str, limit: int = 10) -> list[Hit[K]]:
        """Return the items the retrievers agree on most, best first."""
        if limit <= 0:
            return []
        rankings = [
            [hit.key for hit in retriever.search(query, self._depth)]
            for retriever in self._retrievers
        ]
        fused = reciprocal_rank_fusion(
            rankings, weights=self._weights, rank_constant=self._rank_constant
        )
        if self._prior is not None and self._prior_weight > 0:
            fused = [
                Hit(
                    hit.key,
                    hit.score + self._prior_weight / (self._rank_constant + self._prior(hit.key)),
                )
                for hit in fused
            ]
            # Stable: items the prior does not tell apart keep their fused order.
            fused.sort(key=lambda hit: -hit.score)
        return fused[:limit]
