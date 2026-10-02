"""Combining several rankings of the same items into one."""

from collections.abc import Hashable, Sequence

from digsite.search.retriever import Hit

DEFAULT_RANK_CONSTANT = 60


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]],
    *,
    weights: Sequence[float] | None = None,
    rank_constant: int = DEFAULT_RANK_CONSTANT,
) -> list[Hit[K]]:
    """Fuse rankings by the positions of their items (Cormack et al., 2009).

    An item scores `weight / (rank_constant + position)` for each ranking it
    appears in, positions counted from 1, and the scores are added. Only
    positions are used, so rankings whose scores are not comparable, such as
    BM25 scores and cosine similarities, can be fused as they are.

    Args:
        rankings: Items by decreasing relevance, one sequence per ranking. An
            item repeated in a ranking counts at its first position.
        weights: How much each ranking counts. All the same by default.
        rank_constant: Dampens the advantage of the very first positions: the
            larger it is, the less being first instead of tenth matters.

    Returns:
        Every item of any ranking, by decreasing fused score. Ties keep the
        order in which the items were first seen.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"Expected one weight per ranking: {len(rankings)} rankings")
    if rank_constant < 0:
        raise ValueError("The rank constant must not be negative")

    scores: dict[K, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[K] = set()
        for key in ranking:
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + weight / (rank_constant + len(seen))
    # The sort is stable, so equal scores stay in first-seen order.
    return [Hit(key, score) for key, score in sorted(scores.items(), key=lambda item: -item[1])]
