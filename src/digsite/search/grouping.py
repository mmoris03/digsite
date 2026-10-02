"""Collapsing a ranking of passages into a ranking of documents."""

from collections.abc import Callable, Hashable, Iterable

from digsite.search.retriever import Hit, Retriever

# Several passages of one document may rank high: how many to fetch per document wanted.
DEFAULT_PER_GROUP = 5


def best_per_group[K, G: Hashable](
    hits: Iterable[Hit[K]], group_of: Callable[[K], G], limit: int
) -> list[Hit[K]]:
    """Keep, for each group, only its best-ranked hit.

    Searching passages returns several from the same document. A result list
    should show each document once, represented by its best passage.

    Args:
        hits: Hits by decreasing score.
        group_of: Maps a hit's key to its group, e.g. a chunk to its document.
        limit: Maximum number of groups to return.
    """
    seen: set[G] = set()
    best: list[Hit[K]] = []
    for hit in hits:
        if len(best) >= limit:
            break
        group = group_of(hit.key)
        if group not in seen:
            seen.add(group)
            best.append(hit)
    return best


def limit_per_group[K, G: Hashable](
    keys: Iterable[K], group_of: Callable[[K], G], limit: int, per_group: int
) -> list[K]:
    """Take the first keys, letting no group contribute more than a few.

    A language model given six passages of one long page learns less than one
    given passages of three pages. This keeps the order of a ranking and skips
    a passage when its document already has its share.

    Args:
        keys: Keys by decreasing relevance.
        group_of: Maps a key to its group, e.g. a chunk to its document.
        limit: Maximum number of keys to return.
        per_group: Maximum number of keys of one group.
    """
    taken: dict[G, int] = {}
    chosen: list[K] = []
    for key in keys:
        if len(chosen) >= limit:
            break
        group = group_of(key)
        if taken.get(group, 0) < per_group:
            taken[group] = taken.get(group, 0) + 1
            chosen.append(key)
    return chosen


def search_best_per_group[K, G: Hashable](
    retriever: Retriever[K],
    query: str,
    group_of: Callable[[K], G],
    limit: int,
    *,
    per_group: int = DEFAULT_PER_GROUP,
) -> list[Hit[K]]:
    """Search passages and return the best one of each of the `limit` best documents.

    How many passages it takes to reach `limit` documents is not known
    beforehand: the best passages may all come from a few long documents. So
    more and more passages are asked for until there are enough documents or
    no passages are left. The first documents are therefore the same whatever
    `limit` is.

    Args:
        retriever: Searches the passages.
        query: What to look for.
        group_of: Maps a passage's key to its document's.
        limit: Maximum number of documents.
        per_group: Passages asked for, at first, for every document wanted.
    """
    if limit <= 0:
        return []
    wanted = limit * per_group
    while True:
        hits = retriever.search(query, wanted)
        best = best_per_group(hits, group_of, limit)
        if len(best) >= limit or len(hits) < wanted:
            return best
        wanted *= 2


class GroupedRetriever[K, G: Hashable]:
    """Presents a retriever of passages as a retriever of their documents.

    Args:
        retriever: Searches the passages.
        group_of: Maps a passage's key to its document's.
        per_group: See `search_best_per_group`.
    """

    def __init__(
        self,
        retriever: Retriever[K],
        group_of: Callable[[K], G],
        *,
        per_group: int = DEFAULT_PER_GROUP,
    ) -> None:
        self._retriever = retriever
        self._group_of = group_of
        self._per_group = per_group

    def search(self, query: str, limit: int = 10) -> list[Hit[G]]:
        """Return the documents of the best passages, each with the score of its best one."""
        best = search_best_per_group(
            self._retriever, query, self._group_of, limit, per_group=self._per_group
        )
        return [Hit(self._group_of(hit.key), hit.score) for hit in best]
