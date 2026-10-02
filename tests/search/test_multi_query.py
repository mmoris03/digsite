import pytest

from digsite.search.multi_query import MultiQueryRetriever, search_queries
from digsite.search.retriever import Hit


class CannedRetriever:
    """Answers each query with a fixed ranking, and remembers what it was asked."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self._rankings = rankings
        self.asked: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10) -> list[Hit[str]]:
        self.asked.append((query, limit))
        keys = self._rankings.get(query, [])[:limit]
        return [Hit(key, 10.0 - rank) for rank, key in enumerate(keys)]


RANKINGS = {
    "como leo argumentos": ["cmdline", "argparse", "tutorial"],
    "read arguments": ["argparse", "stdlib"],
    "argumentos": ["glossary"],
}


def test_what_several_queries_find_comes_first() -> None:
    retriever = CannedRetriever(RANKINGS)

    hits = search_queries(retriever, ["como leo argumentos", "read arguments"])

    # Both queries find `argparse`; each of the others is found by only one.
    assert [hit.key for hit in hits] == ["argparse", "cmdline", "stdlib", "tutorial"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_a_single_query_keeps_the_ranking_and_scores_of_the_retriever() -> None:
    retriever = CannedRetriever(RANKINGS)

    hits = search_queries(retriever, ["como leo argumentos"])

    assert hits == [Hit("cmdline", 10.0), Hit("argparse", 9.0), Hit("tutorial", 8.0)]


def test_every_query_is_searched_to_the_same_depth_and_the_result_is_cut() -> None:
    retriever = CannedRetriever(RANKINGS)

    hits = search_queries(retriever, list(RANKINGS), limit=2)

    assert retriever.asked == [("como leo argumentos", 2), ("read arguments", 2), ("argumentos", 2)]
    assert len(hits) == 2


def test_no_queries_find_nothing() -> None:
    assert search_queries(CannedRetriever(RANKINGS), []) == []


def test_a_multi_query_retriever_searches_for_what_the_rewriter_gives() -> None:
    retriever = CannedRetriever(RANKINGS)
    rewritten: list[str] = []

    def rewrite(query: str) -> list[str]:
        rewritten.append(query)
        return [query, "read arguments"]

    hits = MultiQueryRetriever(retriever, rewrite).search("como leo argumentos", limit=5)

    assert rewritten == ["como leo argumentos"]
    assert [query for query, _ in retriever.asked] == ["como leo argumentos", "read arguments"]
    assert hits[0].key == "argparse"
