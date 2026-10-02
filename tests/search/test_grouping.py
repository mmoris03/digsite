from digsite.search.grouping import (
    GroupedRetriever,
    best_per_group,
    limit_per_group,
    search_best_per_group,
)
from digsite.search.retriever import Hit

# Chunks 1-3 belong to document "a", 4-5 to "b", 6 to "c".
DOCUMENT_OF = {1: "a", 2: "a", 3: "a", 4: "b", 5: "b", 6: "c"}
HITS = [Hit(2, 0.9), Hit(1, 0.8), Hit(4, 0.7), Hit(3, 0.6), Hit(6, 0.5), Hit(5, 0.4)]


def test_keeps_the_best_hit_of_each_group_in_rank_order() -> None:
    best = best_per_group(HITS, DOCUMENT_OF.__getitem__, limit=10)

    assert best == [Hit(2, 0.9), Hit(4, 0.7), Hit(6, 0.5)]


def test_stops_at_the_limit() -> None:
    assert best_per_group(HITS, DOCUMENT_OF.__getitem__, limit=2) == [Hit(2, 0.9), Hit(4, 0.7)]
    assert best_per_group(HITS, DOCUMENT_OF.__getitem__, limit=0) == []


def test_does_not_read_more_hits_than_it_needs() -> None:
    consumed = []

    def hits():
        for hit in HITS:
            consumed.append(hit.key)
            yield hit

    best_per_group(hits(), DOCUMENT_OF.__getitem__, limit=1)

    assert consumed == [2, 1]


def test_no_hits_give_no_groups() -> None:
    assert best_per_group([], DOCUMENT_OF.__getitem__, limit=5) == []


def test_limit_per_group_skips_what_exceeds_a_groups_share() -> None:
    ranked = [2, 1, 4, 3, 6, 5]

    assert limit_per_group(ranked, DOCUMENT_OF.__getitem__, limit=10, per_group=1) == [2, 4, 6]
    assert limit_per_group(ranked, DOCUMENT_OF.__getitem__, limit=10, per_group=2) == [
        2, 1, 4, 6, 5,
    ]  # fmt: skip
    assert limit_per_group(ranked, DOCUMENT_OF.__getitem__, limit=10, per_group=3) == ranked


def test_limit_per_group_stops_at_the_limit() -> None:
    ranked = [2, 1, 4, 3, 6, 5]

    assert limit_per_group(ranked, DOCUMENT_OF.__getitem__, limit=3, per_group=2) == [2, 1, 4]
    assert limit_per_group(ranked, DOCUMENT_OF.__getitem__, limit=0, per_group=2) == []
    assert limit_per_group([], DOCUMENT_OF.__getitem__, limit=3, per_group=2) == []


class PassageRetriever:
    """Returns the sample hits, and remembers how many it was asked for."""

    def __init__(self) -> None:
        self.limits: list[int] = []

    def search(self, query: str, limit: int = 10) -> list[Hit[int]]:
        self.limits.append(limit)
        return HITS[:limit]


def test_a_grouped_retriever_returns_documents_with_the_score_of_their_best_passage() -> None:
    retriever = GroupedRetriever(PassageRetriever(), DOCUMENT_OF.__getitem__)

    assert retriever.search("anything") == [Hit("a", 0.9), Hit("b", 0.7), Hit("c", 0.5)]
    assert retriever.search("anything", limit=2) == [Hit("a", 0.9), Hit("b", 0.7)]


def test_a_grouped_retriever_asks_for_several_passages_per_document() -> None:
    passages = PassageRetriever()

    GroupedRetriever(passages, DOCUMENT_OF.__getitem__).search("anything", limit=3)
    GroupedRetriever(passages, DOCUMENT_OF.__getitem__, per_group=2).search("anything", limit=3)

    assert passages.limits == [15, 6]


def test_more_passages_are_asked_for_until_there_are_enough_documents() -> None:
    passages = PassageRetriever()

    # The best two passages belong to the same document, so two are not enough for two documents.
    best = search_best_per_group(passages, "anything", DOCUMENT_OF.__getitem__, 2, per_group=1)

    assert best == [Hit(2, 0.9), Hit(4, 0.7)]
    assert passages.limits == [2, 4]


def test_the_search_stops_when_the_passages_run_out() -> None:
    passages = PassageRetriever()

    best = search_best_per_group(passages, "anything", DOCUMENT_OF.__getitem__, 5, per_group=1)

    # There are only three documents among the six passages.
    assert [hit.key for hit in best] == [2, 4, 6]
    assert passages.limits == [5, 10]


def test_the_first_documents_do_not_depend_on_how_many_are_wanted() -> None:
    passages = PassageRetriever()

    def search(limit: int) -> list[Hit[int]]:
        return search_best_per_group(passages, "q", DOCUMENT_OF.__getitem__, limit, per_group=1)

    assert search(1) == search(3)[:1]
    assert search(2) == search(3)[:2]
    assert search(0) == []
