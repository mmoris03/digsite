import pytest

from digsite.embedding import HashingEmbedder
from digsite.index.analyzer import Analyzer, AnalyzerSettings, Language
from digsite.index.pipeline import index_texts
from digsite.index.vector_index import VectorIndex
from digsite.search.dense import DenseSearcher
from digsite.search.hybrid import HybridSearcher
from digsite.search.lexical import LexicalSearcher
from digsite.search.retriever import Hit


class FixedRetriever:
    """Answers every query with the same ranking, and remembers what it was asked."""

    def __init__(self, *keys: str) -> None:
        self._keys = keys
        self.limits: list[int] = []

    def search(self, query: str, limit: int = 10) -> list[Hit[str]]:
        self.limits.append(limit)
        return [Hit(key, 1.0 / rank) for rank, key in enumerate(self._keys[:limit], start=1)]


def test_fuses_the_rankings_of_its_retrievers() -> None:
    hybrid = HybridSearcher([FixedRetriever("a", "b", "c"), FixedRetriever("b", "d", "a")])

    hits = hybrid.search("anything")

    assert [hit.key for hit in hits] == ["b", "a", "d", "c"]
    assert hits[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_finds_what_only_one_retriever_finds() -> None:
    hybrid = HybridSearcher([FixedRetriever("a"), FixedRetriever("b")])

    assert {hit.key for hit in hybrid.search("anything")} == {"a", "b"}


def test_respects_the_limit() -> None:
    hybrid = HybridSearcher([FixedRetriever("a", "b", "c"), FixedRetriever("d", "e")])

    assert len(hybrid.search("anything", limit=2)) == 2
    assert hybrid.search("anything", limit=0) == []


def test_asks_each_retriever_for_the_same_depth_whatever_the_limit() -> None:
    first, second = FixedRetriever("a"), FixedRetriever("b")

    HybridSearcher([first, second], depth=50).search("anything", limit=5)
    HybridSearcher([first, second], depth=50).search("anything", limit=200)

    assert first.limits == second.limits == [50, 50]


def test_fewer_results_are_the_start_of_more_results() -> None:
    hybrid = HybridSearcher(
        [FixedRetriever("a", "b", "c", "d", "e"), FixedRetriever("e", "d", "f", "a")], depth=3
    )

    everything = hybrid.search("anything", limit=100)

    # Only the top three of each ranking take part, however many results are wanted.
    assert {hit.key for hit in everything} == {"a", "b", "c", "e", "d", "f"}
    assert hybrid.search("anything", limit=2) == everything[:2]


def test_the_depth_must_be_positive() -> None:
    with pytest.raises(ValueError, match="depth"):
        HybridSearcher([FixedRetriever("a")], depth=0)


def test_an_item_ranked_low_by_one_can_win_thanks_to_the_other() -> None:
    # `z` is beyond the five results asked for in the first ranking, and first in the second.
    first = FixedRetriever("a", "b", "c", "d", "e", "f", "z")
    second = FixedRetriever("z", "q")

    hits = HybridSearcher([first, second]).search("anything", limit=5)

    assert hits[0].key == "z"


def test_weights_tilt_the_result_towards_a_retriever() -> None:
    retrievers = [FixedRetriever("a", "b"), FixedRetriever("b", "a")]

    assert HybridSearcher(retrievers, weights=[3, 1]).search("anything")[0].key == "a"
    assert HybridSearcher(retrievers, weights=[1, 3]).search("anything")[0].key == "b"


def test_combines_lexical_and_semantic_search() -> None:
    documents = {
        "caching": "The cache stores results. Entries are evicted when the cache is full.",
        "logging": "Every request is written to the access log. Logs rotate daily.",
        "install": "Install the package with pip and restart the service.",
    }
    analyzer = Analyzer(AnalyzerSettings(Language.ENGLISH))
    embedder = HashingEmbedder()
    lexical = LexicalSearcher(index_texts(documents.items(), analyzer), analyzer)
    semantic = DenseSearcher(
        VectorIndex(list(documents), embedder.embed_passages(list(documents.values()))), embedder
    )

    hits = HybridSearcher([lexical, semantic]).search("rotating logs")

    assert hits[0].key == "logging"
    # Lexical search alone returns only the document that shares a word with the query.
    assert len(lexical.search("rotating logs")) == 1
    assert len(hits) == 3


AUTHORITY = {"a": 3, "b": 1, "c": 2, "z": 1}


def test_a_prior_breaks_the_tie_between_items_the_retrievers_rank_alike() -> None:
    # `a` and `c` are first in one ranking each; the prior prefers `c`.
    retrievers = [FixedRetriever("a", "x"), FixedRetriever("c", "y")]

    plain = HybridSearcher(retrievers).search("anything")
    prior = {"a": 3, "c": 1, "x": 4, "y": 2}
    helped = HybridSearcher(retrievers, prior=prior.__getitem__, prior_weight=0.5)

    assert [hit.key for hit in plain] == ["a", "c", "x", "y"]
    assert [hit.key for hit in helped.search("anything")] == ["c", "a", "y", "x"]


def test_a_prior_adds_its_own_reciprocal_rank_to_the_score() -> None:
    hybrid = HybridSearcher(
        [FixedRetriever("a", "b")], prior=AUTHORITY.__getitem__, prior_weight=0.5
    )

    hits = hybrid.search("anything")

    # `b` is second for the retriever and first for the prior: not enough at this weight.
    assert hits == [
        Hit("a", pytest.approx(1 / 61 + 0.5 / 63)),
        Hit("b", pytest.approx(1 / 62 + 0.5 / 61)),
    ]


def test_a_heavy_prior_overrides_the_retrievers() -> None:
    hybrid = HybridSearcher(
        [FixedRetriever("a", "c", "b")], prior=AUTHORITY.__getitem__, prior_weight=5
    )

    assert [hit.key for hit in hybrid.search("anything")] == ["b", "c", "a"]


def test_a_prior_never_brings_in_what_no_retriever_found() -> None:
    hybrid = HybridSearcher([FixedRetriever("a")], prior=AUTHORITY.__getitem__, prior_weight=5)

    assert [hit.key for hit in hybrid.search("anything")] == ["a"]


def test_a_prior_is_applied_before_the_results_are_cut() -> None:
    # `z` is fourth for the retriever and first for the prior.
    hybrid = HybridSearcher(
        [FixedRetriever("a", "c", "b", "z")],
        prior={**AUTHORITY, "b": 4}.__getitem__,
        prior_weight=5,
    )

    assert [hit.key for hit in hybrid.search("anything", limit=1)] == ["z"]


def test_a_prior_without_weight_changes_nothing() -> None:
    retrievers = [FixedRetriever("a", "b", "c")]

    assert HybridSearcher(retrievers, prior=AUTHORITY.__getitem__).search("q") == HybridSearcher(
        retrievers
    ).search("q")


def test_a_prior_weight_needs_a_prior() -> None:
    with pytest.raises(ValueError, match="needs a prior"):
        HybridSearcher([FixedRetriever("a")], prior_weight=0.5)


def test_the_prior_weight_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        HybridSearcher([FixedRetriever("a")], prior=AUTHORITY.__getitem__, prior_weight=-1)


def test_needs_a_retriever() -> None:
    with pytest.raises(ValueError, match="at least one retriever"):
        HybridSearcher([])


def test_weights_must_match_the_retrievers() -> None:
    with pytest.raises(ValueError, match="one weight per retriever"):
        HybridSearcher([FixedRetriever("a")], weights=[1.0, 2.0])
