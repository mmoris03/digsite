import pytest

from digsite.index.analyzer import Analyzer, AnalyzerSettings, Language
from digsite.index.pipeline import index_texts
from digsite.search.expansion import ExpansionSettings
from digsite.search.lexical import LexicalSearcher
from digsite.search.retriever import Hit

DOCUMENTS = {
    "caching": "The cache stores results. Entries are evicted when the cache is full.",
    "logging": "Every request is written to the access log. Logs rotate daily.",
    "install": "Install the package with pip and restart the service.",
}


def searcher(language: Language | None = Language.ENGLISH) -> LexicalSearcher[str]:
    analyzer = Analyzer(AnalyzerSettings(language))
    return LexicalSearcher(index_texts(DOCUMENTS.items(), analyzer), analyzer)


def test_finds_the_document_about_the_query() -> None:
    hits = searcher().search("how does cache eviction work")

    assert hits[0].key == "caching"
    assert hits[0].score > 0


def test_the_query_is_analyzed_like_the_documents() -> None:
    # "logs", "Logged" and "log" share a stem; "evicting" matches "evicted".
    assert searcher().search("LOGGED")[0].key == "logging"
    assert searcher().search("evicting")[0].key == "caching"


def test_without_stemming_only_exact_words_match() -> None:
    plain = searcher(language=None)

    assert plain.search("evicting") == []
    assert plain.search("evicted")[0].key == "caching"


def test_a_query_of_stop_words_matches_nothing() -> None:
    assert searcher().search("the is to") == []


def test_results_are_hits_limited_to_the_requested_number() -> None:
    hits = searcher().search("the cache and the log and the package", limit=2)

    assert len(hits) == 2
    assert all(isinstance(hit, Hit) for hit in hits)
    assert hits[0].score >= hits[1].score


def test_query_terms_are_the_analyzed_query_at_full_weight() -> None:
    assert searcher().query_terms("The caches are LOGGED") == {"cach": 1.0, "log": 1.0}


# Two documents say "cache" and also "lru"; a third is about the same thing but
# never uses the word "cache".
RELATED = {
    "policy": "The cache evicts entries with an lru policy.",
    "tuning": "Tune the cache: the lru list decides what the cache drops.",
    "theory": "An lru list keeps the most recent entries at the front.",
    "logging": "Every request is written to the access log.",
    "install": "Install the package and restart the service.",
}


def expanding_searcher(weight: float = 0.5) -> LexicalSearcher[str]:
    """Expand with the one term that both of the two best results share."""
    analyzer = Analyzer(AnalyzerSettings(Language.ENGLISH))
    settings = ExpansionSettings(
        feedback_documents=2, terms=1, weight=weight, min_feedback_documents=2
    )
    return LexicalSearcher(
        index_texts(RELATED.items(), analyzer),
        analyzer,
        expansion=settings,
        text_of=RELATED.__getitem__,
    )


def test_expansion_finds_a_document_that_shares_no_word_with_the_query() -> None:
    analyzer = Analyzer(AnalyzerSettings(Language.ENGLISH))
    plain = LexicalSearcher(index_texts(RELATED.items(), analyzer), analyzer)

    assert "theory" not in [hit.key for hit in plain.search("cache")]
    assert "theory" in [hit.key for hit in expanding_searcher().search("cache")]


def test_expansion_adds_weighted_terms_after_the_original_ones() -> None:
    assert expanding_searcher(weight=0.3).query_terms("cache") == {"cach": 1.0, "lru": 0.3}


def test_expansion_keeps_the_original_matches_on_top() -> None:
    keys = [hit.key for hit in expanding_searcher().search("cache")]

    assert set(keys[:2]) == {"policy", "tuning"}
    assert keys[2] == "theory"


def test_a_query_that_matches_nothing_is_not_expanded() -> None:
    expanded = expanding_searcher()

    assert expanded.query_terms("zeppelin") == {"zeppelin": 1.0}
    assert expanded.search("zeppelin") == []
    assert expanded.query_terms("the of") == {}


def test_expansion_needs_access_to_the_texts() -> None:
    analyzer = Analyzer(AnalyzerSettings(Language.ENGLISH))
    index = index_texts(RELATED.items(), analyzer)

    with pytest.raises(ValueError, match="text_of"):
        LexicalSearcher(index, analyzer, expansion=ExpansionSettings())
