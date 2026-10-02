import sqlite3

import pytest

from digsite.index.analyzer import Analyzer, AnalyzerSettings, Language
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.pipeline import index_texts
from digsite.search.lexical import LexicalSearcher

TEXTS = {
    10: "The cache stores results. Entries are evicted when the cache is full.",
    20: "Every request is written to the access log. Logs rotate daily.",
    35: "Install the package with pip and restart the service.",
}
SETTINGS = AnalyzerSettings(Language.ENGLISH, remove_stopwords=True, stem=True)


@pytest.fixture
def index_store(connection: sqlite3.Connection) -> LexicalIndexStore:
    return LexicalIndexStore(connection)


def test_an_empty_corpus_has_no_index(index_store: LexicalIndexStore) -> None:
    assert index_store.load() is None


def test_index_round_trip(index_store: LexicalIndexStore) -> None:
    index = index_texts(TEXTS.items(), Analyzer(SETTINGS))

    index_store.save(index, SETTINGS)
    loaded = index_store.load()

    assert loaded is not None
    restored, settings = loaded
    assert settings == SETTINGS
    assert restored.keys == [10, 20, 35]
    assert restored.lengths.tolist() == index.lengths.tolist()
    assert restored.vocabulary_size == index.vocabulary_size
    for term, postings in index.terms():
        copy = restored.postings(term)
        assert copy is not None
        assert copy.positions.tolist() == postings.positions.tolist()
        assert copy.frequencies.tolist() == postings.frequencies.tolist()


def test_a_loaded_index_ranks_exactly_like_the_original(index_store: LexicalIndexStore) -> None:
    analyzer = Analyzer(SETTINGS)
    index = index_texts(TEXTS.items(), analyzer)
    index_store.save(index, SETTINGS)
    loaded = index_store.load()
    assert loaded is not None

    original = LexicalSearcher(index, analyzer).search("cache logs package")
    restored = LexicalSearcher(loaded[0], Analyzer(loaded[1])).search("cache logs package")

    assert restored == original
    assert len(restored) == 3


@pytest.mark.parametrize(
    "settings",
    [
        AnalyzerSettings(None),
        AnalyzerSettings(Language.SPANISH, remove_stopwords=False, stem=True),
        AnalyzerSettings(Language.ENGLISH, remove_stopwords=True, stem=False),
    ],
)
def test_analyzer_settings_are_stored_with_the_index(
    index_store: LexicalIndexStore, settings: AnalyzerSettings
) -> None:
    index_store.save(index_texts(TEXTS.items(), Analyzer(settings)), settings)
    loaded = index_store.load()

    assert loaded is not None
    assert loaded[1] == settings


def test_saving_replaces_the_previous_index(index_store: LexicalIndexStore) -> None:
    index_store.save(index_texts(TEXTS.items(), Analyzer(SETTINGS)), SETTINGS)
    plain = AnalyzerSettings(None)

    index_store.save(index_texts({7: "only one document now"}.items(), Analyzer(plain)), plain)
    loaded = index_store.load()

    assert loaded is not None
    assert loaded[0].keys == [7]
    assert loaded[0].postings("cach") is None
    assert loaded[0].postings("document") is not None
    assert loaded[1] == plain


def test_an_empty_index_can_be_stored(index_store: LexicalIndexStore) -> None:
    index_store.save(index_texts({}.items(), Analyzer(SETTINGS)), SETTINGS)
    loaded = index_store.load()

    assert loaded is not None
    assert len(loaded[0]) == 0
