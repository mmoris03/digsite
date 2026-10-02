import sqlite3

import pytest

from digsite.embedding import Embedder, HashingEmbedder, create_embedder
from digsite.index.analyzer import AnalyzerSettings, Language
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.pipeline import build_authority, build_index
from digsite.models import Link
from digsite.search.corpus import CorpusSearch, MissingIndexError, SearchMode
from digsite.search.expansion import ExpansionSettings
from digsite.search.results import section_of
from digsite.store import AuthorityStore, ChunkStore, CrawlStore, DocumentStore

SITE = "https://example.com"
PAGES = {
    "/caching": ("Caching", "The cache stores results. Entries are evicted when it is full."),
    "/logging": ("Logging", "Every request is written to the access log. Logs rotate daily."),
    "/home": ("Home", "Start here. Caching and logging each have their own page."),
}
# Both pages link to the home page, which links back to one of them.
LINKS = {"/caching": ["/home"], "/logging": ["/home"], "/home": ["/caching"]}


def add_pages(connection: sqlite3.Connection) -> dict[str, int]:
    """Store the sample pages and their documents. Returns the page id of each path."""
    store, documents = CrawlStore(connection), DocumentStore(connection)
    page_ids = {}
    for path, (title, text) in PAGES.items():
        links = [Link(f"{SITE}{target}") for target in LINKS[path]]
        page_id = store.save_page(f"{SITE}{path}", 0, b"<p>x</p>", links=links)
        documents.save(page_id, title, f"# {title}\n\n{text}", f"hash-{page_id}", page_id)
        page_ids[path] = page_id
    return page_ids


def index(connection: sqlite3.Connection, embedder: Embedder | None) -> None:
    build_index(
        DocumentStore(connection),
        ChunkStore(connection),
        LexicalIndexStore(connection),
        AnalyzerSettings(Language.ENGLISH),
        embedder=embedder,
    )


@pytest.fixture
def pages(connection: sqlite3.Connection) -> dict[str, int]:
    """An indexed corpus of three pages, embedded without a model and scored by links."""
    page_ids = add_pages(connection)
    index(connection, HashingEmbedder(64))
    build_authority(CrawlStore(connection), DocumentStore(connection), AuthorityStore(connection))
    return page_ids


def first_page(corpus: CorpusSearch, mode: SearchMode, query: str) -> int:
    return corpus.page_ids[corpus.retriever(mode).search(query)[0].key]


@pytest.mark.parametrize("mode", list(SearchMode))
def test_every_mode_finds_the_page_about_the_query(
    connection: sqlite3.Connection, pages: dict[str, int], mode: SearchMode
) -> None:
    corpus = CorpusSearch(connection)

    assert first_page(corpus, mode, "when are cache entries evicted") == pages["/caching"]
    assert first_page(corpus, mode, "logs rotate daily") == pages["/logging"]


def test_results_are_chunks_of_the_corpus(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection)

    hits = corpus.hybrid.search("cache")

    assert {hit.key for hit in hits} <= set(corpus.page_ids)
    assert set(corpus.page_ids.values()) == set(pages.values())


def test_the_default_mode_is_hybrid_when_the_corpus_has_embeddings(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection)

    assert corpus.has_embeddings
    assert corpus.default_mode is SearchMode.HYBRID


def test_without_embeddings_only_lexical_search_is_available(
    connection: sqlite3.Connection,
) -> None:
    page_ids = add_pages(connection)
    index(connection, None)
    corpus = CorpusSearch(connection)

    assert not corpus.has_embeddings
    assert corpus.default_mode is SearchMode.LEXICAL
    assert first_page(corpus, SearchMode.LEXICAL, "cache") == page_ids["/caching"]
    for mode in (SearchMode.SEMANTIC, SearchMode.HYBRID):
        with pytest.raises(MissingIndexError, match="has no embeddings"):
            corpus.retriever(mode)


def test_a_corpus_that_was_not_indexed_cannot_be_searched(connection: sqlite3.Connection) -> None:
    add_pages(connection)

    with pytest.raises(MissingIndexError, match="run 'digsite index' first"):
        CorpusSearch(connection).retriever(SearchMode.LEXICAL)


def test_authority_moves_the_most_linked_page_up(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    query = "cache entries"

    plain = CorpusSearch(connection)
    weighted = CorpusSearch(connection, authority_weight=50)

    # Every page is retrieved, as semantic search ranks them all. The home page
    # only mentions caching, but both other pages link to it.
    assert first_page(plain, SearchMode.HYBRID, query) == pages["/caching"]
    assert first_page(weighted, SearchMode.HYBRID, query) == pages["/home"]


def test_authority_weights_can_be_compared_on_one_corpus(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection)

    weighted = corpus.hybrid_with_authority(50)

    assert corpus.page_ids[weighted.search("cache entries")[0].key] == pages["/home"]
    assert first_page(corpus, SearchMode.HYBRID, "cache entries") == pages["/caching"]
    assert corpus.hybrid_with_authority(0).search("cache") == corpus.hybrid.search("cache")


def test_authority_does_not_touch_the_other_modes(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection, authority_weight=50)

    assert first_page(corpus, SearchMode.LEXICAL, "cache entries") == pages["/caching"]
    assert first_page(corpus, SearchMode.SEMANTIC, "cache entries") == pages["/caching"]


def test_authority_needs_the_links_to_have_been_scored(connection: sqlite3.Connection) -> None:
    add_pages(connection)
    index(connection, HashingEmbedder(64))

    with pytest.raises(MissingIndexError, match="has no link scores"):
        CorpusSearch(connection, authority_weight=1).retriever(SearchMode.HYBRID)


def test_expansion_applies_to_lexical_search_and_to_the_lexical_half_of_hybrid(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection, expansion=ExpansionSettings(terms=3))

    terms = corpus.lexical.query_terms("cache")

    assert len(terms) == 4
    assert first_page(corpus, SearchMode.HYBRID, "cache") == pages["/caching"]


def test_the_embedding_model_is_loaded_once_and_only_if_needed(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    loaded = []

    def factory(name: str) -> Embedder:
        loaded.append(name)
        return create_embedder(name)

    corpus = CorpusSearch(connection, embedder_factory=factory)

    corpus.retriever(SearchMode.LEXICAL).search("cache")
    assert loaded == []

    corpus.retriever(SearchMode.SEMANTIC).search("cache")
    corpus.retriever(SearchMode.HYBRID).search("cache")
    assert loaded == ["hashing-64"]
    assert corpus.retriever(SearchMode.LEXICAL) is corpus.lexical


def test_documents_are_found_with_their_best_passage(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection)

    results = corpus.find_documents("when are cache entries evicted", SearchMode.HYBRID)

    first = results[0]
    assert first.url == f"{SITE}/caching"
    assert first.title == "Caching"
    assert "evicted" in first.passage
    assert corpus.page_ids[first.chunk_id] == pages["/caching"]
    # Each document appears once, however many of its passages match.
    assert len({result.url for result in results}) == len(results)


def test_finding_documents_respects_the_limit_and_the_mode(
    connection: sqlite3.Connection, pages: dict[str, int]
) -> None:
    corpus = CorpusSearch(connection)

    assert len(corpus.find_documents("cache", SearchMode.SEMANTIC, limit=3)) == 3
    assert len(corpus.find_documents("cache", SearchMode.SEMANTIC, limit=1)) == 1
    lexical = corpus.find_documents("zeppelin", SearchMode.LEXICAL)
    assert lexical == []


def test_the_section_is_the_heading_path_below_the_title() -> None:
    assert section_of("Logging > Handlers > Files", "Logging") == "Handlers > Files"
    assert section_of("Logging", "Logging") == ""
    assert section_of("Other > Part", "Logging") == "Other > Part"


def test_the_stores_of_the_corpus_are_shared(connection: sqlite3.Connection) -> None:
    corpus = CorpusSearch(connection)

    assert corpus.chunks is corpus.chunks
    assert corpus.documents.stats().unique == 0
