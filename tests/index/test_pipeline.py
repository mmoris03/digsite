import sqlite3

import pytest

from digsite.embedding import HashingEmbedder
from digsite.index.analyzer import Analyzer, AnalyzerSettings, Language
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.pipeline import (
    AuthorityStats,
    build_authority,
    build_index,
    document_passages,
    index_texts,
)
from digsite.models import Document, Duplicate, DuplicateKind, Link, Passage
from digsite.store import AuthorityStore, ChunkStore, CrawlStore, DocumentStore

SITE = "https://example.com"
ENGLISH = AnalyzerSettings(Language.ENGLISH)

CACHING = "# Caching\n\nEntries are evicted.\n\n## Tuning\n\nThe lru list decides what is dropped."
LOGGING = "# Logging\n\nLogs rotate daily."


@pytest.fixture
def chunk_store(connection: sqlite3.Connection) -> ChunkStore:
    return ChunkStore(connection)


@pytest.fixture
def index_store(connection: sqlite3.Connection) -> LexicalIndexStore:
    return LexicalIndexStore(connection)


def add_document(
    store: CrawlStore, documents: DocumentStore, path: str, title: str, text: str
) -> int:
    page_id = store.save_page(f"{SITE}{path}", 0, b"<p>x</p>")
    documents.save(page_id, title, text, f"hash-{page_id}", page_id if text else None)
    return page_id


def test_index_texts_analyzes_every_item() -> None:
    index = index_texts({"a": "Running cats", "b": "The cat runs"}.items(), Analyzer(ENGLISH))

    cat = index.postings("cat")

    assert index.keys == ["a", "b"]
    assert cat is not None
    assert cat.positions.tolist() == [0, 1]
    assert index.postings("the") is None


def test_passages_outside_any_heading_take_the_document_title() -> None:
    document = Document(1, f"{SITE}/a", "Glossary", "An opening line.\n\n# Terms\n\nA term.")

    assert document_passages(document) == [
        Passage(("Glossary",), "An opening line."),
        Passage(("Terms",), "A term."),
    ]


def test_passages_of_an_untitled_document_have_no_context() -> None:
    document = Document(1, f"{SITE}/a", "", "Just text.")

    assert document_passages(document) == [Passage((), "Just text.")]


def test_chunks_and_indexes_the_unique_documents(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    caching = add_document(store, document_store, "/a", "Caching", CACHING)
    logging = add_document(store, document_store, "/b", "Logging", LOGGING)
    copy = add_document(store, document_store, "/copy", "Caching", CACHING)
    add_document(store, document_store, "/empty", "Nothing", "")
    document_store.set_duplicates({copy: Duplicate(caching, DuplicateKind.EXACT)})

    stats = build_index(document_store, chunk_store, index_store, ENGLISH)

    chunks = chunk_store.chunks()
    assert [(chunk.page_id, chunk.context) for chunk in chunks] == [
        (caching, "Caching"),
        (caching, "Caching > Tuning"),
        (logging, "Logging"),
    ]
    loaded = index_store.load()
    assert loaded is not None
    index, settings = loaded
    assert settings == ENGLISH
    assert index.keys == [chunk.id for chunk in chunks]
    assert (stats.documents, stats.chunks) == (2, 3)
    assert (stats.terms, stats.postings) == (index.vocabulary_size, index.postings_count)
    assert (stats.embedded, stats.reused) == (0, 0)


def test_the_heading_path_is_searchable_in_every_chunk(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    add_document(store, document_store, "/a", "Caching", CACHING)

    build_index(document_store, chunk_store, index_store, ENGLISH)
    loaded = index_store.load()

    assert loaded is not None
    postings = loaded[0].postings("cach")
    assert postings is not None
    # "Caching" is only in the title, yet both chunks of the document match it.
    assert len(postings.positions) == 2


def test_respects_the_chunk_size(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    text = "# Long\n\n" + "\n\n".join(f"Paragraph number {number}." for number in range(40))
    add_document(store, document_store, "/a", "Long", text)

    stats = build_index(document_store, chunk_store, index_store, ENGLISH, max_chars=100)

    assert stats.chunks > 5
    assert all(len(chunk.text) <= 100 for chunk in chunk_store.chunks())


def test_embeds_every_chunk_with_the_given_model(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    add_document(store, document_store, "/a", "Caching", CACHING)
    add_document(store, document_store, "/b", "Logging", LOGGING)
    embedder = HashingEmbedder(32)

    stats = build_index(document_store, chunk_store, index_store, ENGLISH, embedder=embedder)

    keys, vectors = chunk_store.vectors(embedder.name)
    assert (stats.embedded, stats.reused) == (3, 0)
    assert keys == [chunk.id for chunk in chunk_store.chunks()]
    assert vectors.shape == (3, 32)
    assert chunk_store.embedding_model() == "hashing-32"


def test_a_second_run_only_embeds_what_changed(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    add_document(store, document_store, "/a", "Caching", CACHING)
    embedder = HashingEmbedder(32)
    build_index(document_store, chunk_store, index_store, ENGLISH, embedder=embedder)

    add_document(store, document_store, "/b", "Logging", LOGGING)
    stats = build_index(document_store, chunk_store, index_store, ENGLISH, embedder=embedder)

    assert (stats.chunks, stats.embedded, stats.reused) == (3, 1, 2)
    assert len(chunk_store.vectors(embedder.name)[0]) == 3


def test_indexing_without_embeddings_forgets_the_model(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    add_document(store, document_store, "/a", "Caching", CACHING)
    build_index(document_store, chunk_store, index_store, ENGLISH, embedder=HashingEmbedder(32))

    build_index(document_store, chunk_store, index_store, ENGLISH)

    assert chunk_store.embedding_model() is None


def test_documents_are_scored_by_the_links_between_them(
    store: CrawlStore, document_store: DocumentStore, connection: sqlite3.Connection
) -> None:
    authority_store = AuthorityStore(connection)

    def add_page(path: str, links: list[str]) -> int:
        targets = [Link(f"{SITE}{target}") for target in links]
        page_id = store.save_page(f"{SITE}{path}", 0, b"<p>x</p>", links=targets)
        document_store.save(page_id, path, f"text of {path}", f"hash-{page_id}", page_id)
        return page_id

    hub = add_page("/hub", ["/a"])
    a = add_page("/a", ["/hub", "/b", "https://elsewhere.example/"])
    b = add_page("/b", ["/moved"])
    copy = add_page("/b-copy", ["/hub"])
    store.record_redirect(f"{SITE}/moved", f"{SITE}/a")
    document_store.set_duplicates({copy: Duplicate(b, DuplicateKind.EXACT)})

    stats = build_authority(store, document_store, authority_store)

    scores = authority_store.scores()
    # hub -> a, a -> hub, a -> b, b -> a (through the redirect), b -> hub (from its copy).
    assert stats == AuthorityStats(documents=3, links=5)
    assert set(scores) == {hub, a, b}
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores[a] > scores[hub] > scores[b]


def test_scoring_again_forgets_documents_that_are_gone(
    store: CrawlStore, document_store: DocumentStore, connection: sqlite3.Connection
) -> None:
    authority_store = AuthorityStore(connection)
    first = add_document(store, document_store, "/a", "A", "Text of a.")
    second = add_document(store, document_store, "/b", "B", "Text of b.")
    build_authority(store, document_store, authority_store)

    document_store.set_duplicates({second: Duplicate(first, DuplicateKind.NEAR)})
    stats = build_authority(store, document_store, authority_store)

    assert stats == AuthorityStats(documents=1, links=0)
    assert authority_store.scores() == {first: 1.0}


def test_rebuilding_replaces_the_lexical_index(
    store: CrawlStore,
    document_store: DocumentStore,
    chunk_store: ChunkStore,
    index_store: LexicalIndexStore,
) -> None:
    add_document(store, document_store, "/a", "Caching", CACHING)
    build_index(document_store, chunk_store, index_store, ENGLISH)

    add_document(store, document_store, "/b", "Logging", LOGGING)
    stats = build_index(document_store, chunk_store, index_store, AnalyzerSettings(None))
    loaded = index_store.load()

    assert stats.documents == 2
    assert loaded is not None
    assert loaded[1] == AnalyzerSettings(None)
    assert loaded[0].postings("logs") is not None
