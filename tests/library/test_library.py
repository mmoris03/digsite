from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest

from digsite.embedding import Embedder, HashingEmbedder
from digsite.index.analyzer import Language
from digsite.library import (
    BuildRequest,
    Library,
    UnknownCollectionError,
    build_collection,
    is_collection_id,
    new_collection_id,
)
from digsite.search.corpus import MissingIndexError, SearchMode
from digsite.store import CrawlStore, SchemaVersionError, connect
from digsite.store.database import MIGRATIONS
from library.sites import GUIDE, guide_site


class CountingFactory:
    """Builds the model-free embedder, and counts how often it was asked to."""

    def __init__(self) -> None:
        self.built: Counter[str] = Counter()

    def __call__(self, name: str) -> Embedder:
        self.built[name] += 1
        return HashingEmbedder()


def add(library: Library, collection_id: str, *, embedder: Embedder | None = None) -> None:
    build_collection(
        BuildRequest(GUIDE, language=Language.ENGLISH),
        library.path_of(collection_id),
        embedder=embedder,
        client=guide_site().client,
        delay_seconds=0,
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://docs.python.org/3/tutorial/", "docs-python-org-3-tutorial"),
        ("https://docs.python.org/3/tutorial/index.html", "docs-python-org-3-tutorial"),
        ("https://www.example.com/", "example-com"),
        ("https://example.com/Ñandú/página.html", "example-com-and-p-gina-html"),
        ("https://example.com/" + "a" * 80, "example-com-" + "a" * 38),
        ("https:///???", "site"),
    ],
)
def test_ids_are_read_from_the_address(url: str, expected: str) -> None:
    assert new_collection_id(url, taken=set()) == expected
    assert is_collection_id(expected)


def test_ids_never_repeat() -> None:
    taken = {"example-com", "example-com-2"}

    assert new_collection_id("https://example.com/", taken) == "example-com-3"


@pytest.mark.parametrize("text", ["", "-a", "A", "a b", "../corpus", "a.db", "a" * 64])
def test_only_safe_file_names_are_ids(text: str) -> None:
    assert not is_collection_id(text)


def test_collections_are_listed_by_title_and_described(tmp_path: Path) -> None:
    library = Library(tmp_path, embedder_factory=CountingFactory())
    add(library, "guide-with-meaning", embedder=HashingEmbedder())
    add(library, "guide-words")
    (tmp_path / "half-built.db.partial").write_bytes(b"")
    (tmp_path / "Not An Id.db").write_bytes(b"")

    summaries = library.summaries()

    assert library.ids() == ["guide-with-meaning", "guide-words"]
    first, second = summaries
    assert first.title == second.title == "Example Guide"
    assert first.source == GUIDE
    assert first.documents == 3
    assert first.chunks > 0
    assert first.index_up_to_date
    assert first.modes == tuple(SearchMode)
    assert first.default_mode is SearchMode.HYBRID
    assert second.modes == (SearchMode.LEXICAL,)
    assert second.default_mode is SearchMode.LEXICAL
    library.close()


def test_a_corpus_from_the_command_line_is_a_collection_too(tmp_path: Path) -> None:
    library = Library(tmp_path)
    with closing(connect(tmp_path / "corpus.db")) as connection:
        CrawlStore(connection).save_page("https://example.com/", 0, b"<p>not indexed yet</p>")

    # Crawled but not indexed: there is nothing to search yet.
    assert library.ids() == ["corpus"]
    assert library.summaries() == []
    with pytest.raises(MissingIndexError):
        library.corpus("corpus")
    library.close()


def test_a_collection_without_a_description_is_named_by_its_id(tmp_path: Path) -> None:
    library = Library(tmp_path)
    add(library, "guide")
    with closing(connect(tmp_path / "guide.db")) as connection, connection:
        connection.execute("DELETE FROM settings WHERE name = 'collection.info'")

    (summary,) = library.summaries()

    assert (summary.title, summary.source) == ("guide", "")
    library.close()


def test_a_collection_of_another_version_is_left_out_of_the_list(tmp_path: Path) -> None:
    library = Library(tmp_path)
    add(library, "guide")
    connect(tmp_path / "old.db").close()
    with closing(connect(tmp_path / "old.db")) as connection:
        connection.execute(f"PRAGMA user_version = {len(MIGRATIONS) - 1}")

    assert [summary.id for summary in library.summaries()] == ["guide"]
    with pytest.raises(SchemaVersionError):
        library.summary("old")
    library.close()


def test_a_collection_is_loaded_once_and_shares_the_embedding_model(tmp_path: Path) -> None:
    factory = CountingFactory()
    library = Library(tmp_path, embedder_factory=factory)
    add(library, "first", embedder=HashingEmbedder())
    add(library, "second", embedder=HashingEmbedder())

    first = library.corpus("first")
    assert library.corpus("first") is first
    library.load_all()
    hits = library.corpus("second").find_documents("logs rotated", SearchMode.HYBRID)

    assert hits[0].url == f"{GUIDE}logging.html"
    assert factory.built == {"hashing-256": 1}
    library.close()


@pytest.mark.parametrize("collection_id", ["missing", "../../etc/passwd", "Bad"])
def test_an_unknown_collection_is_reported(tmp_path: Path, collection_id: str) -> None:
    library = Library(tmp_path)

    with pytest.raises(UnknownCollectionError):
        library.corpus(collection_id)
    with pytest.raises(UnknownCollectionError):
        library.summary(collection_id)


def test_what_unfinished_builds_left_is_removed(tmp_path: Path) -> None:
    library = Library(tmp_path / "new")
    (tmp_path / "new" / "a.db.partial").write_bytes(b"")
    (tmp_path / "new" / "kept.db").write_bytes(b"")

    assert library.remove_partial_builds() == ["a"]
    assert [path.name for path in (tmp_path / "new").iterdir()] == ["kept.db"]
