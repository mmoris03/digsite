"""A directory of collections: one searchable corpus per file."""

import logging
import re
import sqlite3
import threading
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from digsite.embedding import Embedder, create_embedder
from digsite.search.corpus import CorpusSearch, SearchMode
from digsite.store import (
    CollectionStore,
    SchemaVersionError,
    connect,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "corpus"
SUFFIX = ".db"
# A collection being built is written under this suffix, then renamed when complete.
PARTIAL_SUFFIX = ".db.partial"

# Collection ids name files, so they are restricted to what is safe in any file system.
_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
_SLUG_LENGTH = 50


def is_collection_id(text: str) -> bool:
    return _ID_RE.fullmatch(text) is not None


def new_collection_id(url: str, taken: Collection[str]) -> str:
    """Make an id for a collection built from a URL, unlike any of `taken`.

    The id is readable: the host and path of the URL, e.g.
    'docs-python-org-3-tutorial' for https://docs.python.org/3/tutorial/.
    """
    parts = urlsplit(url)
    path = re.sub(r"/index\.html?$", "/", parts.path)
    slug = re.sub(r"[^a-z0-9]+", "-", f"{parts.hostname or ''}{path}".lower())
    slug = slug.removeprefix("www-").strip("-")[:_SLUG_LENGTH].strip("-") or "site"
    candidate, number = slug, 2
    while candidate in taken:
        candidate = f"{slug}-{number}"
        number += 1
    return candidate


class UnknownCollectionError(LookupError):
    """There is no collection with the given id."""


@dataclass(frozen=True, slots=True)
class CollectionSummary:
    """A collection as a list of them shows it.

    Attributes:
        id: Identifies the collection, and names its file.
        title: Name to show.
        source: Where its documents come from, if known.
        documents: Documents that are searched.
        chunks: Passages they are split into.
        modes: Search modes the collection allows.
        index_up_to_date: False if the documents changed after the index was built.
    """

    id: str
    title: str
    source: str
    documents: int
    chunks: int
    modes: tuple[SearchMode, ...]
    index_up_to_date: bool

    @property
    def default_mode(self) -> SearchMode:
        return SearchMode.HYBRID if SearchMode.HYBRID in self.modes else SearchMode.LEXICAL


class _OpenCollection:
    """A collection opened for searching, and whether its indexes are loaded yet."""

    def __init__(self, connection: sqlite3.Connection, corpus: CorpusSearch) -> None:
        self.connection = connection
        self.corpus = corpus
        self.lock = threading.Lock()
        self.loaded = False


class Library:
    """The collections in a directory, each in its own file, `<id>.db`.

    Collections are opened read-only, on a connection that request threads may
    share, and kept open. One embedding model of each name is loaded and
    shared by every collection, and by whatever builds new ones.

    Args:
        directory: Where the collections are. It is created if missing.
        embedder_factory: Builds the embedding model with a given name.
    """

    def __init__(
        self, directory: Path, *, embedder_factory: Callable[[str], Embedder] = create_embedder
    ) -> None:
        self._directory = directory
        self._embedder_factory = embedder_factory
        self._lock = threading.Lock()
        self._open: dict[str, _OpenCollection] = {}
        # Loading a model takes seconds: it must not hold up opening collections.
        self._embedders_lock = threading.Lock()
        self._embedders: dict[str, Embedder] = {}
        directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def path_of(self, collection_id: str) -> Path:
        """The file of a collection, which may not exist yet.

        Raises:
            UnknownCollectionError: If the id is not a valid collection id.
        """
        if not is_collection_id(collection_id):
            raise UnknownCollectionError(f"{collection_id!r} is not a collection id")
        return self._directory / f"{collection_id}{SUFFIX}"

    def ids(self) -> list[str]:
        """The ids of the collections in the directory, in alphabetical order."""
        return sorted(
            path.name.removesuffix(SUFFIX)
            for path in self._directory.glob(f"*{SUFFIX}")
            if is_collection_id(path.name.removesuffix(SUFFIX))
        )

    def embedder(self, name: str) -> Embedder:
        """The embedding model with the given name, loaded once and shared."""
        with self._embedders_lock:
            if name not in self._embedders:
                self._embedders[name] = self._embedder_factory(name)
            return self._embedders[name]

    def corpus(self, collection_id: str) -> CorpusSearch:
        """A collection, ready to search: its indexes are loaded the first time.

        Raises:
            UnknownCollectionError: If there is no such collection.
            MissingIndexError: If the collection was never indexed.
        """
        entry = self._entry(collection_id)
        with entry.lock:
            if not entry.loaded:
                corpus = entry.corpus
                corpus.retriever(corpus.default_mode)
                _ = corpus.page_ids
                entry.loaded = True
        return entry.corpus

    def summary(self, collection_id: str) -> CollectionSummary:
        """Describe a collection without loading its indexes.

        Raises:
            UnknownCollectionError: If there is no such collection.
        """
        entry = self._entry(collection_id)
        corpus = entry.corpus
        info = CollectionStore(entry.connection).info()
        documents = corpus.documents.stats().unique
        modes = tuple(SearchMode) if corpus.has_embeddings else (SearchMode.LEXICAL,)
        return CollectionSummary(
            id=collection_id,
            title=info.title if info else collection_id,
            source=info.source if info else "",
            documents=documents,
            chunks=corpus.chunks.count(),
            modes=modes,
            index_up_to_date=corpus.chunks.document_count() == documents,
        )

    def summaries(self) -> list[CollectionSummary]:
        """Describe every collection that can be searched, by title.

        A file that cannot be opened, or holds nothing indexed yet, is left out:
        it may be a corpus being built from the command line.
        """
        found = []
        for collection_id in self.ids():
            try:
                summary = self.summary(collection_id)
            except (SchemaVersionError, UnknownCollectionError) as error:
                logger.warning("leaving out collection %s: %s", collection_id, error)
                continue
            if summary.chunks:
                found.append(summary)
        return sorted(found, key=lambda summary: summary.title.casefold())

    def load_all(self) -> None:
        """Load the indexes of every collection, so that no search waits for it."""
        for summary in self.summaries():
            try:
                self.corpus(summary.id)
            except Exception:
                logger.exception("could not load collection %s", summary.id)

    def remove_partial_builds(self) -> list[str]:
        """Delete what builds that never finished left behind, and say which they were.

        Only to be called when no build is running, e.g. when a server starts.
        """
        removed = []
        for path in self._directory.glob(f"*{PARTIAL_SUFFIX}"):
            path.unlink(missing_ok=True)
            removed.append(path.name.removesuffix(PARTIAL_SUFFIX))
        return sorted(removed)

    def close(self) -> None:
        with self._lock:
            for entry in self._open.values():
                entry.connection.close()
            self._open.clear()

    def _entry(self, collection_id: str) -> _OpenCollection:
        path = self.path_of(collection_id)
        with self._lock:
            entry = self._open.get(collection_id)
            if entry is None:
                if not path.is_file():
                    raise UnknownCollectionError(f"there is no collection {collection_id!r}")
                connection = connect(path, read_only=True)
                corpus = CorpusSearch(connection, embedder_factory=self.embedder)
                entry = self._open[collection_id] = _OpenCollection(connection, corpus)
            return entry
