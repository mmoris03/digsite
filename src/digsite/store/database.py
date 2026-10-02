"""SQLite connection and schema migrations.

The schema version lives in the database itself (`PRAGMA user_version`). Each
entry in `MIGRATIONS` moves the schema one version forward, so an existing
corpus is upgraded in place when a later release adds tables.
"""

import sqlite3
from pathlib import Path

MIGRATIONS: tuple[str, ...] = (
    # 1: crawl results
    """
    CREATE TABLE pages (
        id           INTEGER PRIMARY KEY,
        url          TEXT NOT NULL UNIQUE,
        depth        INTEGER NOT NULL,            -- hops from the nearest seed
        status       INTEGER NOT NULL,            -- HTTP status; 0 when there was no response
        content_type TEXT NOT NULL DEFAULT '',
        title        TEXT NOT NULL DEFAULT '',
        noindex      INTEGER NOT NULL DEFAULT 0,
        html         BLOB,                        -- zlib-compressed body, as received
        skip_reason  TEXT,
        skip_detail  TEXT NOT NULL DEFAULT '',
        fetched_at   TEXT NOT NULL,
        -- A requested URL is either stored or skipped, never both.
        CHECK ((html IS NULL) <> (skip_reason IS NULL))
    ) STRICT;

    CREATE TABLE links (
        source_id  INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
        target_url TEXT NOT NULL,
        anchor     TEXT NOT NULL DEFAULT '',
        nofollow   INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (source_id, target_url)
    ) STRICT, WITHOUT ROWID;

    CREATE INDEX links_by_target ON links(target_url);

    CREATE TABLE redirects (
        source_url TEXT PRIMARY KEY,
        target_url TEXT NOT NULL
    ) STRICT, WITHOUT ROWID;
    """,
    # 2: documents extracted from pages, and their duplicates
    """
    CREATE TABLE documents (
        page_id        INTEGER PRIMARY KEY REFERENCES pages(id) ON DELETE CASCADE,
        title          TEXT NOT NULL,
        text           TEXT NOT NULL,      -- main content as Markdown; '' if nothing was extracted
        content_hash   TEXT NOT NULL,      -- SHA-256 of the whitespace-normalised text
        simhash        BLOB,               -- fingerprint, big-endian
        duplicate_of   INTEGER REFERENCES documents(page_id),
        duplicate_kind TEXT,
        -- Only documents with content have a fingerprint.
        CHECK ((text = '') = (simhash IS NULL)),
        CHECK ((duplicate_of IS NULL) = (duplicate_kind IS NULL))
    ) STRICT;

    CREATE INDEX documents_by_duplicate ON documents(duplicate_of);
    """,
    # 3: lexical (inverted) index over the documents
    """
    CREATE TABLE lexical_items (
        position INTEGER PRIMARY KEY,      -- dense number from 0, as used in postings
        item_id  INTEGER NOT NULL UNIQUE,  -- page id of the indexed document
        length   INTEGER NOT NULL          -- number of terms
    ) STRICT;

    CREATE TABLE lexical_terms (
        term        TEXT PRIMARY KEY,
        positions   BLOB NOT NULL,         -- little-endian uint32 array
        frequencies BLOB NOT NULL          -- little-endian uint32 array, same length
    ) STRICT, WITHOUT ROWID;

    CREATE TABLE settings (
        name  TEXT PRIMARY KEY,
        value TEXT NOT NULL                -- JSON
    ) STRICT, WITHOUT ROWID;
    """,
    # 4: documents split into chunks, which become the indexed items, and their embeddings
    """
    CREATE TABLE chunks (
        id        INTEGER PRIMARY KEY,
        page_id   INTEGER NOT NULL REFERENCES documents(page_id) ON DELETE CASCADE,
        ordinal   INTEGER NOT NULL,        -- position within the document, from 0
        context   TEXT NOT NULL,           -- heading path, 'Title > Section'
        text      TEXT NOT NULL,
        text_hash TEXT NOT NULL,           -- SHA-256 of what is indexed: context and text
        UNIQUE (page_id, ordinal)
    ) STRICT;

    CREATE INDEX chunks_by_hash ON chunks(text_hash);

    -- Keyed by content, not by chunk: re-indexing reuses the vectors of unchanged text.
    CREATE TABLE embeddings (
        model     TEXT NOT NULL,
        text_hash TEXT NOT NULL,
        vector    BLOB NOT NULL,           -- little-endian float32 array
        PRIMARY KEY (model, text_hash)
    ) STRICT, WITHOUT ROWID;

    -- The lexical index of version 3 was built over documents; its items are now chunks.
    DELETE FROM lexical_items;
    DELETE FROM lexical_terms;
    DELETE FROM settings;
    """,
    # 5: importance of each document according to the link graph
    """
    CREATE TABLE authority (
        page_id  INTEGER PRIMARY KEY REFERENCES documents(page_id) ON DELETE CASCADE,
        pagerank REAL NOT NULL             -- the column adds up to 1
    ) STRICT;
    """,
)


class SchemaVersionError(RuntimeError):
    """The database was written by a newer version of this program."""


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the corpus database, creating it and applying pending migrations.

    Args:
        path: Database file, or ":memory:" for a throwaway database.
        read_only: Open an existing database without ever writing to it, not
            even to upgrade its schema. Such a connection may be used by
            several threads at once, as a web server does.

    Raises:
        SchemaVersionError: If the file has a newer schema than this code
            knows, or, when opening it read-only, an older one.
        FileNotFoundError: If the file does not exist and `read_only` is set.
    """
    if read_only:
        return _open_read_only(Path(path))
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        _migrate(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def _open_read_only(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    # SQLite itself is built serialized, so threads may share a connection, but
    # the sqlite3 module's statement cache hands one prepared statement to two
    # threads at once: rows come back wrong. Without it, each query prepares its own.
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro",
        uri=True,
        check_same_thread=False,
        cached_statements=0,
    )
    try:
        current = schema_version(connection)
        if current > len(MIGRATIONS):
            raise SchemaVersionError(
                f"Database schema is at version {current}, "
                f"but this version only knows up to {len(MIGRATIONS)}"
            )
        if current < len(MIGRATIONS):
            raise SchemaVersionError(
                f"Database schema is at version {current} and needs upgrading to "
                f"{len(MIGRATIONS)}; run any other digsite command on it first"
            )
    except BaseException:
        connection.close()
        raise
    return connection


def schema_version(connection: sqlite3.Connection) -> int:
    version: int = connection.execute("PRAGMA user_version").fetchone()[0]
    return version


def _migrate(connection: sqlite3.Connection) -> None:
    current = schema_version(connection)
    if current > len(MIGRATIONS):
        raise SchemaVersionError(
            f"Database schema is at version {current}, "
            f"but this version only knows up to {len(MIGRATIONS)}"
        )
    for number, script in enumerate(MIGRATIONS[current:], start=current + 1):
        try:
            # One transaction per migration: it is applied entirely or not at all.
            connection.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
        except sqlite3.Error:
            connection.rollback()
            raise
