import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import pytest

from digsite.store import CrawlStore, SchemaVersionError, connect
from digsite.store import database as database_module
from digsite.store.database import MIGRATIONS, schema_version


def table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {row[0] for row in rows}


def test_a_new_database_is_created_at_the_latest_schema(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "corpus.db"

    with closing(connect(path)) as connection:
        assert schema_version(connection) == len(MIGRATIONS)
        assert {
            "pages",
            "links",
            "redirects",
            "documents",
            "lexical_items",
            "lexical_terms",
            "settings",
            "chunks",
            "embeddings",
            "authority",
        } <= table_names(connection)

    assert path.exists()


def test_an_index_built_over_documents_is_discarded_when_chunks_arrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Up to version 3 the lexical index pointed at documents; from 4 on, at chunks.
    path = tmp_path / "corpus.db"
    monkeypatch.setattr(database_module, "MIGRATIONS", MIGRATIONS[:3])
    with closing(connect(path)) as old, old:
        page_id = CrawlStore(old).save_page("https://example.com/", 0, b"<p>home</p>")
        old.execute(
            "INSERT INTO lexical_items (position, item_id, length) VALUES (0, ?, 3)", (page_id,)
        )
        old.execute("INSERT INTO lexical_terms VALUES ('home', x'00000000', x'01000000')")
        old.execute("INSERT INTO settings VALUES ('lexical_index.analyzer', '{}')")
    monkeypatch.undo()

    with closing(connect(path)) as upgraded:
        assert schema_version(upgraded) == len(MIGRATIONS)
        for table in ("lexical_items", "lexical_terms", "settings", "chunks"):
            assert upgraded.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert CrawlStore(upgraded).html("https://example.com/") == b"<p>home</p>"


def test_a_corpus_crawled_by_the_first_release_gains_the_documents_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.db"
    monkeypatch.setattr(database_module, "MIGRATIONS", MIGRATIONS[:1])
    with closing(connect(path)) as old:
        CrawlStore(old).save_page("https://example.com/", 0, b"<p>home</p>")
        assert "documents" not in table_names(old)
    monkeypatch.undo()

    with closing(connect(path)) as upgraded:
        assert schema_version(upgraded) == len(MIGRATIONS)
        assert "documents" in table_names(upgraded)
        assert CrawlStore(upgraded).html("https://example.com/") == b"<p>home</p>"


def test_reopening_keeps_the_data_and_does_not_migrate_again(tmp_path: Path) -> None:
    path = tmp_path / "corpus.db"
    with closing(connect(path)) as connection, connection:
        connection.execute(
            "INSERT INTO redirects (source_url, target_url) VALUES ('https://a/', 'https://b/')"
        )

    with closing(connect(path)) as reopened:
        assert reopened.execute("SELECT COUNT(*) FROM redirects").fetchone()[0] == 1
        assert schema_version(reopened) == len(MIGRATIONS)


def test_an_older_database_is_upgraded_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.db"
    connect(path).close()
    newer = (*MIGRATIONS, "CREATE TABLE added_later (id INTEGER PRIMARY KEY) STRICT;")
    monkeypatch.setattr(database_module, "MIGRATIONS", newer)

    with closing(connect(path)) as upgraded:
        assert schema_version(upgraded) == len(newer)
        assert "added_later" in table_names(upgraded)
        assert "pages" in table_names(upgraded)


def test_a_failed_migration_leaves_the_database_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "corpus.db"
    connect(path).close()
    broken = (*MIGRATIONS, "CREATE TABLE half_done (id INTEGER); CREATE TABLE pages (x);")
    monkeypatch.setattr(database_module, "MIGRATIONS", broken)

    with pytest.raises(sqlite3.OperationalError):
        connect(path)

    monkeypatch.undo()
    with closing(connect(path)) as connection:
        assert schema_version(connection) == len(MIGRATIONS)
        assert "half_done" not in table_names(connection)


def test_refuses_a_database_from_a_newer_version(tmp_path: Path) -> None:
    path = tmp_path / "corpus.db"
    with closing(connect(path)) as connection:
        connection.execute(f"PRAGMA user_version = {len(MIGRATIONS) + 1}")

    with pytest.raises(SchemaVersionError):
        connect(path)


def test_enforces_foreign_keys_and_column_types(connection: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO links (source_id, target_url) VALUES (999, 'https://a/')")
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO redirects (source_url, target_url) VALUES ('https://a/', NULL)"
        )


def test_a_page_cannot_be_both_stored_and_skipped(connection: sqlite3.Connection) -> None:
    insert = (
        "INSERT INTO pages (url, depth, status, html, skip_reason, fetched_at) "
        "VALUES ('https://a/', 0, 200, ?, ?, '2026-01-01T00:00:00+00:00')"
    )

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(insert, (b"body", "robots"))
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(insert, (None, None))


def test_a_read_only_connection_reads_but_never_writes(tmp_path: Path) -> None:
    path = tmp_path / "corpus.db"
    with closing(connect(path)) as writer:
        CrawlStore(writer).save_page("https://example.com/", 0, b"<p>hello</p>")

    with closing(connect(path, read_only=True)) as reader:
        assert CrawlStore(reader).html("https://example.com/") == b"<p>hello</p>"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            CrawlStore(reader).save_page("https://example.com/other", 0, b"<p>x</p>")


def test_a_read_only_connection_can_be_shared_by_threads(tmp_path: Path) -> None:
    # With the sqlite3 module's statement cache, this returned wrong rows about
    # once in five runs, and failed with "API misuse" under heavier load.
    path = tmp_path / "corpus.db"
    urls = [f"https://example.com/{number}" for number in range(50)]
    with closing(connect(path)) as writer:
        store = CrawlStore(writer)
        for url in urls:
            store.save_page(url, 0, url.encode())

    def read_all(_: int) -> list[bytes | None]:
        return [CrawlStore(reader).html(url) for url in urls * 4]

    with (
        closing(connect(path, read_only=True)) as reader,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        results = list(pool.map(read_all, range(16)))

    assert all(bodies == [url.encode() for url in urls * 4] for bodies in results)


def test_a_missing_database_is_not_created_when_opened_read_only(tmp_path: Path) -> None:
    path = tmp_path / "corpus.db"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        connect(path, read_only=True)
    assert not path.exists()


@pytest.mark.parametrize(
    ("offset", "message"),
    [(-1, "run any other digsite command"), (1, "only knows up to")],
)
def test_a_read_only_connection_needs_the_latest_schema(
    tmp_path: Path, offset: int, message: str
) -> None:
    path = tmp_path / "corpus.db"
    connect(path).close()
    with closing(sqlite3.connect(path)) as raw:
        raw.execute(f"PRAGMA user_version = {len(MIGRATIONS) + offset}")

    with pytest.raises(SchemaVersionError, match=message):
        connect(path, read_only=True)
