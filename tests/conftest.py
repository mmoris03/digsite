import sqlite3
from collections.abc import Iterator
from contextlib import closing

import pytest

from digsite.store import CrawlStore, DocumentStore, connect
from fakes import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def connection() -> Iterator[sqlite3.Connection]:
    with closing(connect(":memory:")) as opened:
        yield opened


@pytest.fixture
def store(connection: sqlite3.Connection) -> CrawlStore:
    return CrawlStore(connection)


@pytest.fixture
def document_store(connection: sqlite3.Connection) -> DocumentStore:
    return DocumentStore(connection)
