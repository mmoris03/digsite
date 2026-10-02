"""Persistence: a single SQLite file holds the whole corpus."""

from digsite.store.authority_store import AuthorityStore
from digsite.store.chunk_store import ChunkStore
from digsite.store.crawl_store import CrawlStore
from digsite.store.database import SchemaVersionError, connect
from digsite.store.document_store import DocumentStore

__all__ = [
    "AuthorityStore",
    "ChunkStore",
    "CrawlStore",
    "DocumentStore",
    "SchemaVersionError",
    "connect",
]
