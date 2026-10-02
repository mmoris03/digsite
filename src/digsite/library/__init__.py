"""Collections: one searchable corpus per website, listed, opened and built."""

from digsite.index.analyzer import Language
from digsite.library.build import (
    MAX_PAGES,
    BuildError,
    BuildRequest,
    BuildStage,
    build_collection,
    crawl_scope,
    describe,
)
from digsite.library.library import (
    DEFAULT_COLLECTION,
    CollectionSummary,
    Library,
    UnknownCollectionError,
    is_collection_id,
    new_collection_id,
)
from digsite.library.queue import BuildQueue, BuildStatus

__all__ = [
    "DEFAULT_COLLECTION",
    "MAX_PAGES",
    "BuildError",
    "BuildQueue",
    "BuildRequest",
    "BuildStage",
    "BuildStatus",
    "CollectionSummary",
    "Language",
    "Library",
    "UnknownCollectionError",
    "build_collection",
    "crawl_scope",
    "describe",
    "is_collection_id",
    "new_collection_id",
]
