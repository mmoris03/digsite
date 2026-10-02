import numpy as np
import pytest

from digsite.embedding import HashingEmbedder
from digsite.index.vector_index import VectorIndex
from digsite.search.dense import DenseSearcher
from digsite.search.retriever import Hit

DOCUMENTS = {
    "caching": "The cache stores results. Entries are evicted when the cache is full.",
    "logging": "Every request is written to the access log. Logs rotate daily.",
    "install": "Install the package with pip and restart the service.",
}


def searcher(embedder: HashingEmbedder | None = None) -> DenseSearcher[str]:
    embedder = embedder or HashingEmbedder()
    vectors = embedder.embed_passages(list(DOCUMENTS.values()))
    return DenseSearcher(VectorIndex(list(DOCUMENTS), vectors), embedder)


def test_finds_the_closest_document() -> None:
    hits = searcher().search("when is the cache full")

    assert hits[0].key == "caching"
    assert all(isinstance(hit, Hit) for hit in hits)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_every_document_is_ranked_even_without_shared_words() -> None:
    # Unlike lexical search, similarity is defined for every pair.
    assert len(searcher().search("zeppelin", limit=10)) == 3


def test_respects_the_limit() -> None:
    assert len(searcher().search("cache", limit=2)) == 2


def test_the_embedder_must_match_the_index() -> None:
    vectors = HashingEmbedder(32).embed_passages(list(DOCUMENTS.values()))
    index = VectorIndex(list(DOCUMENTS), vectors)

    with pytest.raises(ValueError, match="32 dimensions"):
        DenseSearcher(index, HashingEmbedder(64))


def test_an_empty_index_can_be_searched() -> None:
    empty: VectorIndex[str] = VectorIndex([], np.zeros((0, 0), dtype=np.float32))

    assert DenseSearcher(empty, HashingEmbedder()).search("cache") == []
