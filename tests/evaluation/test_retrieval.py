import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from digsite.embedding import HashingEmbedder, Vectors
from digsite.evaluation.retrieval import (
    QueryMetrics,
    RetrievalDataset,
    embed_documents,
    evaluate,
    evaluate_queries,
    load_beir,
    load_labelled_queries,
    summarize,
)
from digsite.search.retriever import Hit


def write_beir(directory: Path) -> Path:
    documents = [
        {"_id": "d1", "title": "Caching", "text": "The cache stores results."},
        {"_id": "d2", "title": "", "text": "Logs rotate daily."},
        {"_id": 3, "text": "A numeric id and no title."},
    ]
    queries = [{"_id": "q1", "text": "cache"}, {"_id": 2, "text": "logs"}]
    (directory / "qrels").mkdir(parents=True)
    (directory / "corpus.jsonl").write_text(
        "\n".join(json.dumps(record) for record in documents) + "\n\n", encoding="utf-8"
    )
    (directory / "queries.jsonl").write_text(
        "\n".join(json.dumps(record) for record in queries), encoding="utf-8"
    )
    (directory / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\n2\td2\t1\n", encoding="utf-8"
    )
    return directory


def test_load_beir_reads_documents_queries_and_judgements(tmp_path: Path) -> None:
    dataset = load_beir(write_beir(tmp_path))

    assert dataset.documents == {
        "d1": "Caching\nThe cache stores results.",
        "d2": "Logs rotate daily.",
        "3": "A numeric id and no title.",
    }
    assert dataset.queries == {"q1": "cache", "2": "logs"}
    assert dataset.judgements == {"q1": {"d1": 2, "d2": 0}, "2": {"d2": 1}}


DATASET = RetrievalDataset(
    documents={"d1": "one", "d2": "two", "d3": "three"},
    queries={"q1": "first", "q2": "second", "q3": "nothing relevant", "unjudged": "ignored"},
    judgements={"q1": {"d1": 1}, "q2": {"d2": 1, "d3": 1}, "q3": {"d1": 0}, "missing": {"d1": 1}},
)


class CannedRetriever:
    """Returns a fixed ranking per query and records what it was asked."""

    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings
        self.asked: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 10) -> list[Hit[str]]:
        self.asked.append((query, limit))
        keys = self.rankings.get(query, [])
        return [Hit(key, 1.0 / rank) for rank, key in enumerate(keys, start=1)]


def test_evaluate_averages_over_the_queries_that_have_relevant_documents() -> None:
    retriever = CannedRetriever({"first": ["d1", "d2"], "second": ["d1", "d2"]})

    metrics = evaluate(DATASET, retriever)

    # q3 has no relevant document and "missing" has no query text: both are skipped.
    assert [query for query, _ in retriever.asked] == ["first", "second"]
    assert metrics.queries == 2
    # q1: perfect. q2: one of two relevant documents, at rank 2.
    assert metrics.mrr == pytest.approx((1.0 + 0.5) / 2)
    assert metrics.map == pytest.approx((1.0 + 0.25) / 2)
    assert metrics.precision_at_10 == pytest.approx((0.1 + 0.1) / 2)
    assert metrics.recall_at_100 == pytest.approx((1.0 + 0.5) / 2)
    assert 0.0 < metrics.ndcg_at_10 < 1.0


def test_evaluate_queries_scores_each_query_on_its_own() -> None:
    retriever = CannedRetriever({"first": ["d1", "d2"], "second": ["d1", "d2"]})

    per_query = evaluate_queries(DATASET, retriever)

    assert list(per_query) == ["q1", "q2"]
    assert per_query["q1"] == QueryMetrics(
        ndcg_at_10=1.0,
        average_precision=1.0,
        reciprocal_rank=1.0,
        precision_at_10=0.1,
        recall_at_100=1.0,
    )
    assert per_query["q2"].reciprocal_rank == 0.5
    assert per_query["q2"].recall_at_100 == 0.5


def test_summarize_is_the_mean_of_the_queries() -> None:
    retriever = CannedRetriever({"first": ["d1", "d2"], "second": ["d1", "d2"]})

    assert summarize(evaluate_queries(DATASET, retriever)) == evaluate(DATASET, retriever)


def test_evaluate_asks_for_a_deep_ranking() -> None:
    retriever = CannedRetriever({})

    evaluate(DATASET, retriever)

    assert [limit for _, limit in retriever.asked] == [1000, 1000]


def test_evaluate_with_nothing_to_evaluate_returns_zeros() -> None:
    empty = RetrievalDataset(documents={}, queries={}, judgements={})

    metrics = evaluate(empty, CannedRetriever({}))

    assert metrics.queries == 0
    assert metrics.ndcg_at_10 == 0.0


PAGES = {"https://example.com/a": "Text of a.", "https://example.com/b": "Text of b."}


def test_labelled_queries_are_their_own_ids(tmp_path: Path) -> None:
    path = tmp_path / "queries.tsv"
    path.write_text(
        "# a comment\n"
        "\n"
        "how to cache\thttps://example.com/a\n"
        "rotating logs\thttps://example.com/b\thttps://example.com/a \n",
        encoding="utf-8",
    )

    dataset = load_labelled_queries(path, PAGES)

    assert dataset.documents == PAGES
    assert dataset.queries == {"how to cache": "how to cache", "rotating logs": "rotating logs"}
    assert dataset.judgements == {
        "how to cache": {"https://example.com/a": 1},
        "rotating logs": {"https://example.com/b": 1, "https://example.com/a": 1},
    }


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("cache\thttps://example.com/a\ncache\thttps://example.com/b\n", "line 2: the query is"),
        ("cache\n", "line 1: the query has no relevant document"),
        (
            "cache\thttps://example.com/gone\n",
            "line 1: not in the corpus: https://example.com/gone",
        ),
    ],
)
def test_labelled_queries_that_cannot_be_evaluated_are_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "queries.tsv"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_labelled_queries(path, PAGES)


class CountingEmbedder(HashingEmbedder):
    """A hashing embedder that counts how many passages it is asked to embed."""

    def __init__(self) -> None:
        super().__init__(16)
        self.embedded = 0

    def embed_passages(self, texts: Sequence[str]) -> Vectors:
        self.embedded += len(texts)
        return super().embed_passages(texts)


def test_embed_documents_returns_one_row_per_document_in_order() -> None:
    embedder = HashingEmbedder(16)

    vectors = embed_documents(DATASET, embedder)

    assert vectors.shape == (3, 16)
    assert np.array_equal(vectors[1], embedder.embed_passages(["two"])[0])


def test_embed_documents_reads_the_cache_instead_of_embedding_again(tmp_path: Path) -> None:
    embedder = CountingEmbedder()

    first = embed_documents(DATASET, embedder, tmp_path / "cache")
    second = embed_documents(DATASET, embedder, tmp_path / "cache")

    assert embedder.embedded == 3
    assert np.array_equal(first, second)
    assert [path.name for path in (tmp_path / "cache").iterdir()] == ["hashing-16.npy"]


def test_embed_documents_ignores_a_cache_that_does_not_fit_the_collection(tmp_path: Path) -> None:
    embedder = CountingEmbedder()
    embed_documents(DATASET, embedder, tmp_path)
    larger = RetrievalDataset(
        documents={**DATASET.documents, "d4": "four"}, queries={}, judgements={}
    )

    vectors = embed_documents(larger, embedder, tmp_path)

    assert vectors.shape == (4, 16)
    assert embedder.embedded == 3 + 4
