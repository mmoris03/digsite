"""Evaluation of retrieval against a test collection with relevance judgements."""

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from digsite.embedding import Embedder, Vectors
from digsite.evaluation import ranking_metrics
from digsite.search.retriever import Retriever

logger = logging.getLogger(__name__)

RANKING_DEPTH = 1000
_EMBEDDING_BATCH = 64


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    """A test collection: documents, queries and which documents answer which query.

    Attributes:
        documents: Text of each document, by id.
        queries: Text of each query, by id.
        judgements: For each query id, the gain of each judged document id.
    """

    documents: dict[str, str]
    queries: dict[str, str]
    judgements: dict[str, dict[str, int]]


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """How good the ranking returned for one query is."""

    ndcg_at_10: float
    average_precision: float
    reciprocal_rank: float
    precision_at_10: float
    recall_at_100: float


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Means over the evaluated queries."""

    queries: int
    ndcg_at_10: float
    map: float
    mrr: float
    precision_at_10: float
    recall_at_100: float


def load_beir(directory: Path, split: str = "test") -> RetrievalDataset:
    """Load a collection stored in the BEIR layout.

    Expects `corpus.jsonl` and `queries.jsonl`, one JSON object per line with
    `_id` and `text` (documents may also have `title`), and `qrels/<split>.tsv`
    with a header and `query-id`, `corpus-id`, `score` columns.
    """
    documents = {}
    for record in _read_jsonl(directory / "corpus.jsonl"):
        title = record.get("title") or ""
        documents[str(record["_id"])] = f"{title}\n{record['text']}".strip()

    queries = {
        str(record["_id"]): record["text"] for record in _read_jsonl(directory / "queries.jsonl")
    }

    judgements: dict[str, dict[str, int]] = {}
    with (directory / "qrels" / f"{split}.tsv").open(encoding="utf-8") as file:
        next(file)  # header
        for line in file:
            fields = line.split()
            if len(fields) >= 3:
                judgements.setdefault(fields[0], {})[fields[1]] = int(fields[2])
    return RetrievalDataset(documents, queries, judgements)


def load_labelled_queries(path: Path, documents: dict[str, str]) -> RetrievalDataset:
    """Load queries judged against documents that are already at hand.

    The file has one query per line: the query, then the ids of its relevant
    documents, separated by tabs. Blank lines and lines starting with `#` are
    ignored. A query is its own id.

    Args:
        path: The file with the queries.
        documents: Text of each document the queries were judged against, by id.

    Raises:
        ValueError: If a query is repeated, has no relevant document, or names
            a document that is not among `documents`. The last one usually
            means the file was written for a different corpus.
    """
    queries: dict[str, str] = {}
    judgements: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            query, *relevant = (field.strip() for field in line.split("\t"))
            relevant = [document for document in relevant if document]
            if query in queries:
                raise ValueError(f"{path}, line {number}: the query is repeated: {query!r}")
            if not relevant:
                raise ValueError(f"{path}, line {number}: the query has no relevant document")
            unknown = [document for document in relevant if document not in documents]
            if unknown:
                raise ValueError(f"{path}, line {number}: not in the corpus: {', '.join(unknown)}")
            queries[query] = query
            judgements[query] = dict.fromkeys(relevant, 1)
    return RetrievalDataset(documents, queries, judgements)


def evaluate_queries(
    dataset: RetrievalDataset, retriever: Retriever[str]
) -> dict[str, QueryMetrics]:
    """Run every judged query through a retriever and score each ranking.

    Queries without any relevant document are left out, as they cannot tell a
    good ranking from a bad one.

    Args:
        dataset: The test collection.
        retriever: Searches the documents of the collection, keyed by their id.

    Returns:
        The metrics of each evaluated query, by query id. Two retrievers
        evaluated on the same dataset give the same queries in the same order,
        which is what a paired comparison needs.
    """
    per_query = {}
    for query_id, gains in dataset.judgements.items():
        query = dataset.queries.get(query_id)
        if query is None or not any(gain > 0 for gain in gains.values()):
            continue
        ranking = [hit.key for hit in retriever.search(query, RANKING_DEPTH)]
        per_query[query_id] = QueryMetrics(
            ndcg_at_10=ranking_metrics.ndcg_at(ranking, gains, 10),
            average_precision=ranking_metrics.average_precision(ranking, gains),
            reciprocal_rank=ranking_metrics.reciprocal_rank(ranking, gains),
            precision_at_10=ranking_metrics.precision_at(ranking, gains, 10),
            recall_at_100=ranking_metrics.recall_at(ranking, gains, 100),
        )
    return per_query


def summarize(per_query: Mapping[str, QueryMetrics]) -> RetrievalMetrics:
    """Average the metrics of the evaluated queries."""
    count = len(per_query)
    if count == 0:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    metrics = per_query.values()
    return RetrievalMetrics(
        queries=count,
        ndcg_at_10=sum(query.ndcg_at_10 for query in metrics) / count,
        map=sum(query.average_precision for query in metrics) / count,
        mrr=sum(query.reciprocal_rank for query in metrics) / count,
        precision_at_10=sum(query.precision_at_10 for query in metrics) / count,
        recall_at_100=sum(query.recall_at_100 for query in metrics) / count,
    )


def evaluate(dataset: RetrievalDataset, retriever: Retriever[str]) -> RetrievalMetrics:
    """Run every judged query through a retriever and average the metrics."""
    return summarize(evaluate_queries(dataset, retriever))


def embed_documents(
    dataset: RetrievalDataset, embedder: Embedder, cache_directory: Path | None = None
) -> Vectors:
    """Embed every document of a collection, in the order of `dataset.documents`.

    Embedding a collection takes minutes and does not change between runs, so
    with a cache directory the vectors are saved there and read back next time.
    """
    cache = None
    if cache_directory is not None:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embedder.name)
        cache = cache_directory / f"{slug}.npy"
        if cache.exists():
            stored: Vectors = np.load(cache)
            if stored.shape == (len(dataset.documents), embedder.dimension):
                return stored

    texts = list(dataset.documents.values())
    batches = []
    for start in range(0, len(texts), _EMBEDDING_BATCH):
        batches.append(embedder.embed_passages(texts[start : start + _EMBEDDING_BATCH]))
        logger.info(
            "embedded %d/%d documents", min(start + _EMBEDDING_BATCH, len(texts)), len(texts)
        )
    vectors = (
        np.concatenate(batches) if batches else np.zeros((0, embedder.dimension), dtype=np.float32)
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, vectors)
    return vectors


def _read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
