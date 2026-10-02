"""Commands that score a stage against labelled data."""

import argparse
import logging
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from digsite.answer.plan import plan_query
from digsite.cli import evaluate_answers
from digsite.cli.common import (
    Subparsers,
    analysis_options,
    analyzer_settings,
    authority_options,
    bm25_params,
    corpus_options,
    corpus_path,
    embedding_options,
    expansion_options,
    expansion_settings,
    fail,
    fusion_options,
    language_model,
    language_model_options,
    missing_corpus,
    ranking_options,
    verbosity_options,
)
from digsite.embedding import create_embedder
from digsite.evaluation import near_duplicates, retrieval
from digsite.evaluation.significance import paired_randomization_test
from digsite.index.analyzer import Analyzer
from digsite.index.pipeline import index_texts
from digsite.index.vector_index import VectorIndex
from digsite.ingest.dedup import DEFAULT_MAX_DISTANCE, SHINGLE_SIZE, SIMHASH_BITS
from digsite.llm import LanguageModel, LanguageModelError
from digsite.search.corpus import CorpusSearch, MissingIndexError, SearchMode
from digsite.search.dense import DenseSearcher
from digsite.search.grouping import GroupedRetriever
from digsite.search.hybrid import HybridSearcher
from digsite.search.lexical import LexicalSearcher
from digsite.search.multi_query import MultiQueryRetriever
from digsite.search.retriever import Retriever
from digsite.store import DocumentStore, connect

logger = logging.getLogger(__name__)

# Names, in the results, of hybrid search with link authority and with a language model's help.
_WITH_AUTHORITY = "hybrid+links"
_WITH_REWRITES = "hybrid+rewrites"


def register(commands: Subparsers) -> None:
    evaluate = commands.add_parser("eval", help="evaluate a stage against labelled data")
    evaluations = evaluate.add_subparsers(dest="evaluation", required=True)
    verbosity = verbosity_options()

    duplicates = evaluations.add_parser(
        "near-duplicates",
        parents=[verbosity],
        help="score near-duplicate detection against labelled duplicate pairs",
    )
    duplicates.add_argument(
        "--texts", type=Path, required=True, help="file with one 'id text' entry per line"
    )
    duplicates.add_argument(
        "--pairs", type=Path, required=True, help="file with one duplicate 'id id' pair per line"
    )
    duplicates.add_argument(
        "--bits",
        type=int,
        default=SIMHASH_BITS,
        help=f"fingerprint length (default: {SIMHASH_BITS})",
    )
    duplicates.add_argument(
        "--shingle-size",
        type=int,
        default=SHINGLE_SIZE,
        help=f"words per shingle (default: {SHINGLE_SIZE})",
    )
    duplicates.add_argument(
        "--distance",
        type=int,
        action="append",
        metavar="BITS",
        help=f"maximum Hamming distance to score (repeatable; default: {DEFAULT_MAX_DISTANCE})",
    )
    duplicates.set_defaults(handler=_near_duplicates)

    modes = [mode.value for mode in SearchMode]
    ranking = evaluations.add_parser(
        "retrieval",
        parents=[
            verbosity,
            analysis_options(),
            ranking_options(),
            expansion_options(),
            embedding_options(),
            fusion_options(),
        ],
        help="score retrieval on a test collection in the BEIR layout",
    )
    ranking.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="directory with corpus.jsonl, queries.jsonl and qrels/test.tsv",
    )
    ranking.add_argument(
        "--retriever",
        action="append",
        choices=modes,
        help="match words (lexical), meaning (semantic) or both (hybrid); "
        "repeat it to compare several (default: lexical)",
    )
    ranking.set_defaults(handler=_retrieval)

    search = evaluations.add_parser(
        "search",
        parents=[
            corpus_options(),
            ranking_options(),
            expansion_options(),
            fusion_options(),
            authority_options(),
            language_model_options(),
        ],
        help="score search on the crawled corpus against queries labelled with their pages",
    )
    search.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="file with one query per line, followed by the URLs of its relevant pages, "
        "separated by tabs",
    )
    search.add_argument(
        "--retriever",
        action="append",
        choices=modes,
        help="search mode to score; repeat it to compare several "
        "(default: every mode the corpus was indexed for)",
    )
    search.add_argument(
        "--rewrites",
        type=int,
        default=0,
        help="also score hybrid search helped by a language model, which adds this many "
        "searches to each query (default: 0, no language model is used)",
    )
    search.set_defaults(handler=_search)

    evaluate_answers.register(evaluations)


def _near_duplicates(args: argparse.Namespace) -> int:
    for path in (args.texts, args.pairs):
        if not path.exists():
            return fail(f"{path} does not exist")
    texts = near_duplicates.load_texts(args.texts)
    truth = near_duplicates.load_pairs(args.pairs)
    results = near_duplicates.evaluate(
        texts,
        truth,
        bits=args.bits,
        shingle_size=args.shingle_size,
        distances=args.distance or [DEFAULT_MAX_DISTANCE],
    )

    print(
        f"{len(texts)} texts, {len(truth)} duplicate pairs; "
        f"{args.bits}-bit fingerprints of {args.shingle_size}-word shingles"
    )
    print(
        f"{'distance':>8} {'precision':>10} {'recall':>8} {'f1':>8} {'tp':>7} {'fp':>7} {'fn':>7}"
    )
    for distance, metrics in results.items():
        print(
            f"{distance:>8} {metrics.precision:>10.4f} {metrics.recall:>8.4f} {metrics.f1:>8.4f} "
            f"{metrics.true_positives:>7} {metrics.false_positives:>7} "
            f"{metrics.false_negatives:>7}"
        )
    return 0


def _retrieval(args: argparse.Namespace) -> int:
    for name in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"):
        if not (args.dataset / name).exists():
            return fail(f"{args.dataset / name} does not exist")
    modes = [SearchMode(name) for name in dict.fromkeys(args.retriever or ["lexical"])]
    if args.expand and SearchMode.LEXICAL not in modes and SearchMode.HYBRID not in modes:
        return fail("--expand does not apply to semantic search")
    dataset = retrieval.load_beir(args.dataset)

    lexical: Retriever[str] | None = None
    semantic: Retriever[str] | None = None
    if SearchMode.LEXICAL in modes or SearchMode.HYBRID in modes:
        analyzer = Analyzer(analyzer_settings(args))
        lexical = LexicalSearcher(
            index_texts(dataset.documents.items(), analyzer),
            analyzer,
            bm25_params(args),
            expansion=expansion_settings(args),
            text_of=dataset.documents.__getitem__,
        )
    if SearchMode.SEMANTIC in modes or SearchMode.HYBRID in modes:
        embedder = create_embedder(args.model)
        vectors = retrieval.embed_documents(dataset, embedder, args.dataset / ".cache")
        semantic = DenseSearcher(VectorIndex(list(dataset.documents), vectors), embedder)

    results = []
    for mode in modes:
        retriever: Retriever[str]
        if mode is SearchMode.HYBRID:
            assert lexical is not None
            assert semantic is not None
            retriever = HybridSearcher([lexical, semantic], rank_constant=args.rank_constant)
        else:
            chosen = lexical if mode is SearchMode.LEXICAL else semantic
            assert chosen is not None
            retriever = chosen
        results.append((mode.value, retrieval.evaluate_queries(dataset, retriever)))
    _report(len(dataset.documents), results)
    return 0


def _search(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    if not args.queries.exists():
        return fail(f"{args.queries} does not exist")

    with closing(connect(path)) as connection:
        documents = DocumentStore(connection).documents()
        url_of = {document.page_id: document.url for document in documents}
        try:
            dataset = retrieval.load_labelled_queries(
                args.queries, {document.url: document.text for document in documents}
            )
        except ValueError as error:
            return fail(str(error))

        corpus = CorpusSearch(
            connection,
            bm25=bm25_params(args),
            expansion=expansion_settings(args),
            rank_constant=args.rank_constant,
        )
        available = list(SearchMode) if corpus.has_embeddings else [SearchMode.LEXICAL]
        modes = [SearchMode(name) for name in dict.fromkeys(args.retriever or available)]
        if args.expand and modes == [SearchMode.SEMANTIC]:
            return fail("--expand does not apply to semantic search")
        if args.authority_weight and SearchMode.HYBRID not in modes:
            return fail("--authority-weight only applies to hybrid search")
        if args.rewrites > 0 and SearchMode.HYBRID not in modes:
            return fail("--rewrites only applies to hybrid search")

        def document_of(chunk_id: int) -> str:
            # A chunk of a document that is no longer unique belongs to no judged page.
            return url_of.get(corpus.page_ids[chunk_id], "")

        results = []
        try:
            retrievers = [(mode.value, corpus.retriever(mode)) for mode in modes]
            if args.authority_weight:
                # Next to plain hybrid search, so that the table shows what the links change.
                weighted = corpus.hybrid_with_authority(args.authority_weight)
                retrievers.append((_WITH_AUTHORITY, weighted))
            if args.rewrites > 0:
                rewritten = MultiQueryRetriever(
                    corpus.hybrid, _rewriter(language_model(args), args.rewrites)
                )
                retrievers.append((_WITH_REWRITES, rewritten))
        except MissingIndexError as error:
            return fail(str(error))
        try:
            for name, chunk_retriever in retrievers:
                retriever = GroupedRetriever(chunk_retriever, document_of)
                results.append((name, retrieval.evaluate_queries(dataset, retriever)))
        except LanguageModelError as error:
            return fail(str(error))
    _report(len(documents), results)
    return 0


def _rewriter(model: LanguageModel, rewrites: int) -> Callable[[str], Sequence[str]]:
    """Build the function that gives the searches to run for a query, asking the model once."""
    known: dict[str, tuple[str, ...]] = {}

    def rewrite(query: str) -> tuple[str, ...]:
        if query not in known:
            known[query] = plan_query(model, query, rewrites=rewrites).queries
            logger.info("searching for: %s", " | ".join(known[query]))
        return known[query]

    return rewrite


def _report(documents: int, results: list[tuple[str, dict[str, retrieval.QueryMetrics]]]) -> None:
    """Print one row of metrics per retriever and, if several, how they differ from the first."""
    compared = len(results) > 1
    baseline = [query.ndcg_at_10 for query in results[0][1].values()]
    print(f"{documents} documents, {len(baseline)} queries")
    print(
        f"{'retriever':<15} {'nDCG@10':>8} {'MAP':>8} {'MRR':>8} {'P@10':>8} {'R@100':>8}"
        + (f" {'p':>8}" if compared else "")
    )
    for number, (name, per_query) in enumerate(results):
        metrics = retrieval.summarize(per_query)
        row = (
            f"{name:<15} {metrics.ndcg_at_10:>8.4f} {metrics.map:>8.4f} {metrics.mrr:>8.4f} "
            f"{metrics.precision_at_10:>8.4f} {metrics.recall_at_100:>8.4f}"
        )
        if number > 0:
            scores = [query.ndcg_at_10 for query in per_query.values()]
            row += f" {paired_randomization_test(scores, baseline):>8.4f}"
        print(row)
    if compared:
        print(
            "p: how likely a difference in nDCG@10 from the first row this large would be "
            "if the two were equally good"
        )
