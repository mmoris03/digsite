"""Options and helpers shared by several commands."""

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from digsite.answer import AnswerSettings
from digsite.embedding import DEFAULT_MODEL
from digsite.index.analyzer import AnalyzerSettings, Language
from digsite.library import DEFAULT_COLLECTION, is_collection_id
from digsite.llm import DEFAULT_MODEL as DEFAULT_LANGUAGE_MODEL
from digsite.llm import DEFAULT_URL as DEFAULT_LANGUAGE_MODEL_URL
from digsite.llm import LanguageModel, create_language_model
from digsite.search.bm25 import Bm25Params, Bm25Variant
from digsite.search.corpus import SearchMode
from digsite.search.expansion import ExpansionSettings
from digsite.search.fusion import DEFAULT_RANK_CONSTANT

CORPUS_FILENAME = f"{DEFAULT_COLLECTION}.db"

# argparse's type for the object returned by add_subparsers().
type Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]


def verbosity_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    return parser


def library_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, parents=[verbosity_options()])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="directory that holds the collections, one file each (default: data)",
    )
    return parser


def _collection_id(text: str) -> str:
    if not is_collection_id(text):
        raise argparse.ArgumentTypeError(
            "use lowercase letters, digits and hyphens, starting with a letter or digit"
        )
    return text


def corpus_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False, parents=[library_options()])
    parser.add_argument(
        "--collection",
        type=_collection_id,
        default=DEFAULT_COLLECTION,
        metavar="ID",
        help=f"collection to work on, stored as <data-dir>/<ID>.db (default: {DEFAULT_COLLECTION})",
    )
    return parser


def analysis_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--language",
        choices=[language.value for language in Language],
        help="language of the stop-word list and the stemmer (default: none, plain words)",
    )
    parser.add_argument(
        "--keep-stopwords", action="store_true", help="do not remove the language's stop words"
    )
    parser.add_argument("--no-stemming", action="store_true", help="do not reduce words to stems")
    return parser


def embedding_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="sentence-transformers model to embed with, or 'hashing' for the model-free "
        f"embedder (default: {DEFAULT_MODEL})",
    )
    return parser


def ranking_options() -> argparse.ArgumentParser:
    defaults = Bm25Params()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in Bm25Variant],
        default=defaults.variant.value,
        help=f"BM25 formulation (default: {defaults.variant.value})",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=defaults.k1,
        help=f"term-frequency saturation (default: {defaults.k1})",
    )
    parser.add_argument(
        "--b", type=float, default=defaults.b, help=f"length normalisation (default: {defaults.b})"
    )
    return parser


def expansion_options() -> argparse.ArgumentParser:
    defaults = ExpansionSettings()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--expand",
        action="store_true",
        help="add to the query the terms that characterise its best results",
    )
    parser.add_argument(
        "--feedback-documents",
        type=int,
        default=defaults.feedback_documents,
        help=f"with --expand, results taken as relevant (default: {defaults.feedback_documents})",
    )
    parser.add_argument(
        "--expansion-terms",
        type=int,
        default=defaults.terms,
        help=f"with --expand, terms added to the query (default: {defaults.terms})",
    )
    parser.add_argument(
        "--expansion-weight",
        type=float,
        default=defaults.weight,
        help=f"with --expand, weight of each added term (default: {defaults.weight})",
    )
    return parser


def _non_negative[N: (int, float)](kind: type[N]) -> Callable[[str], N]:
    """Build an argparse type that reads a number and rejects negative ones."""

    def parse(text: str) -> N:
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if value < 0:
            raise argparse.ArgumentTypeError("must not be negative")
        return value

    return parse


def fusion_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--rank-constant",
        type=_non_negative(int),
        default=DEFAULT_RANK_CONSTANT,
        help="in hybrid search, how little the very first positions of each ranking stand out "
        f"(default: {DEFAULT_RANK_CONSTANT})",
    )
    return parser


def authority_options() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--authority-weight",
        type=_non_negative(float),
        default=0.0,
        help="in hybrid search, how much the links a document receives count, a retriever "
        "counting 1 (default: 0, links are ignored)",
    )
    return parser


def expansion_settings(args: argparse.Namespace) -> ExpansionSettings | None:
    if not args.expand:
        return None
    return ExpansionSettings(
        feedback_documents=args.feedback_documents,
        terms=args.expansion_terms,
        weight=args.expansion_weight,
    )


def analyzer_settings(args: argparse.Namespace) -> AnalyzerSettings:
    return AnalyzerSettings(
        language=Language(args.language) if args.language else None,
        remove_stopwords=not args.keep_stopwords,
        stem=not args.no_stemming,
    )


def bm25_params(args: argparse.Namespace) -> Bm25Params:
    return Bm25Params(variant=Bm25Variant(args.variant), k1=args.k1, b=args.b)


def corpus_path(args: argparse.Namespace) -> Path:
    path: Path = args.data_dir / f"{args.collection}.db"
    return path


def fail(message: str, exit_code: int = 1) -> int:
    """Print an error and return the exit code to end the command with."""
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def missing_corpus(path: Path) -> bool:
    """Report a missing corpus file; tell whether it is missing."""
    if path.exists():
        return False
    fail(f"{path} does not exist; run 'digsite crawl' first")
    return True


def language_model_options() -> argparse.ArgumentParser:
    # Environment variables set the defaults once for every command, as a
    # container does; an option on the command line still wins.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--llm",
        default=os.environ.get("DIGSITE_LLM") or DEFAULT_LANGUAGE_MODEL,
        metavar="NAME",
        help="language model, as 'ollama list' shows it "
        f"(default: $DIGSITE_LLM, or {DEFAULT_LANGUAGE_MODEL})",
    )
    parser.add_argument(
        "--llm-url",
        default=os.environ.get("DIGSITE_LLM_URL") or DEFAULT_LANGUAGE_MODEL_URL,
        metavar="URL",
        help=f"where Ollama listens (default: $DIGSITE_LLM_URL, or {DEFAULT_LANGUAGE_MODEL_URL})",
    )
    return parser


def answer_options() -> argparse.ArgumentParser:
    defaults = AnswerSettings()
    parser = argparse.ArgumentParser(add_help=False, parents=[language_model_options()])
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in SearchMode],
        help="how passages are searched for "
        "(default: hybrid if the corpus has embeddings, lexical otherwise)",
    )
    parser.add_argument(
        "--passages",
        type=int,
        default=defaults.passages,
        help=f"passages given to the model (default: {defaults.passages})",
    )
    parser.add_argument(
        "--rewrites",
        type=int,
        default=defaults.rewrites,
        help="searches the model adds to the question; 0 searches only for the question "
        f"as typed (default: {defaults.rewrites})",
    )
    return parser


def language_model(args: argparse.Namespace) -> LanguageModel:
    return create_language_model(args.llm, url=args.llm_url)


def search_mode(args: argparse.Namespace) -> SearchMode | None:
    return SearchMode(args.mode) if args.mode else None


def answer_settings(args: argparse.Namespace) -> AnswerSettings:
    return AnswerSettings(rewrites=max(args.rewrites, 0), passages=args.passages)


def invalid_answer_options(args: argparse.Namespace) -> str | None:
    """Say what is wrong with the answer options, or None if nothing is."""
    if args.passages < 1:
        return "--passages must be at least 1"
    return None
