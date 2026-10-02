"""The ask command."""

import argparse
import sys
from contextlib import closing

from digsite.answer import Answer, Answerer, Source
from digsite.cli.common import (
    Subparsers,
    answer_options,
    answer_settings,
    corpus_options,
    corpus_path,
    fail,
    invalid_answer_options,
    language_model,
    missing_corpus,
    search_mode,
)
from digsite.llm import LanguageModelError
from digsite.search.corpus import CorpusSearch, MissingIndexError
from digsite.store import connect


def register(commands: Subparsers) -> None:
    ask = commands.add_parser(
        "ask",
        parents=[corpus_options(), answer_options()],
        help="answer a question from the indexed documents, citing them",
    )
    ask.add_argument("question", nargs="+", help="what to answer")
    ask.add_argument(
        "--show-passages",
        action="store_true",
        help="print every passage the model was given, not only the ones it cites",
    )
    ask.set_defaults(handler=_ask)


def _ask(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    if problem := invalid_answer_options(args):
        return fail(problem)
    with closing(connect(path)) as connection:
        try:
            answerer = Answerer.for_corpus(
                CorpusSearch(connection),
                language_model(args),
                mode=search_mode(args),
                settings=answer_settings(args),
            )
            answer = answerer.ask(" ".join(args.question))
        except (LanguageModelError, MissingIndexError) as error:
            return fail(str(error))
    print(format_answer(answer, show_passages=args.show_passages))
    if answer.answered and not answer.cited:
        print("warning: the answer cites none of its passages", file=sys.stderr)
    return 0


def format_answer(answer: Answer, *, show_passages: bool = False) -> str:
    """Lay out an answer for the terminal: the text, then the sources it cites."""
    lines = []
    if answer.answered:
        lines += [answer.text, ""]
        listed = answer.cited_sources
        heading = "Sources"
    else:
        lines += ["The documents do not answer this question.", ""]
        listed = answer.sources[:3]
        heading = "Closest passages"
    if listed:
        lines.append(f"{heading}:")
        lines += [_reference(source) for source in listed]
    if show_passages:
        lines += ["", "Passages given to the model:"]
        for source in answer.sources:
            lines += [_reference(source), *(f"      {line}" for line in source.text.splitlines())]
    return "\n".join(lines).rstrip()


def _reference(source: Source) -> str:
    where = " > ".join(part for part in (source.title, source.section) if part)
    return f" [{source.number}] {where}\n     {source.url}"
