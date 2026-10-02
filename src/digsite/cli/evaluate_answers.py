"""The command that scores answers against labelled questions."""

import argparse
import json
import logging
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TextIO

from digsite.answer import Answer, Answerer
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
from digsite.evaluation import answers, retrieval
from digsite.llm import LanguageModelError
from digsite.search.corpus import CorpusSearch, MissingIndexError
from digsite.store import DocumentStore, connect

logger = logging.getLogger(__name__)


def register(evaluations: Subparsers) -> None:
    parser = evaluations.add_parser(
        "answers",
        parents=[corpus_options(), answer_options()],
        help="score the answers to labelled questions by the pages they cite",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        required=True,
        help="file with one question per line, followed by the URLs of the pages that "
        "answer it, separated by tabs",
    )
    parser.add_argument(
        "--unanswerable",
        type=Path,
        help="file with one question per line that the corpus does not answer",
    )
    parser.add_argument(
        "--limit", type=int, help="ask only the first N questions of each file (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="file to write every answer to, one JSON object per line. If it exists, the "
        "questions it already answers are not asked again, so a run can be resumed.",
    )
    parser.set_defaults(handler=_answers)


def _answers(args: argparse.Namespace) -> int:
    path = corpus_path(args)
    if missing_corpus(path):
        return 1
    for file in (args.queries, args.unanswerable):
        if file is not None and not file.exists():
            return fail(f"{file} does not exist")
    if problem := invalid_answer_options(args):
        return fail(problem)

    settings = {
        "llm": args.llm,
        "mode": args.mode,
        "passages": args.passages,
        "rewrites": args.rewrites,
    }
    with closing(connect(path)) as connection:
        documents = DocumentStore(connection).documents()
        try:
            dataset = retrieval.load_labelled_queries(
                args.queries, {document.url: document.text for document in documents}
            )
            saved = _read_saved(args.output, settings)
            answerer = Answerer.for_corpus(
                CorpusSearch(connection),
                language_model(args),
                mode=search_mode(args),
                settings=answer_settings(args),
            )
        except (ValueError, MissingIndexError) as error:
            return fail(str(error))

        answerable = list(dataset.queries)[: args.limit]
        unanswerable = (
            answers.load_questions(args.unanswerable)[: args.limit] if args.unanswerable else []
        )
        questions = [*answerable, *unanswerable]
        answered: dict[str, Answer] = {}
        with _output(args.output, settings, new=not saved) as output:
            for number, question in enumerate(questions, start=1):
                if question in saved:
                    answered[question] = saved[question]
                    continue
                logger.info("question %d of %d: %s", number, len(questions), question)
                try:
                    answer = answerer.ask(question)
                except LanguageModelError as error:
                    return fail(str(error))
                answered[question] = answer
                if output is not None:
                    output.write(json.dumps(answers.to_record(answer), ensure_ascii=False) + "\n")
                    output.flush()

    known = answers.summarize(
        answers.judge_answer(answered[question], dataset.judgements[question])
        for question in answerable
    )
    print(f"{known.questions} questions that the corpus answers")
    for name, value in (
        ("answered", known.answered),
        ("cites a relevant page", known.supported),
        ("cites nothing", known.uncited),
        ("citation precision", known.citation_precision),
        ("relevant page found", known.relevant_found),
    ):
        print(f"  {name:<22} {value:.3f}")
    if unanswerable:
        declined = sum(not answered[question].answered for question in unanswerable)
        print(f"\n{len(unanswerable)} questions that the corpus does not answer")
        print(f"  {'declined':<22} {declined / len(unanswerable):.3f}")
    return 0


def _read_saved(path: Path | None, settings: dict[str, object]) -> dict[str, Answer]:
    """Read the answers of an earlier run, by question.

    Raises:
        ValueError: If the file is not the output of this command, or that run
            used other settings: its answers cannot be mixed with new ones.
    """
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        lines = [line for line in file if line.strip()]
    if not lines:
        return {}
    try:
        used = json.loads(lines[0])["settings"]
        saved = [answers.from_record(json.loads(line)) for line in lines[1:]]
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f"{path} is not the output of an earlier run: {error!r}") from None
    if used != settings:
        raise ValueError(
            f"{path} holds answers obtained with other settings ({used}); "
            "delete it or choose another file"
        )
    return {answer.question: answer for answer in saved}


@contextmanager
def _output(
    path: Path | None, settings: dict[str, object], *, new: bool
) -> Iterator[TextIO | None]:
    """Open the output file, or nothing if there is none.

    A new file starts with a line that records the settings of the run; an
    existing one is appended to.
    """
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if new else "a", encoding="utf-8") as file:
        if new:
            file.write(json.dumps({"settings": settings}) + "\n")
        yield file
