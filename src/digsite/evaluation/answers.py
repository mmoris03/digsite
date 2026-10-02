"""Evaluation of answers against the pages known to answer each question.

What is checked is where an answer points, not what it says: whether the
question was answered at all, whether the answer cites its sources, and
whether those sources are pages judged relevant to the question. Whether the
text is right and complete takes a reader, human or not, and is not measured
here.
"""

from collections.abc import Collection, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from digsite.answer import Answer, QueryKind, QueryPlan, Source


@dataclass(frozen=True, slots=True)
class AnswerJudgement:
    """What one answer did, seen against the pages that answer its question.

    Attributes:
        answered: Whether an answer was given.
        citations: Sources the answer cites.
        relevant_citations: Those among them that are passages of a relevant page.
        relevant_found: Whether a passage of a relevant page was among those
            the model was given. If not, no right answer was possible.
    """

    answered: bool
    citations: int
    relevant_citations: int
    relevant_found: bool


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    """How a set of questions was answered.

    Attributes:
        questions: Questions asked.
        answered: Share of the questions that got an answer.
        supported: Share of the questions whose answer cites a relevant page.
            This is the share of answers a reader can check and find grounded
            in the right place.
        uncited: Share of the questions whose answer cites nothing.
        citation_precision: Of all the citations made, the share that point to
            a relevant page.
        relevant_found: Share of the questions for which a passage of a
            relevant page was given to the model: the ceiling that retrieval
            puts on `supported`.
    """

    questions: int
    answered: float
    supported: float
    uncited: float
    citation_precision: float
    relevant_found: float


def judge_answer(answer: Answer, relevant: Collection[str]) -> AnswerJudgement:
    """Compare an answer with the pages judged relevant to its question.

    Args:
        answer: The answer to judge.
        relevant: URLs of the pages that answer the question. Empty for a
            question that the corpus does not answer.
    """
    cited = answer.cited_sources
    return AnswerJudgement(
        answered=answer.answered,
        citations=len(cited),
        relevant_citations=sum(source.url in relevant for source in cited),
        relevant_found=any(source.url in relevant for source in answer.sources),
    )


def summarize(judgements: Iterable[AnswerJudgement]) -> AnswerMetrics:
    """Aggregate the judgements of a set of questions."""
    judged = list(judgements)
    count = len(judged)
    if count == 0:
        return AnswerMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    citations = sum(judgement.citations for judgement in judged)
    return AnswerMetrics(
        questions=count,
        answered=sum(judgement.answered for judgement in judged) / count,
        supported=sum(judgement.relevant_citations > 0 for judgement in judged) / count,
        uncited=sum(j.answered and j.citations == 0 for j in judged) / count,
        citation_precision=(
            sum(judgement.relevant_citations for judgement in judged) / citations
            if citations
            else 0.0
        ),
        relevant_found=sum(judgement.relevant_found for judgement in judged) / count,
    )


def to_record(answer: Answer) -> dict[str, object]:
    """Turn an answer into plain data that can be written as JSON and read back."""
    return {
        "question": answer.question,
        "answered": answer.answered,
        "text": answer.text,
        "cited": list(answer.cited),
        "kind": answer.plan.kind.value,
        "language": answer.plan.language,
        "intent": answer.plan.intent,
        "queries": list(answer.plan.queries),
        "sources": [asdict(source) for source in answer.sources],
    }


def from_record(record: Mapping[str, Any]) -> Answer:
    """Rebuild an answer from the data `to_record` produced.

    Raises:
        ValueError: If the record does not have that shape.
    """
    try:
        return Answer(
            question=str(record["question"]),
            answered=bool(record["answered"]),
            text=str(record["text"]),
            sources=tuple(Source(**source) for source in record["sources"]),
            cited=tuple(int(number) for number in record["cited"]),
            plan=QueryPlan(
                question=str(record["question"]),
                language=str(record["language"]),
                intent=str(record["intent"]),
                kind=QueryKind(record["kind"]),
                queries=tuple(str(query) for query in record["queries"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Not the record of an answer: {error!r}") from None


def load_questions(path: Path) -> list[str]:
    """Load questions from a file with one per line.

    Blank lines and lines starting with `#` are ignored, and so is whatever
    follows a tab, so a file of labelled queries can be read as plain questions.
    """
    questions = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            question = line.split("\t")[0].strip()
            if question and not line.startswith("#"):
                questions.append(question)
    return questions
