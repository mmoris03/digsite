"""Writing an answer from passages, with citations, or declining to."""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from digsite.answer.plan import QueryKind, QueryPlan
from digsite.llm import LanguageModel, Schema
from digsite.llm.replies import parse_json_object

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Source:
    """A passage given to the model to answer from.

    Attributes:
        number: How the answer cites it, from 1.
        chunk_id: The chunk the passage is.
        url: Address of the document it belongs to.
        title: Title of that document.
        section: Headings the passage sits under, below the title.
        text: The passage.
    """

    number: int
    chunk_id: int
    url: str
    title: str
    section: str
    text: str


@dataclass(frozen=True, slots=True)
class Draft:
    """What the model wrote.

    Attributes:
        answered: Whether the model found the answer in the passages.
        text: The answer, with citations like [1]. Empty if not answered.
        cited: Numbers of the sources the text cites, in order of first mention.
    """

    answered: bool
    text: str
    cited: tuple[int, ...]


NOT_ANSWERED = Draft(answered=False, text="", cited=())

_SCHEMA: Schema = {
    "type": "object",
    "properties": {"answerable": {"type": "boolean"}, "answer": {"type": "string"}},
    "required": ["answerable", "answer"],
}

_SYSTEM = (
    "You answer questions about technical documentation. You only say what the passages you "
    "are given say, and you say where each thing comes from."
)

_STYLE = {
    QueryKind.FACT: "Answer in one or two sentences.",
    QueryKind.HOW_TO: "Give the steps in order. Include the commands or code the passages show.",
    QueryKind.EXPLANATION: "Explain it in one or two short paragraphs.",
    QueryKind.LOOKUP: "Say in one or two sentences which passage covers it and what it contains.",
}

_PROMPT = """Passages:

{passages}

Question: {question}

Rules:
- Use only what the passages say. Add nothing from your own knowledge.
- After each statement, cite the passage it comes from by its number in square brackets, \
like [1], or [1][2] if it comes from two.
- {style}
- Write the answer in {language}, even if the passages are in another language.
- If the passages do not contain the answer, do not guess: set "answerable" to false and \
leave "answer" empty.

Reply with a JSON object with two fields: "answerable", true or false, and "answer", the \
answer as text."""

# One or more citations in a row, like [1] or [1][3] or [1, 3]. A bracket right
# after a name or a closing bracket is code, as in `items[0]` or `f(x)[1]`.
_CITATIONS_RE = re.compile(r"(?<![\w)\]])(?:\[\d+(?:\s*,\s*\d+)*\])+")
_NUMBER_RE = re.compile(r"\d+")


def write_answer(model: LanguageModel, plan: QueryPlan, sources: Sequence[Source]) -> Draft:
    """Ask a language model to answer a question from the given passages.

    The model is told to answer only from the passages and to cite them, and
    to decline when they do not contain the answer. Without passages it is
    not asked at all: there is nothing to answer from.

    Args:
        model: The model to ask.
        plan: The question and what kind of answer it calls for.
        sources: The passages, numbered as the answer should cite them.

    Raises:
        LanguageModelError: If the model could not be reached.
    """
    if not sources:
        return NOT_ANSWERED
    prompt = _PROMPT.format(
        passages="\n\n".join(_render(source) for source in sources),
        question=plan.question,
        style=_STYLE[plan.kind],
        # Named, a language is followed; "the language of the question" often is not.
        language=plan.language or "the language of the question",
    )
    reply = model.generate(prompt, system=_SYSTEM, schema=_SCHEMA)
    try:
        fields = parse_json_object(reply)
    except ValueError as error:
        logger.warning("could not read the model's answer: %s", error)
        return NOT_ANSWERED

    text = fields.get("answer")
    if fields.get("answerable") is not True or not isinstance(text, str) or not text.strip():
        return NOT_ANSWERED
    text = text.strip()
    return Draft(answered=True, text=text, cited=cited_sources(text, len(sources)))


def cited_sources(text: str, available: int) -> tuple[int, ...]:
    """Find which sources a text cites.

    Args:
        text: An answer with citations like [1] or [2][5].
        available: How many sources there were. A citation of any other number
            refers to nothing and is ignored.

    Returns:
        The numbers cited, each once, in order of first mention.
    """
    cited: dict[int, None] = {}
    for run in _CITATIONS_RE.finditer(text):
        for digits in _NUMBER_RE.findall(run.group()):
            number = int(digits)
            if 1 <= number <= available:
                cited.setdefault(number)
    return tuple(cited)


def _render(source: Source) -> str:
    heading = " > ".join(part for part in (source.title, source.section) if part)
    return (
        f"[{source.number}] {heading}\n{source.text}"
        if heading
        else (f"[{source.number}]\n{source.text}")
    )
