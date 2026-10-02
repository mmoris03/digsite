"""Understanding a question before searching for its answer.

People do not ask the way documentation is written. A language model reads
the question and says what is being asked, what kind of answer it calls for,
and how else to search for it.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

from digsite.llm import LanguageModel, Schema
from digsite.llm.replies import parse_json_object

logger = logging.getLogger(__name__)

DEFAULT_REWRITES = 2


class QueryKind(StrEnum):
    """What kind of answer a question calls for."""

    FACT = "fact"  # a short fact or a definition
    HOW_TO = "how_to"  # how to do something
    EXPLANATION = "explanation"  # why or how something works, or how things differ
    LOOKUP = "lookup"  # a particular page or section, not an answer


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """What is being asked and how to search for it.

    Attributes:
        question: The question as typed.
        language: The language the question is written in, by its English
            name. Empty if the question was not analyzed.
        intent: What the person wants to know, in one sentence. Empty if the
            question was not analyzed.
        kind: The kind of answer it calls for.
        queries: The searches to run, the question itself first, none repeated.
    """

    question: str
    language: str
    intent: str
    kind: QueryKind
    queries: tuple[str, ...]


_SYSTEM = (
    "You prepare searches for a search engine over technical documentation. "
    "You never answer the question yourself."
)

_PROMPT = """Question: {question}

Reply with a JSON object with these fields, in this order:
- "language": the language the question is written in, in English, for example "Spanish".
- "intent": one sentence, in English, saying what the person wants to know.
- "kind": "fact" if they want a short fact or a definition; "how_to" if they want to know how \
to do something; "explanation" if they want to understand why or how something works, or how \
two things differ; "lookup" if they are looking for a particular page or section and not for \
an answer.
- "keywords": the question as a search query in its own language: only the terms that the \
documentation would use, without question words.
- "english": the question as a search query in English: only the terms that English \
documentation would use. Part of the documentation is only in English."""

_SCHEMA: Schema = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "intent": {"type": "string"},
        "kind": {"type": "string", "enum": [kind.value for kind in QueryKind]},
        "keywords": {"type": "string"},
        "english": {"type": "string"},
    },
    "required": ["language", "intent", "kind", "keywords", "english"],
}

# The searches a reading of the question adds, in the order they are used.
_REWRITES = ("keywords", "english")


def plain_plan(question: str) -> QueryPlan:
    """The plan for a question nobody analyzed: search for it as typed."""
    return QueryPlan(question, "", "", QueryKind.EXPLANATION, (question,))


def plan_query(
    model: LanguageModel, question: str, *, rewrites: int = DEFAULT_REWRITES
) -> QueryPlan:
    """Ask a language model what a question is after and how else to search for it.

    The question itself is always searched for, whatever the model adds. If
    the model's reply cannot be used, the plan is the plain one: a worse
    search is better than no answer.

    Args:
        model: The model to ask.
        question: The question as typed.
        rewrites: How many of the searches the model proposes are added to the
            question: its key terms in the question's language, then in English.

    Raises:
        LanguageModelError: If the model could not be reached.
    """
    if rewrites < 1:
        return plain_plan(question)
    reply = model.generate(_PROMPT.format(question=question), system=_SYSTEM, schema=_SCHEMA)
    try:
        fields = parse_json_object(reply)
        kind_name = fields.get("kind")
        if not isinstance(kind_name, str):
            raise ValueError("'kind' is missing")
        kind = QueryKind(kind_name)
    except ValueError as error:
        logger.warning("could not use the model's reading of the question: %s", error)
        return plain_plan(question)

    queries = [question]
    seen = {_normalize(question)}
    for name in _REWRITES[:rewrites]:
        query = fields.get(name)
        if not isinstance(query, str):
            continue
        query = " ".join(query.split())
        if query and _normalize(query) not in seen:
            seen.add(_normalize(query))
            queries.append(query)
    return QueryPlan(
        question=question,
        language=_text(fields.get("language")),
        intent=_text(fields.get("intent")),
        kind=kind,
        queries=tuple(queries),
    )


def _text(value: object) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _normalize(query: str) -> str:
    """Reduce a query to what makes it a different search: its words, whatever their case."""
    return " ".join(query.casefold().split()).strip(" ?¿!¡.")
