"""What the HTTP interface accepts and returns.

These are the terms of the interface, kept apart from the domain types so that
either can change without silently changing the other.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from digsite.answer import Answer, Source
from digsite.search.corpus import SearchMode
from digsite.search.results import DocumentResult

MAX_QUERY_LENGTH = 500

type QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)
]


class Status(BaseModel):
    """What the corpus holds and what can be done with it."""

    documents: int = Field(description="Documents that are searched")
    chunks: int = Field(description="Passages the documents are split into")
    index_up_to_date: bool = Field(
        description="False if the documents changed since the index was built"
    )
    embedding_model: str | None = Field(description="Model that embedded the passages, if any")
    modes: list[SearchMode] = Field(description="Search modes the corpus allows")
    default_mode: SearchMode
    language_model: str = Field(description="Model that answers questions")


class SearchResult(BaseModel):
    """A document, shown by the passage that matched best."""

    url: str
    title: str
    section: str = Field(description="Headings the passage sits under, below the title")
    passage: str
    score: float = Field(description="Score of the passage; its scale depends on the mode")

    @classmethod
    def of(cls, result: DocumentResult) -> Self:
        return cls(
            url=result.url,
            title=result.title,
            section=result.section,
            passage=result.passage,
            score=result.score,
        )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    results: list[SearchResult]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: QueryText
    mode: SearchMode | None = Field(
        default=None, description="How passages are searched for; the server's default if absent"
    )


class Passage(BaseModel):
    """A passage the language model was given to answer from."""

    number: int = Field(description="How the answer cites it, as in [1]")
    url: str
    title: str
    section: str
    text: str

    @classmethod
    def of(cls, source: Source) -> Self:
        return cls(
            number=source.number,
            url=source.url,
            title=source.title,
            section=source.section,
            text=source.text,
        )


class AskResponse(BaseModel):
    question: str
    answered: bool = Field(description="False if the documents do not answer the question")
    answer: str = Field(description="The answer, citing passages like [1]; empty if not answered")
    cited: list[int] = Field(description="Numbers of the passages the answer cites, in order")
    passages: list[Passage] = Field(description="Every passage the model was given")
    queries: list[str] = Field(description="The searches that were run")
    kind: str = Field(description="The kind of answer the question was taken to call for")
    seconds: float = Field(description="Time it took to answer")

    @classmethod
    def of(cls, answer: Answer, *, seconds: float) -> Self:
        return cls(
            question=answer.question,
            answered=answer.answered,
            answer=answer.text,
            cited=list(answer.cited),
            passages=[Passage.of(source) for source in answer.sources],
            queries=list(answer.plan.queries),
            kind=answer.plan.kind.value,
            seconds=round(seconds, 1),
        )
