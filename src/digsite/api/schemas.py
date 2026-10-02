"""What the HTTP interface accepts and returns.

These are the terms of the interface, kept apart from the domain types so that
either can change without silently changing the other.
"""

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from digsite.answer import Answer, Source
from digsite.library import MAX_PAGES, BuildStage, BuildStatus, CollectionSummary, Language
from digsite.search.corpus import SearchMode
from digsite.search.results import DocumentResult

MAX_QUERY_LENGTH = 500
MAX_URL_LENGTH = 2000

type QueryText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_LENGTH)
]


class Status(BaseModel):
    """What the server holds and is doing."""

    collections: int = Field(description="Collections that can be searched")
    building: bool = Field(description="True while a website is being added")
    language_model: str = Field(description="Model that answers questions")


class Collection(BaseModel):
    """A website made searchable."""

    id: str
    title: str
    source: str = Field(description="Where the crawl started; empty if not known")
    documents: int = Field(description="Documents that are searched")
    chunks: int = Field(description="Passages the documents are split into")
    modes: list[SearchMode] = Field(description="Search modes the collection allows")
    default_mode: SearchMode
    index_up_to_date: bool = Field(
        description="False if the documents changed since the index was built"
    )

    @classmethod
    def of(cls, summary: CollectionSummary, default_mode: SearchMode | None) -> Self:
        modes = list(summary.modes)
        return cls(
            id=summary.id,
            title=summary.title,
            source=summary.source,
            documents=summary.documents,
            chunks=summary.chunks,
            modes=modes,
            default_mode=default_mode if default_mode in modes else summary.default_mode,
            index_up_to_date=summary.index_up_to_date,
        )


class Build(BaseModel):
    """A website being added, or that was."""

    id: str
    collection: str = Field(description="Id the collection will have")
    url: str
    stage: BuildStage
    done: int = Field(description="How far the stage has got")
    total: int = Field(description="Out of how much; 0 if not known. For crawling, at most")
    error: str | None = Field(description="Why it failed, if it did")
    submitted: float = Field(description="When it was asked for, in seconds since the epoch")
    finished: float | None

    @classmethod
    def of(cls, status: BuildStatus) -> Self:
        return cls(
            id=status.id,
            collection=status.collection_id,
            url=status.url,
            stage=status.stage,
            done=status.done,
            total=status.total,
            error=status.error,
            submitted=status.submitted,
            finished=status.finished,
        )


class Collections(BaseModel):
    collections: list[Collection]
    builds: list[Build] = Field(description="Websites added since the server started, latest first")


class AddCollectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_URL_LENGTH)
    ] = Field(description="Where to start; only pages under its directory are crawled")
    max_pages: int = Field(default=50, ge=1, le=MAX_PAGES, description="Most pages to crawl")
    language: Language | None = Field(
        default=None,
        description="Language of the pages, for stop words and stemming; none indexes plain words",
    )


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
    collection: str
    query: str
    mode: SearchMode
    results: list[SearchResult]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collection: str = Field(description="Id of the collection to answer from")
    question: QueryText
    mode: SearchMode | None = Field(
        default=None,
        description="How passages are searched for; the collection's default if absent",
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
    collection: str
    question: str
    answered: bool = Field(description="False if the documents do not answer the question")
    answer: str = Field(description="The answer, citing passages like [1]; empty if not answered")
    cited: list[int] = Field(description="Numbers of the passages the answer cites, in order")
    passages: list[Passage] = Field(description="Every passage the model was given")
    queries: list[str] = Field(description="The searches that were run")
    kind: str = Field(description="The kind of answer the question was taken to call for")
    seconds: float = Field(description="Time it took to answer")

    @classmethod
    def of(cls, collection: str, answer: Answer, *, seconds: float) -> Self:
        return cls(
            collection=collection,
            question=answer.question,
            answered=answer.answered,
            answer=answer.text,
            cited=list(answer.cited),
            passages=[Passage.of(source) for source in answer.sources],
            queries=list(answer.plan.queries),
            kind=answer.plan.kind.value,
            seconds=round(seconds, 1),
        )
