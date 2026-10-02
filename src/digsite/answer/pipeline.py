"""Answering a question from the corpus: understand it, search, select passages, write."""

import logging
import time
from dataclasses import dataclass
from typing import Self

from digsite.answer.generate import Source, write_answer
from digsite.answer.plan import DEFAULT_REWRITES, QueryPlan, plain_plan, plan_query
from digsite.llm import LanguageModel
from digsite.search.corpus import CorpusSearch, SearchMode
from digsite.search.grouping import limit_per_group
from digsite.search.multi_query import search_queries
from digsite.search.results import section_of
from digsite.search.retriever import Retriever
from digsite.store import ChunkStore, DocumentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnswerSettings:
    """How a question is answered.

    Attributes:
        rewrites: Searches a language model adds to the question. 0 searches
            only for the question as typed, and saves a call to the model.
        depth: Chunks taken from the results of each search.
        passages: Passages given to the model to answer from.
        per_document: Most passages of one document among them.
    """

    rewrites: int = DEFAULT_REWRITES
    depth: int = 30
    passages: int = 6
    per_document: int = 3


@dataclass(frozen=True, slots=True)
class Answer:
    """The answer to a question, and what it was written from.

    Attributes:
        question: The question as typed.
        answered: Whether the passages contained the answer. If not, `text` is empty.
        text: The answer, citing its sources by number, like [1].
        sources: The passages the model was given, by their number.
        cited: Numbers of the sources the answer cites, in order of first mention.
        plan: How the question was read and searched for.
    """

    question: str
    answered: bool
    text: str
    sources: tuple[Source, ...]
    cited: tuple[int, ...]
    plan: QueryPlan

    @property
    def cited_sources(self) -> tuple[Source, ...]:
        """The sources the answer cites, in order of first mention."""
        by_number = {source.number: source for source in self.sources}
        return tuple(by_number[number] for number in self.cited)


class Answerer:
    """Answers questions from an indexed corpus with a language model.

    The model is used twice: to read the question and propose other ways of
    searching for it, and to write the answer from the passages found. The
    searching in between is the same as `digsite search` does, once per query,
    with the rankings fused by reciprocal rank.

    Args:
        retriever: Searches the chunks of the corpus.
        chunks: Where the chunks are read from.
        documents: Where their documents are read from.
        model: The language model.
        settings: How questions are answered.
    """

    def __init__(
        self,
        retriever: Retriever[int],
        chunks: ChunkStore,
        documents: DocumentStore,
        model: LanguageModel,
        settings: AnswerSettings | None = None,
    ) -> None:
        self._retriever = retriever
        self._chunks = chunks
        self._documents = documents
        self._model = model
        self._settings = settings or AnswerSettings()

    @classmethod
    def for_corpus(
        cls,
        corpus: CorpusSearch,
        model: LanguageModel,
        *,
        mode: SearchMode | None = None,
        settings: AnswerSettings | None = None,
    ) -> Self:
        """Build what answers questions from an indexed corpus.

        Args:
            corpus: The corpus. Its retrievers are shared, not loaded again.
            model: The language model.
            mode: How passages are searched for. By default, hybrid search if
                the corpus has embeddings, lexical search otherwise.
            settings: How questions are answered.

        Raises:
            MissingIndexError: If the corpus was not indexed for the search mode.
        """
        return cls(
            corpus.retriever(mode or corpus.default_mode),
            corpus.chunks,
            corpus.documents,
            model,
            settings,
        )

    def ask(self, question: str) -> Answer:
        """Answer a question, or decline if the corpus does not contain the answer.

        Raises:
            LanguageModelError: If the model could not be reached.
        """
        settings = self._settings
        started = time.perf_counter()
        if settings.rewrites > 0:
            logger.info("reading the question with %s", self._model.name)
            plan = plan_query(self._model, question, rewrites=settings.rewrites)
        else:
            plan = plain_plan(question)
        planned = time.perf_counter()

        logger.info("searching for: %s", " | ".join(plan.queries))
        sources = self.find_sources(plan)
        searched = time.perf_counter()

        logger.info("writing the answer from %d passages", len(sources))
        draft = write_answer(self._model, plan, sources)
        logger.info(
            "done: %.1f s reading the question, %.1f s searching, %.1f s writing",
            planned - started,
            searched - planned,
            time.perf_counter() - searched,
        )
        return Answer(
            question=question,
            answered=draft.answered,
            text=draft.text,
            sources=tuple(sources),
            cited=draft.cited,
            plan=plan,
        )

    def find_sources(self, plan: QueryPlan) -> list[Source]:
        """Search for every query of a plan and select the passages to answer from."""
        settings = self._settings
        fused = [hit.key for hit in search_queries(self._retriever, plan.queries, settings.depth)]

        chunks = {key: chunk for key in fused if (chunk := self._chunks.chunk(key)) is not None}
        chosen = limit_per_group(
            (key for key in fused if key in chunks),
            lambda key: chunks[key].page_id,
            settings.passages,
            settings.per_document,
        )
        sources: list[Source] = []
        for chunk_id in chosen:
            chunk = chunks[chunk_id]
            document = self._documents.document(chunk.page_id)
            if document is None:
                continue
            sources.append(
                Source(
                    number=len(sources) + 1,
                    chunk_id=chunk.id,
                    url=document.url,
                    title=document.title,
                    section=section_of(chunk.context, document.title),
                    text=chunk.text,
                )
            )
        return sources
