import sqlite3

import pytest

from digsite.answer import Answerer, AnswerSettings, QueryKind
from digsite.index.analyzer import AnalyzerSettings, Language
from digsite.index.lexical_index_store import LexicalIndexStore
from digsite.index.pipeline import build_index
from digsite.llm import LanguageModelError
from digsite.search.corpus import CorpusSearch, MissingIndexError, SearchMode
from digsite.search.retriever import Hit
from digsite.store import ChunkStore, CrawlStore, DocumentStore
from fakes import ScriptedModel

SITE = "https://example.com"
PAGES = {
    "/caching": (
        "Caching",
        "# Caching\n\nThe cache stores the results of expensive computations.\n\n"
        "## Eviction\n\nWhen the cache is full, the entry used least recently is removed.\n\n"
        "## Expiry\n\nEvery cache entry expires after a configurable time.\n\n"
        "## Keys\n\nEvery cache entry is identified by a key.",
    ),
    "/logging": ("Logging", "# Logging\n\nLogs are rotated daily and kept for thirty days."),
    "/queues": ("Queues", "# Queues\n\nA job that fails is retried three times."),
}

READING = {
    "language": "English",
    "intent": "The person wants to know what is removed from a full cache.",
    "kind": "fact",
    "keywords": "cache eviction",
    "english": "cache eviction",
}
ANSWER = {
    "answerable": True,
    "answer": "The entry used least recently is removed [1].",
}


@pytest.fixture
def corpus(connection: sqlite3.Connection) -> sqlite3.Connection:
    """An indexed corpus of three pages. Lexical search only: no model is needed."""
    store, documents = CrawlStore(connection), DocumentStore(connection)
    for path, (title, text) in PAGES.items():
        page_id = store.save_page(f"{SITE}{path}", 0, b"<p>x</p>")
        documents.save(page_id, title, text, f"hash-{page_id}", page_id)
    build_index(
        documents,
        ChunkStore(connection),
        LexicalIndexStore(connection),
        AnalyzerSettings(Language.ENGLISH),
    )
    return connection


def answerer(
    connection: sqlite3.Connection, model: ScriptedModel, settings: AnswerSettings | None = None
) -> Answerer:
    return Answerer(
        CorpusSearch(connection).lexical,
        ChunkStore(connection),
        DocumentStore(connection),
        model,
        settings,
    )


def test_a_question_is_read_searched_for_and_answered(corpus: sqlite3.Connection) -> None:
    model = ScriptedModel(READING, ANSWER)

    answer = answerer(corpus, model).ask("What happens when the cache is full?")

    assert answer.answered
    assert answer.text == "The entry used least recently is removed [1]."
    assert answer.cited == (1,)
    assert answer.plan.kind is QueryKind.FACT
    assert answer.plan.queries == ("What happens when the cache is full?", "cache eviction")
    (cited,) = answer.cited_sources
    assert cited.url == f"{SITE}/caching"
    assert cited.title == "Caching"
    assert cited.section == "Eviction"
    assert cited.text == "When the cache is full, the entry used least recently is removed."


def test_the_model_answers_from_the_passages_that_were_found(corpus: sqlite3.Connection) -> None:
    model = ScriptedModel(READING, ANSWER)

    answer = answerer(corpus, model).ask("What happens when the cache is full?")

    reading, writing = model.calls
    assert "What happens when the cache is full?" in reading.prompt
    for source in answer.sources:
        assert f"[{source.number}] " in writing.prompt
        assert source.text in writing.prompt
    assert [source.number for source in answer.sources] == list(range(1, len(answer.sources) + 1))
    # The kind of question the model saw in the first call shapes the second one.
    assert "Answer in one or two sentences." in writing.prompt


def test_the_queries_the_model_adds_bring_in_passages_the_question_does_not_find(
    corpus: sqlite3.Connection,
) -> None:
    question = "What happens to a task that goes wrong?"
    reading = {**READING, "keywords": "failed job retried", "english": ""}
    declined = {"answerable": False, "answer": ""}

    plain = answerer(corpus, ScriptedModel(declined), AnswerSettings(rewrites=0)).ask(question)
    helped = answerer(corpus, ScriptedModel(reading, declined)).ask(question)

    # No word of the question is in the page about queues; the added query finds it.
    assert f"{SITE}/queues" not in {source.url for source in plain.sources}
    assert helped.sources[0].url == f"{SITE}/queues"


def test_a_passage_found_by_several_queries_comes_first(corpus: sqlite3.Connection) -> None:
    reading = {**READING, "keywords": "logs rotated", "english": "cache expires"}
    model = ScriptedModel(reading, {"answerable": False, "answer": ""})

    answer = answerer(corpus, model).ask("cache entry expires after a configurable time")

    # Two of the three queries put the passage about expiry first.
    assert answer.sources[0].section == "Expiry"
    assert f"{SITE}/logging" in {source.url for source in answer.sources}


def test_without_rewrites_the_model_is_only_asked_for_the_answer(
    corpus: sqlite3.Connection,
) -> None:
    model = ScriptedModel(ANSWER)

    answer = answerer(corpus, model, AnswerSettings(rewrites=0)).ask("full cache entry removed")

    assert len(model.calls) == 1
    assert answer.plan.queries == ("full cache entry removed",)
    assert answer.answered


def test_the_passages_are_limited_in_number_and_per_document(corpus: sqlite3.Connection) -> None:
    question = "cache entry"
    declined = {"answerable": False, "answer": ""}

    def sources(settings: AnswerSettings) -> list[str]:
        answer = answerer(corpus, ScriptedModel(declined), settings).ask(question)
        return [source.section for source in answer.sources]

    # Four chunks of the caching page mention the cache.
    assert len(sources(AnswerSettings(rewrites=0, per_document=4))) == 4
    assert len(sources(AnswerSettings(rewrites=0, per_document=2))) == 2
    assert len(sources(AnswerSettings(rewrites=0, passages=1, per_document=4))) == 1


def test_a_question_the_corpus_does_not_answer_is_declined(corpus: sqlite3.Connection) -> None:
    model = ScriptedModel(READING, {"answerable": False, "answer": ""})

    answer = answerer(corpus, model).ask("What happens when the cache is full?")

    assert not answer.answered
    assert answer.text == ""
    assert answer.cited == ()
    assert answer.cited_sources == ()
    # What was found is still reported, for whoever wants to look.
    assert answer.sources


def test_when_nothing_is_found_the_model_is_not_asked_to_write(
    corpus: sqlite3.Connection,
) -> None:
    model = ScriptedModel()

    answer = answerer(corpus, model, AnswerSettings(rewrites=0)).ask("zeppelin")

    assert not answer.answered
    assert answer.sources == ()
    assert model.calls == []


def test_a_reading_that_cannot_be_used_does_not_stop_the_answer(
    corpus: sqlite3.Connection,
) -> None:
    model = ScriptedModel("no json here", ANSWER)

    answer = answerer(corpus, model).ask("full cache entry removed")

    assert answer.answered
    assert answer.plan.queries == ("full cache entry removed",)


def test_a_model_that_cannot_be_reached_stops_the_answer(corpus: sqlite3.Connection) -> None:
    model = ScriptedModel(LanguageModelError("cannot reach Ollama"))

    with pytest.raises(LanguageModelError):
        answerer(corpus, model).ask("full cache")


def test_an_answerer_can_be_built_for_a_corpus(corpus: sqlite3.Connection) -> None:
    model = ScriptedModel(ANSWER)

    answerer = Answerer.for_corpus(CorpusSearch(corpus), model, settings=AnswerSettings(rewrites=0))

    # The corpus has no embeddings, so passages are searched for by their words.
    answer = answerer.ask("full cache entry removed")
    assert answer.answered
    assert answer.sources[0].url == f"{SITE}/caching"


def test_an_answerer_needs_the_index_its_search_mode_uses(corpus: sqlite3.Connection) -> None:
    with pytest.raises(MissingIndexError, match="has no embeddings"):
        Answerer.for_corpus(CorpusSearch(corpus), ScriptedModel(), mode=SearchMode.SEMANTIC)


def test_chunks_that_are_gone_are_left_out(corpus: sqlite3.Connection) -> None:
    class StaleRetriever:
        """Returns a chunk that exists and one that was deleted since it was indexed."""

        def search(self, query: str, limit: int = 10) -> list[Hit[int]]:
            return [Hit(999_999, 2.0), Hit(1, 1.0)]

    model = ScriptedModel({"answerable": False, "answer": ""})
    stale = Answerer(
        StaleRetriever(),
        ChunkStore(corpus),
        DocumentStore(corpus),
        model,
        AnswerSettings(rewrites=0),
    )

    answer = stale.ask("anything")

    assert [source.chunk_id for source in answer.sources] == [1]
    assert answer.sources[0].number == 1
