import pytest

from digsite.answer.generate import NOT_ANSWERED, Draft, Source, cited_sources, write_answer
from digsite.answer.plan import QueryKind, QueryPlan
from digsite.llm import LanguageModelError
from fakes import ScriptedModel

SOURCES = [
    Source(
        number=1,
        chunk_id=11,
        url="https://example.com/caching",
        title="Caching",
        section="Eviction",
        text="When the cache is full, the entry used least recently is removed.",
    ),
    Source(
        number=2,
        chunk_id=27,
        url="https://example.com/logging",
        title="Logging",
        section="",
        text="Logs are rotated daily and kept for thirty days.",
    ),
]


def plan(
    question: str = "What happens when the cache is full?",
    kind: QueryKind = QueryKind.FACT,
    language: str = "",
) -> QueryPlan:
    return QueryPlan(question, language, "", kind, (question,))


def test_an_answer_comes_back_with_the_sources_it_cites() -> None:
    model = ScriptedModel(
        {
            "answerable": True,
            "answer": " The entry used least recently is removed [1]. ",
        }
    )

    draft = write_answer(model, plan(), SOURCES)

    assert draft == Draft(
        answered=True, text="The entry used least recently is removed [1].", cited=(1,)
    )


def test_the_prompt_shows_the_numbered_passages_and_the_question() -> None:
    model = ScriptedModel({"answerable": False, "answer": ""})

    write_answer(model, plan(), SOURCES)

    (call,) = model.calls
    assert (
        "[1] Caching > Eviction\n"
        "When the cache is full, the entry used least recently is removed.\n\n"
        "[2] Logging\n"
        "Logs are rotated daily and kept for thirty days."
    ) in call.prompt
    assert "Question: What happens when the cache is full?" in call.prompt
    assert "Use only what the passages say" in call.prompt
    assert "Write the answer in the language of the question" in call.prompt
    assert "only say what the passages" in call.system
    assert call.schema is not None
    assert call.schema["required"] == ["answerable", "answer"]


@pytest.mark.parametrize(
    ("kind", "instruction"),
    [
        (QueryKind.FACT, "Answer in one or two sentences."),
        (QueryKind.HOW_TO, "Give the steps in order."),
        (QueryKind.EXPLANATION, "Explain it in one or two short paragraphs."),
        (QueryKind.LOOKUP, "which passage covers it"),
    ],
)
def test_the_kind_of_question_shapes_the_answer_asked_for(
    kind: QueryKind, instruction: str
) -> None:
    model = ScriptedModel({"answerable": False, "answer": ""})

    write_answer(model, plan(kind=kind), SOURCES)

    assert instruction in model.calls[0].prompt


def test_the_language_of_the_question_is_named_when_it_is_known() -> None:
    model = ScriptedModel({"answerable": False, "answer": ""})

    write_answer(model, plan("¿Qué pasa si la caché se llena?", language="Spanish"), SOURCES)

    assert "Write the answer in Spanish, even if the passages" in model.calls[0].prompt


@pytest.mark.parametrize(
    "reply",
    [
        {"answerable": False, "answer": ""},
        {"answerable": False, "answer": "I think so, but they do not say."},
        {"answerable": True, "answer": "   "},
        {"answerable": "yes", "answer": "It is removed [1]."},
        {"answer": "It is removed [1]."},
        {"answerable": True, "answer": None},
    ],
)
def test_anything_short_of_a_clear_answer_is_no_answer(reply: object) -> None:
    assert write_answer(ScriptedModel(reply), plan(), SOURCES) == NOT_ANSWERED


def test_a_reply_that_is_not_json_is_no_answer(caplog: pytest.LogCaptureFixture) -> None:
    assert write_answer(ScriptedModel("The entry is removed."), plan(), SOURCES) == NOT_ANSWERED
    assert "could not read the model's answer" in caplog.text


def test_without_passages_the_model_is_not_asked() -> None:
    model = ScriptedModel()

    assert write_answer(model, plan(), []) == NOT_ANSWERED
    assert model.calls == []


def test_an_answer_that_cites_nothing_is_returned_as_such() -> None:
    model = ScriptedModel({"answerable": True, "answer": "The entry is removed."})

    draft = write_answer(model, plan(), SOURCES)

    assert draft.answered
    assert draft.cited == ()


def test_a_model_that_cannot_be_reached_is_not_hidden() -> None:
    model = ScriptedModel(LanguageModelError("cannot reach Ollama"))

    with pytest.raises(LanguageModelError):
        write_answer(model, plan(), SOURCES)


def test_citations_are_listed_once_in_order_of_first_mention() -> None:
    text = "Logs rotate daily [2]. Entries are evicted [1], the oldest first [2][1]."

    assert cited_sources(text, available=2) == (2, 1)


@pytest.mark.parametrize(
    ("text", "cited"),
    [
        ("Both say so [1][2].", (1, 2)),
        ("Both say so [1, 2].", (1, 2)),
        ("Both say so [2,1].", (2, 1)),
        ("It is removed [1] .", (1,)),
        ("(see [2])", (2,)),
        ("No citation here.", ()),
    ],
)
def test_the_ways_a_citation_can_be_written(text: str, cited: tuple[int, ...]) -> None:
    assert cited_sources(text, available=2) == cited


def test_citations_of_sources_that_do_not_exist_are_ignored() -> None:
    assert cited_sources("As stated [3], and also [0] and [12], but really [2].", 2) == (2,)


@pytest.mark.parametrize(
    "text",
    [
        "Use `sys.argv[1]` for the first argument.",
        "Take items[1] and matrix[1][2].",
        "Call f(x)[1] to get the second value.",
        "The slice a[1:2] is shorter.",
        "Use [1:] to skip the first.",
    ],
)
def test_indexing_in_code_is_not_a_citation(text: str) -> None:
    assert cited_sources(text, available=2) == ()


def test_a_citation_next_to_code_is_still_found() -> None:
    assert cited_sources("Use `sys.argv[1]` for the first argument [2].", 2) == (2,)
