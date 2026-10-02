import pytest

from digsite.answer.plan import QueryKind, QueryPlan, plain_plan, plan_query
from digsite.llm import LanguageModelError
from fakes import ScriptedModel

QUESTION = "¿Cómo leo los argumentos de la línea de comandos?"
READING = {
    "language": "Spanish",
    "intent": "The person wants to read command-line arguments in a script.",
    "kind": "how_to",
    "keywords": "leer argumentos línea de comandos",
    "english": "read command-line arguments",
}


def test_the_plan_is_the_question_plus_what_the_model_adds() -> None:
    plan = plan_query(ScriptedModel(READING), QUESTION)

    assert plan == QueryPlan(
        question=QUESTION,
        language="Spanish",
        intent="The person wants to read command-line arguments in a script.",
        kind=QueryKind.HOW_TO,
        queries=(QUESTION, "leer argumentos línea de comandos", "read command-line arguments"),
    )


def test_the_model_is_asked_once_for_json_about_the_question() -> None:
    model = ScriptedModel(READING)

    plan_query(model, QUESTION)

    (call,) = model.calls
    assert f"Question: {QUESTION}" in call.prompt
    assert "never answer the question" in call.system
    assert call.schema is not None
    assert call.schema["required"] == ["language", "intent", "kind", "keywords", "english"]
    properties = call.schema["properties"]
    assert isinstance(properties, dict)
    assert properties["kind"]["enum"] == ["fact", "how_to", "explanation", "lookup"]


def test_with_one_rewrite_only_the_key_terms_are_added() -> None:
    plan = plan_query(ScriptedModel(READING), QUESTION, rewrites=1)

    assert plan.queries == (QUESTION, "leer argumentos línea de comandos")
    assert plan.language == "Spanish"


def test_a_rewrite_that_repeats_the_question_or_another_is_dropped() -> None:
    reading = {
        **READING,
        "keywords": "  cómo leo los argumentos de la línea de comandos ",
        "english": "COMMAND-LINE arguments",
    }
    same_twice = {
        **READING,
        "keywords": "command-line arguments",
        "english": "Command-Line Arguments",
    }

    assert plan_query(ScriptedModel(reading), QUESTION).queries == (
        QUESTION,
        "COMMAND-LINE arguments",
    )
    assert plan_query(ScriptedModel(same_twice), QUESTION).queries == (
        QUESTION,
        "command-line arguments",
    )


def test_empty_and_malformed_rewrites_are_dropped() -> None:
    reading = {**READING, "keywords": "   ", "english": 7}

    assert plan_query(ScriptedModel(reading), QUESTION).queries == (QUESTION,)


def test_rewrites_lose_their_line_breaks_and_extra_spaces() -> None:
    reading = {**READING, "english": "read   command-line\narguments"}

    assert plan_query(ScriptedModel(reading), QUESTION).queries[-1] == (
        "read command-line arguments"
    )


def test_a_question_in_english_may_get_the_same_search_twice() -> None:
    reading = {
        **READING,
        "language": "English",
        "keywords": "profile code",
        "english": "profile code",
    }

    plan = plan_query(ScriptedModel(reading), "how do I profile code")

    assert plan.queries == ("how do I profile code", "profile code")
    assert plan.language == "English"


@pytest.mark.parametrize(
    "reply",
    [
        "I would search for argparse.",
        {**READING, "kind": "poem"},
        {"intent": "x", "keywords": "argparse", "english": "argparse"},
        ["not", "an", "object"],
    ],
)
def test_a_reply_that_cannot_be_used_falls_back_to_the_question_as_typed(
    reply: object, caplog: pytest.LogCaptureFixture
) -> None:
    plan = plan_query(ScriptedModel(reply), QUESTION)

    assert plan == plain_plan(QUESTION)
    assert "could not use the model's reading of the question" in caplog.text


def test_missing_language_and_intent_are_left_empty() -> None:
    reading = {"kind": "fact", "keywords": "argparse", "english": "argparse", "intent": None}

    plan = plan_query(ScriptedModel(reading), QUESTION)

    assert (plan.language, plan.intent) == ("", "")
    assert plan.queries == (QUESTION, "argparse")


def test_the_plain_plan_searches_for_the_question_as_typed() -> None:
    assert plain_plan(QUESTION) == QueryPlan(QUESTION, "", "", QueryKind.EXPLANATION, (QUESTION,))


def test_without_rewrites_the_model_is_not_asked() -> None:
    model = ScriptedModel()

    assert plan_query(model, QUESTION, rewrites=0) == plain_plan(QUESTION)
    assert model.calls == []


def test_a_model_that_cannot_be_reached_is_not_hidden() -> None:
    model = ScriptedModel(LanguageModelError("cannot reach Ollama"))

    with pytest.raises(LanguageModelError, match="cannot reach Ollama"):
        plan_query(model, QUESTION)
