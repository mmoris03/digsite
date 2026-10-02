import json
from pathlib import Path

import pytest

from digsite.answer import Answer, QueryKind, QueryPlan, Source
from digsite.evaluation.answers import (
    AnswerJudgement,
    AnswerMetrics,
    from_record,
    judge_answer,
    load_questions,
    summarize,
    to_record,
)

SITE = "https://example.com"
SOURCES = (
    Source(1, 11, f"{SITE}/caching", "Caching", "Eviction", "The least recently used goes."),
    Source(2, 12, f"{SITE}/caching", "Caching", "", "The cache stores results."),
    Source(3, 27, f"{SITE}/logging", "Logging", "", "Logs rotate daily."),
)
PLAN = QueryPlan("q", "", "", QueryKind.FACT, ("q",))


def answer(cited: tuple[int, ...] = (), *, answered: bool = True, sources=SOURCES) -> Answer:
    return Answer("q", answered, "text" if answered else "", sources, cited, PLAN)


def test_an_answer_is_judged_by_where_its_citations_point() -> None:
    judgement = judge_answer(answer((1, 3)), {f"{SITE}/caching"})

    assert judgement == AnswerJudgement(
        answered=True, citations=2, relevant_citations=1, relevant_found=True
    )


def test_two_passages_of_one_relevant_page_are_two_relevant_citations() -> None:
    assert judge_answer(answer((1, 2)), {f"{SITE}/caching"}).relevant_citations == 2


def test_an_answer_may_miss_the_relevant_passage_it_was_given() -> None:
    judgement = judge_answer(answer((3,)), {f"{SITE}/caching"})

    assert judgement.relevant_citations == 0
    assert judgement.relevant_found


def test_a_declined_answer_still_says_whether_the_relevant_page_was_found() -> None:
    found = judge_answer(answer(answered=False), {f"{SITE}/logging"})
    missed = judge_answer(answer(answered=False), {f"{SITE}/queues"})

    assert found == AnswerJudgement(False, 0, 0, relevant_found=True)
    assert missed == AnswerJudgement(False, 0, 0, relevant_found=False)


def test_a_question_nothing_answers_has_no_relevant_citation() -> None:
    judgement = judge_answer(answer((1,)), set())

    assert judgement == AnswerJudgement(True, 1, 0, relevant_found=False)


def test_metrics_are_shares_of_the_questions() -> None:
    judgements = [
        AnswerJudgement(True, 2, 2, True),  # answered from the right pages
        AnswerJudgement(True, 2, 1, True),  # one citation points elsewhere
        AnswerJudgement(True, 1, 0, True),  # cites only the wrong page
        AnswerJudgement(True, 0, 0, True),  # cites nothing
        AnswerJudgement(False, 0, 0, False),  # declined; the page was not found
    ]

    metrics = summarize(judgements)

    assert metrics == AnswerMetrics(
        questions=5,
        answered=pytest.approx(4 / 5),
        supported=pytest.approx(2 / 5),
        uncited=pytest.approx(1 / 5),
        citation_precision=pytest.approx(3 / 5),
        relevant_found=pytest.approx(4 / 5),
    )


def test_no_citations_at_all_is_a_precision_of_zero() -> None:
    metrics = summarize([AnswerJudgement(False, 0, 0, True)])

    assert metrics.citation_precision == 0.0
    assert metrics.answered == 0.0


def test_no_questions_give_zeros() -> None:
    assert summarize([]) == AnswerMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_an_answer_survives_being_written_as_json_and_read_back() -> None:
    original = Answer(
        question="what is evicted",
        answered=True,
        text="The least recently used entry [1].",
        sources=SOURCES,
        cited=(1,),
        plan=QueryPlan("what is evicted", "English", "Know it.", QueryKind.FACT, ("a", "b")),
    )

    record = json.loads(json.dumps(to_record(original)))

    assert from_record(record) == original
    assert record["queries"] == ["a", "b"]
    assert record["sources"][0]["url"] == f"{SITE}/caching"


@pytest.mark.parametrize(
    "record",
    [
        {},
        {"question": "q", "answered": True, "text": "t", "cited": [], "sources": []},
        {**to_record(answer((1,))), "sources": [{"number": 1}]},
        {**to_record(answer((1,))), "kind": "poem"},
    ],
)
def test_a_record_of_another_shape_is_rejected(record: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Not the record of an answer"):
        from_record(record)


def test_questions_are_read_one_per_line(tmp_path: Path) -> None:
    path = tmp_path / "questions.tsv"
    path.write_text(
        "# a comment\n\nwhat is a cache\n  how are logs rotated  \n"
        f"labelled query\t{SITE}/caching\t{SITE}/logging\n",
        encoding="utf-8",
    )

    assert load_questions(path) == ["what is a cache", "how are logs rotated", "labelled query"]
