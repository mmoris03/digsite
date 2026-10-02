import pytest

from digsite.llm.replies import parse_json_object


def test_reads_a_json_object() -> None:
    assert parse_json_object('{"answerable": true, "answer": "Yes [1]."}') == {
        "answerable": True,
        "answer": "Yes [1].",
    }


def test_ignores_the_whitespace_around_it() -> None:
    assert parse_json_object('\n  {"a": 1}\n') == {"a": 1}


@pytest.mark.parametrize("opening", ["```json", "```", "```JSON"])
def test_unwraps_a_markdown_code_block(opening: str) -> None:
    assert parse_json_object(f'{opening}\n{{"a": [1, 2]}}\n```') == {"a": [1, 2]}


def test_keeps_code_blocks_that_are_inside_the_json() -> None:
    reply = '{"answer": "Run:\\n```\\npip install x\\n```"}'

    assert parse_json_object(reply) == {"answer": "Run:\n```\npip install x\n```"}


@pytest.mark.parametrize("reply", ["", "Sure! Here is the answer.", '{"a": 1', '{"a": 1} trailing'])
def test_a_reply_that_is_not_json_is_an_error(reply: str) -> None:
    with pytest.raises(ValueError, match="not JSON"):
        parse_json_object(reply)


@pytest.mark.parametrize("reply", ["[1, 2]", '"text"', "42", "null"])
def test_json_that_is_not_an_object_is_an_error(reply: str) -> None:
    with pytest.raises(ValueError, match="not an object"):
        parse_json_object(reply)
