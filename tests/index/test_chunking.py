import random

import pytest

from digsite.index.chunking import split_into_passages
from digsite.models import Passage

DOCUMENT = """# argparse — Command-line parsing

The argparse module makes it easy to write command-line interfaces.

## Core functionality

The parser is a container for argument specifications.

It has options that apply to the parser as a whole.

### The add_argument() method

This method attaches individual argument specifications.

## Tutorial

A gentle introduction.
"""


def test_passages_follow_the_sections_and_remember_their_headings() -> None:
    assert split_into_passages(DOCUMENT) == [
        Passage(
            ("argparse — Command-line parsing",),
            "The argparse module makes it easy to write command-line interfaces.",
        ),
        Passage(
            ("argparse — Command-line parsing", "Core functionality"),
            "The parser is a container for argument specifications.\n\n"
            "It has options that apply to the parser as a whole.",
        ),
        Passage(
            ("argparse — Command-line parsing", "Core functionality", "The add_argument() method"),
            "This method attaches individual argument specifications.",
        ),
        Passage(("argparse — Command-line parsing", "Tutorial"), "A gentle introduction."),
    ]


def test_context_is_the_heading_path() -> None:
    passage = split_into_passages(DOCUMENT)[2]

    assert passage.context == (
        "argparse — Command-line parsing > Core functionality > The add_argument() method"
    )


def test_headings_lose_their_inline_markup() -> None:
    passages = split_into_passages("# `abc` — *Abstract* Base Classes\n\nText.")

    assert passages[0].headings == ("abc — Abstract Base Classes",)


def test_text_before_any_heading_has_no_headings() -> None:
    passages = split_into_passages("An opening line.\n\n# Title\n\nBody.")

    assert passages == [Passage((), "An opening line."), Passage(("Title",), "Body.")]


def test_a_heading_without_text_of_its_own_produces_no_passage() -> None:
    passages = split_into_passages("# Title\n\n## Empty\n\n## Filled\n\nBody.")

    assert passages == [Passage(("Title", "Filled"), "Body.")]


def test_a_heading_that_skips_a_level_still_nests() -> None:
    passages = split_into_passages("# Title\n\n### Deep\n\nBody.\n\n## Back\n\nMore.")

    assert [passage.headings for passage in passages] == [("Title", "Deep"), ("Title", "Back")]


def test_a_hash_inside_a_line_is_not_a_heading() -> None:
    passages = split_into_passages("Use the #channel or C# here.\n#hashtag without a space")

    assert len(passages) == 1
    assert passages[0].headings == ()


CODE = """# Sorting

Use the key argument:

```
# decorate, sort, undecorate
pairs = [(len(word), word) for word in words]

pairs.sort()
```

## Stability

Sorts are stable.
"""


def test_a_comment_inside_a_code_block_is_not_a_heading() -> None:
    passages = split_into_passages(CODE)

    assert [passage.headings for passage in passages] == [("Sorting",), ("Sorting", "Stability")]
    assert "# decorate, sort, undecorate" in passages[0].text


def test_a_code_block_is_kept_whole_with_its_blank_lines() -> None:
    passages = split_into_passages(CODE, max_chars=110)

    assert [passage.text for passage in passages] == [
        "Use the key argument:",
        "```\n# decorate, sort, undecorate\n"
        "pairs = [(len(word), word) for word in words]\n\npairs.sort()\n```",
        "Sorts are stable.",
    ]


def test_a_code_block_too_long_for_a_passage_keeps_its_indentation() -> None:
    code = "```\nfor item in items:\n    if item:\n        handle(item)\n```"

    passages = split_into_passages(code, max_chars=40)

    assert [passage.text for passage in passages] == [
        "```\nfor item in items:\n    if item:",
        "        handle(item)\n```",
    ]


def test_a_comment_on_its_own_between_blank_lines_in_code_is_not_a_heading() -> None:
    text = "# Title\n\n```\nx = 1\n\n# next step\n\ny = 2\n```\n\n## Section\n\nBody."

    passages = split_into_passages(text)

    assert passages == [
        Passage(("Title",), "```\nx = 1\n\n# next step\n\ny = 2\n```"),
        Passage(("Title", "Section"), "Body."),
    ]


def test_a_code_block_may_open_at_the_end_of_a_line_of_text() -> None:
    # Content extraction leaves some fences there instead of on a line of their own.
    text = "# Title\n\nInstall it with ```\n# as root\npip install x\n```\n\n## Next\n\nBody."

    passages = split_into_passages(text)

    assert passages == [
        Passage(("Title",), "Install it with ```\n# as root\npip install x\n```"),
        Passage(("Title", "Next"), "Body."),
    ]


def test_a_block_opened_and_closed_on_one_line_leaves_the_next_heading_alone() -> None:
    passages = split_into_passages("# Title\n\n```pip install x```\n\n## Next\n\nBody.")

    assert [passage.headings for passage in passages] == [("Title",), ("Title", "Next")]


def test_paragraphs_are_packed_while_they_fit() -> None:
    text = "one two three\n\nfour five six\n\nseven eight nine\n\nten"

    # The first two paragraphs and the blank line between them take 28 characters.
    passages = split_into_passages(text, max_chars=28)

    assert [passage.text for passage in passages] == [
        "one two three\n\nfour five six",
        "seven eight nine\n\nten",
    ]


def test_a_long_paragraph_is_split_at_its_line_breaks() -> None:
    text = "a1 a2 a3\nb1 b2 b3\nc1 c2 c3\nd1 d2"

    passages = split_into_passages(text, max_chars=17)

    assert [passage.text for passage in passages] == ["a1 a2 a3\nb1 b2 b3", "c1 c2 c3\nd1 d2"]


def test_a_long_line_is_split_between_sentences() -> None:
    text = 'First sentence here. Is this the second? He said "stop." Then a fourth one!'

    passages = split_into_passages(text, max_chars=42)

    assert [passage.text for passage in passages] == [
        "First sentence here. Is this the second?",
        'He said "stop." Then a fourth one!',
    ]


def test_a_long_sentence_is_split_between_words() -> None:
    text = " ".join(f"w{number}" for number in range(10))

    passages = split_into_passages(text, max_chars=11)

    assert [passage.text for passage in passages] == ["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"]


def test_a_word_longer_than_a_passage_is_cut() -> None:
    passages = split_into_passages("see abcdefghij now", max_chars=4)

    assert [passage.text for passage in passages] == ["see", "abcd", "efgh", "ij", "now"]


def test_empty_and_blank_documents_have_no_passages() -> None:
    assert split_into_passages("") == []
    assert split_into_passages("\n\n   \n") == []
    assert split_into_passages("# Only a heading") == []


def test_the_limit_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least one character"):
        split_into_passages("text", max_chars=0)


@pytest.mark.parametrize("max_chars", [8, 40, 300, 1000])
def test_nothing_is_lost_repeated_or_oversized(max_chars: int) -> None:
    rng = random.Random(max_chars)
    blocks = []
    expected_words = []
    for number in range(120):
        if number % 9 == 0:
            blocks.append("#" * rng.randint(1, 4) + f" Heading {number}")
            continue
        lines = []
        for _ in range(rng.randint(1, 6)):
            words = [f"w{rng.randrange(10_000)}" for _ in range(rng.randint(1, 80))]
            expected_words.extend(words)
            lines.append(" ".join(words))
        blocks.append("\n".join(lines))
    document = "\n\n".join(blocks)

    passages = split_into_passages(document, max_chars=max_chars)

    assert all(len(passage.text) <= max_chars for passage in passages)
    assert all(passage.text.strip() for passage in passages)
    assert [word for passage in passages for word in passage.text.split()] == expected_words
