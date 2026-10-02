"""Splitting documents into passages.

A passage is the unit that gets indexed, retrieved and later handed to a
language model. It has to be small enough to be about one thing and to fit a
model's input, and it has to make sense out of its document. So documents are
split along their own structure, and every passage remembers the headings it
sat under.
"""

import re

from digsite.models import Passage

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_MARKUP_RE = re.compile(r"[`*]")
_FENCE = "```"
# The whitespace after the end of a sentence, which may close a quote (straight or
# typographic) or a bracket.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?][\"'\u201d\u2019)\]])\s+")

# Size is measured in characters, not words: in technical text one "word" can be
# a long identifier that a model splits into many tokens. See docs/decisions.md.
DEFAULT_MAX_CHARS = 1000


def split_into_passages(markdown: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Passage]:
    """Split a Markdown document into passages of at most `max_chars` characters.

    A passage never crosses a heading. Within a section, consecutive paragraphs
    are packed together while they fit. A paragraph that is too long on its own
    is split at line breaks; a line that is still too long, between sentences;
    a sentence, between words; and a single word longer than the limit,
    wherever the limit falls. No word is dropped and none is repeated.

    A fenced code block counts as one paragraph, blank lines included, and a
    line starting with `#` inside it is a comment, not a heading.

    Args:
        markdown: The document text.
        max_chars: Upper bound on the length of a passage.
    """
    if max_chars < 1:
        raise ValueError("Passages must allow at least one character")

    passages: list[Passage] = []
    headings: list[str] = []
    body: list[str] = []

    def close_section() -> None:
        paragraphs = _paragraphs(body, max_chars)
        body.clear()
        for piece in _pack(paragraphs, max_chars, "\n\n"):
            passages.append(Passage(tuple(headings), piece))

    in_code = False
    for line in markdown.splitlines():
        if _toggles_code(line):
            in_code = not in_code
            body.append(line)
            continue
        # Inside code, `# text` is a comment.
        match = None if in_code else _HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue
        close_section()
        level = len(match.group(1))
        # A heading replaces those at its own level and below.
        del headings[level - 1 :]
        headings.append(_MARKUP_RE.sub("", match.group(2)).strip())
    close_section()
    return passages


def _toggles_code(line: str) -> bool:
    """Tell whether a line opens or closes a fenced code block.

    The fence is usually alone on its line, but extraction can leave an opening
    one at the end of a line of text, so fences are counted wherever they are.
    A line with two of them opens and closes a block by itself.
    """
    return line.count(_FENCE) % 2 == 1


def _paragraphs(lines: list[str], max_chars: int) -> list[str]:
    """Group a section's lines into paragraphs, splitting those that do not fit in a passage.

    Paragraphs are separated by blank lines, except inside a fenced code block,
    which is kept whole with its own blank lines.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    for line in lines:
        if _toggles_code(line):
            in_code = not in_code
        if line.strip() or in_code:
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))

    paragraphs = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(block) <= max_chars:
            paragraphs.append(block)
        else:
            paragraphs.extend(_pack(_lines(block, max_chars), max_chars, "\n"))
    return paragraphs


def _lines(block: str, max_chars: int) -> list[str]:
    """Break a paragraph into lines, splitting those that do not fit in a passage."""
    lines = []
    for line in block.splitlines():
        line = line.rstrip()  # indentation is kept: it matters in code
        if not line.strip():
            continue
        if len(line) <= max_chars:
            lines.append(line)
        else:
            lines.extend(_pack(_sentences(line.strip(), max_chars), max_chars, " "))
    return lines


def _sentences(line: str, max_chars: int) -> list[str]:
    """Break a line into sentences, splitting those that do not fit in a passage.

    A full stop after an abbreviation is taken for the end of a sentence. That
    is harmless: sentences are packed back together, so a wrong split only
    shows if it happens to fall where a passage ends.
    """
    sentences = []
    for sentence in _SENTENCE_END_RE.split(line):
        if len(sentence) <= max_chars:
            sentences.append(sentence)
        else:
            sentences.extend(_pack(_words(sentence, max_chars), max_chars, " "))
    return sentences


def _words(sentence: str, max_chars: int) -> list[str]:
    """Break a sentence into words, cutting any word longer than a passage."""
    words: list[str] = []
    for word in sentence.split():
        words.extend(word[start : start + max_chars] for start in range(0, len(word), max_chars))
    return words


def _pack(units: list[str], max_chars: int, separator: str) -> list[str]:
    """Join consecutive units into the fewest pieces of at most `max_chars` characters."""
    pieces = []
    current: list[str] = []
    size = 0
    for unit in units:
        added = len(unit) + (len(separator) if current else 0)
        if current and size + added > max_chars:
            pieces.append(separator.join(current))
            current, size, added = [], 0, len(unit)
        current.append(unit)
        size += added
    if current:
        pieces.append(separator.join(current))
    return pieces
