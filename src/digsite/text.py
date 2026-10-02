"""Text utilities shared by several stages."""

import re

_WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Split text into case-folded word tokens, dropping punctuation."""
    return _WORD_RE.findall(text.casefold())
