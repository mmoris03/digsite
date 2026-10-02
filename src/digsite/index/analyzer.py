"""Text analysis: turning text into the terms that are indexed and searched."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import snowballstemmer

from digsite.index import stopwords
from digsite.text import tokenize


class Language(StrEnum):
    ENGLISH = "english"
    SPANISH = "spanish"


_STOPWORDS = {Language.ENGLISH: stopwords.ENGLISH, Language.SPANISH: stopwords.SPANISH}


@dataclass(frozen=True, slots=True)
class AnalyzerSettings:
    """How text becomes terms.

    Attributes:
        language: Language of the stop-word list and the stemmer. With None,
            terms are just the case-folded words.
        remove_stopwords: Drop the most common words of the language.
        stem: Reduce words to their stem, so that "running" matches "runs".
    """

    language: Language | None = None
    remove_stopwords: bool = True
    stem: bool = True


class Analyzer:
    """Turns text into index terms.

    Documents and queries must go through the same analyzer: a term only
    matches if both sides produced exactly the same string.
    """

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self.settings = settings or AnalyzerSettings()
        language = self.settings.language
        self._stopwords: frozenset[str] = (
            _STOPWORDS[language] if language and self.settings.remove_stopwords else frozenset()
        )
        self._stem_word: Callable[[str], str] | None = None
        if language and self.settings.stem:
            self._stem_word = snowballstemmer.stemmer(language.value).stemWord
        # A corpus repeats a small vocabulary many times: stem each word once.
        self._stems: dict[str, str] = {}

    def analyze(self, text: str) -> list[str]:
        """Return the terms of a text, in order, with repetitions."""
        tokens = [token for token in tokenize(text) if token not in self._stopwords]
        stem_word = self._stem_word
        if stem_word is None:
            return tokens

        stems = self._stems
        terms = []
        for token in tokens:
            stem = stems.get(token)
            if stem is None:
                stem = stems[token] = stem_word(token)
            terms.append(stem)
        return terms
