from digsite.text import tokenize


def test_splits_on_punctuation_and_folds_case() -> None:
    assert tokenize("¡Hola, Mundo! It's 2026.") == ["hola", "mundo", "it", "s", "2026"]


def test_keeps_accented_letters_and_underscores() -> None:
    assert tokenize("El módulo __main__ de Python") == ["el", "módulo", "__main__", "de", "python"]


def test_text_without_words_has_no_tokens() -> None:
    assert tokenize("") == []
    assert tokenize(" ... ¶ — ") == []
