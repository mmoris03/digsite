from digsite.text import tokenize


def test_splits_words_and_lowercases_them():
    """The simplest case: words come out separated and in lowercase."""
    assert tokenize("Hello World") == ["hello", "world"]


def test_empty_text_has_no_tokens():
    """No text, no tokens: an empty list, not an error."""
    assert tokenize("") == []


def test_only_punctuation_has_no_tokens():
    """Separators alone make no word."""
    assert tokenize(" ... ") == []


def test_keeps_order_and_repetitions():
    """Tokens come in text order and a repeated word appears each time.

    Later stages count how often each word occurs, so the result must be a
    list, not a set, and must not be sorted.
    """
    assert tokenize("the cat saw the dog") == ["the", "cat", "saw", "the", "dog"]


def test_case_and_punctuation_do_not_change_the_word():
    """"Python", "python" and "PYTHON!" are the same word."""
    assert tokenize("Python, python; PYTHON!") == ["python", "python", "python"]


def test_keeps_accents_and_enye():
    """Accented letters and ñ are word characters, not separators."""
    assert tokenize("Canción piña ÁRBOL") == ["canción", "piña", "árbol"]


def test_non_latin_scripts_are_words():
    """Word characters are not limited to the Latin alphabet."""
    assert tokenize("日本語 テキスト") == ["日本語", "テキスト"]


def test_casefold_expands_eszett():
    """casefold, not lower: ß becomes ss, so "Straße" matches "STRASSE"."""
    assert tokenize("Straße") == ["strasse"]


def test_any_whitespace_separates_words():
    """Tabs and line breaks separate words just like spaces."""
    assert tokenize("tab\tnew\nline") == ["tab", "new", "line"]


def test_spanish_question_marks_are_dropped():
    """Both the opening ¿ and the closing ? are punctuation."""
    assert tokenize("¿Qué es esto?") == ["qué", "es", "esto"]


def test_letters_and_digits_stay_together():
    """A word with digits in it is a single token."""
    assert tokenize("python3") == ["python3"]


def test_underscore_is_a_word_character():
    """snake_case is one token: the underscore does not separate."""
    assert tokenize("snake_case") == ["snake_case"]


def test_hyphen_separates_words():
    """kebab-case is two tokens: the hyphen does separate, unlike the underscore."""
    assert tokenize("kebab-case") == ["kebab", "case"]


def test_decimal_point_splits_a_number():
    """Known limitation: "3.14" becomes "3" and "14", so the number is lost.

    The expected result is pinned here so that a change in behaviour is noticed.
    """
    assert tokenize("3.14") == ["3", "14"]


def test_plus_signs_are_dropped():
    """Known limitation: "C++" becomes "c", the same word as plain "C"."""
    assert tokenize("C++") == ["c"]


def test_apostrophe_splits_a_word():
    """Known limitation: "don't" becomes "don" and "t"."""
    assert tokenize("don't") == ["don", "t"]
