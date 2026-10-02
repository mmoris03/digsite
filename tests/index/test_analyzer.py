from digsite.index.analyzer import Analyzer, AnalyzerSettings, Language


def analyze(text: str, **settings: object) -> list[str]:
    return Analyzer(AnalyzerSettings(**settings)).analyze(text)  # type: ignore[arg-type]


def test_without_a_language_terms_are_the_case_folded_words() -> None:
    assert analyze("The Cats are RUNNING, the cats!") == [
        "the", "cats", "are", "running", "the", "cats",
    ]  # fmt: skip


def test_english_removes_stop_words_and_stems() -> None:
    assert analyze("The cats are running in the gardens", language=Language.ENGLISH) == [
        "cat", "run", "garden",
    ]  # fmt: skip


def test_spanish_removes_stop_words_and_stems() -> None:
    terms = analyze("Los módulos de la documentación están instalados", language=Language.SPANISH)

    assert terms == ["modul", "document", "instal"]


def test_spanish_stems_match_with_or_without_accents() -> None:
    accented = analyze("documentación", language=Language.SPANISH)
    plain = analyze("documentacion", language=Language.SPANISH)

    assert accented == plain


def test_stop_words_can_be_kept() -> None:
    terms = analyze("the cats are running", language=Language.ENGLISH, remove_stopwords=False)

    assert terms == ["the", "cat", "are", "run"]


def test_stemming_can_be_disabled() -> None:
    terms = analyze("the cats are running", language=Language.ENGLISH, stem=False)

    assert terms == ["cats", "running"]


def test_repetitions_and_order_are_kept() -> None:
    assert analyze("run runs running ran", language=Language.ENGLISH) == [
        "run",
        "run",
        "run",
        "ran",
    ]


def test_the_same_analyzer_gives_the_same_terms_every_time() -> None:
    analyzer = Analyzer(AnalyzerSettings(Language.ENGLISH))

    assert analyzer.analyze("aeroelastic models") == analyzer.analyze("aeroelastic models")


def test_text_without_words_has_no_terms() -> None:
    assert analyze("... — !!", language=Language.ENGLISH) == []
    assert analyze("the of and", language=Language.ENGLISH) == []
