from pathlib import Path

from digsite.evaluation.metrics import SetMetrics
from digsite.evaluation.near_duplicates import (
    evaluate,
    fingerprint_texts,
    load_pairs,
    load_texts,
    predict_pairs,
)

ARTICLE = (
    "The city council approved the new budget on Tuesday after a long debate. "
    "The plan raises spending on public transport and cuts funding for road building. "
    "Opposition members said the vote was rushed and promised to challenge it in court. "
    "The mayor defended the decision and said the city could not afford to wait."
)
TEXTS = {
    "t1": ARTICLE,
    "t2": ARTICLE.replace("on Tuesday", "on Wednesday"),
    "t3": "Researchers described a new species of deep-sea fish found near volcanic vents. "
    "The animal survives crushing pressure and total darkness by feeding on bacteria.",
    "t4": "A completely different text about gardening, tomatoes and the arrival of spring.",
}


def test_load_texts_accepts_space_and_tab_separated_files(tmp_path: Path) -> None:
    path = tmp_path / "texts.txt"
    path.write_text(
        "t120 The Supreme Court postponed a hearing\n"
        "248\tWhat is the greatest mystery?\n"
        "\n"
        "orphan-id\n"
        "t9\t\n",
        encoding="utf-8",
    )

    assert load_texts(path) == {
        "t120": "The Supreme Court postponed a hearing",
        "248": "What is the greatest mystery?",
    }


def test_load_pairs_ignores_order_and_repeats(tmp_path: Path) -> None:
    path = tmp_path / "pairs.txt"
    path.write_text("t2 t1\nt1 t2\nt3\tt4\n\nbroken\n", encoding="utf-8")

    assert load_pairs(path) == {("t1", "t2"), ("t3", "t4")}


def test_predict_pairs_returns_each_close_pair_once() -> None:
    fingerprints = {"a": 0b0000, "b": 0b0001, "c": 0b0011, "far": 0b1111_1111}

    assert predict_pairs(fingerprints, bits=8, max_distance=1) == {("a", "b"), ("b", "c")}
    assert predict_pairs(fingerprints, bits=8, max_distance=0) == set()


def test_fingerprints_depend_on_the_settings() -> None:
    words = fingerprint_texts(TEXTS, bits=64, shingle_size=1)
    pairs = fingerprint_texts(TEXTS, bits=64, shingle_size=2)

    assert words.keys() == pairs.keys() == TEXTS.keys()
    assert words != pairs


def test_evaluate_scores_each_distance() -> None:
    truth = {("t1", "t2")}

    results = evaluate(TEXTS, truth, bits=64, shingle_size=2, distances=[0, 6])

    assert results == {
        0: SetMetrics(true_positives=0, false_positives=0, false_negatives=1),
        6: SetMetrics(true_positives=1, false_positives=0, false_negatives=0),
    }
