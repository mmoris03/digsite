import pytest

from digsite.ranking import rank


@pytest.fixture
def pizza_texts_index():
    """The inverted index of four short texts about pizza, written by hand.

    Unlike the pizzas, these texts repeat words, so they show what counting
    does. A document's id is the position of its text in the list:

    0. "the pizza with the thin crust and the crispy edge"
    1. "the pizza dough needs flour water yeast and salt"
    2. "the best pizza has fresh mozzarella and fresh basil"
    3. "the mozzarella mozzarella and more mozzarella"
    """
    return {
        "and":          {0: 1, 1: 1, 2: 1, 3: 1},
        "basil":        {2: 1},
        "best":         {2: 1},
        "crispy":       {0: 1},
        "crust":        {0: 1},
        "dough":        {1: 1},
        "edge":         {0: 1},
        "flour":        {1: 1},
        "fresh":        {2: 2},
        "has":          {2: 1},
        "more":         {3: 1},
        "mozzarella":   {2: 1, 3: 3},
        "needs":        {1: 1},
        "pizza":        {0: 1, 1: 1, 2: 1},
        "salt":         {1: 1},
        "the":          {0: 3, 1: 1, 2: 1, 3: 1},
        "thin":         {0: 1},
        "water":        {1: 1},
        "with":         {0: 1},
        "yeast":        {1: 1},
    }


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "mozzarella oil oregano sausage",
            [(4, 4), (3, 3), (0, 2), (1, 2), (2, 1)],
            id="more matching words first",
        ),
        pytest.param("sausage garlic", [(1, 1), (4, 1)], id="ties go to the smaller id"),
        pytest.param("garlic", [(1, 1)], id="only documents with a query word"),
        pytest.param(
            "basil oil", [(0, 1), (1, 1), (3, 1), (4, 1)], id="a word that is not indexed"
        ),
        pytest.param("Garlic!", [(1, 1)], id="capitals and punctuation"),
        pytest.param("", [], id="an empty query"),
        pytest.param(" ?! ", [], id="a query with no words"),
    ],
)
def test_rank_orders_the_pizzas_by_how_many_words_they_have(pizza_index, query, expected):
    """No pizza repeats an ingredient, so a pizza scores one point per query word.

    - a tie goes to the smaller id, whatever the order of the words in the
      query: "sausage" comes before "garlic", but pizza 1 before pizza 4;
    - a pizza with none of the words does not appear, rather than scoring 0;
    - a word that is not indexed adds nothing;
    - the query goes through `tokenize`, like the documents did;
    - a query with no words finds nothing.
    """
    assert rank(pizza_index, query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("mozzarella", [(3, 3), (2, 1)], id="more occurrences first"),
        pytest.param("mozzarella basil", [(3, 3), (2, 2)], id="the counts are added"),
        pytest.param(
            "pizza pizza", [(0, 1), (1, 1), (2, 1)], id="a repeated query word counts once"
        ),
    ],
)
def test_rank_scores_by_how_often_the_words_occur(pizza_texts_index, query, expected):
    """A text scores, for each distinct query word, how many times it occurs in it.

    - "mozzarella" occurs three times in text 3 and once in text 2;
    - text 2 scores 1 for "mozzarella" plus 1 for "basil", and text 3 scores
      3 for "mozzarella" alone;
    - typing a word twice in the query does not count it twice.
    """
    assert rank(pizza_texts_index, query) == expected


def test_counting_favours_words_that_are_everywhere(pizza_texts_index):
    """Known limitation: for "the pizza dough", the text about dough is not first.

    Text 0 scores 3 for "the", a word every text has, and 1 for "pizza", so it
    comes before text 1, the only one with "dough". The result is pinned here
    so that this behaviour of counting is documented.
    """
    assert rank(pizza_texts_index, "the pizza dough") == [(0, 4), (1, 3), (2, 2), (3, 1)]


def test_ranking_leaves_the_index_as_it_was(pizza_index):
    """Adding up the scores must not change the postings of the index."""
    before = {word: dict(postings) for word, postings in pizza_index.items()}

    rank(pizza_index, "mozzarella oil")

    assert pizza_index == before
