import pytest

from digsite.index import build_index, search_and, search_or


@pytest.fixture
def pizza_index():
    """The inverted index of five pizzas, written by hand.

    It is the index `build_index` makes from their ingredients, but written out
    so that the search tests do not depend on `build_index`: a document's id is
    the position of its pizza in the list.
    """
    return {
        "anchovies": {3},
        "artichokes": {0},
        "fontina": {2},
        "garlic": {1},
        "gorgonzola": {2},
        "mozzarella": {0, 2, 3, 4},
        "mushrooms": {0},
        "oil": {0, 1, 3, 4},
        "olives": {0},
        "oregano": {1, 3, 4},
        "sausage": {4},
        "stracchino": {2},
        "tomato": {0, 1, 2, 3, 4},
    }


def test_builds_the_inverted_index_of_the_pizzas():
    """Every ingredient maps to the ids of the pizzas that have it.

    The documents are the ingredients of five pizzas. A document's id is its
    position in the list.

    One assertion covers several cases, because the whole index is compared:

    - a word in a single document: "garlic" is only in pizza 1;
    - a word in several, with gaps: "oregano" is in 1, 3 and 4, not in 2;
    - a word in every document: "tomato";
    - a plural is another word: "olives" is indexed, "olive" is not;
    - no word is missing or left over, since a key that is not in the
      expected dict, or one that is only there, makes the comparison fail;
    - the values are sets, not lists, since {0, 1} != [0, 1].
    """
    pizzas = [
        "artichokes mozzarella mushrooms oil olives tomato",
        "garlic oil oregano tomato",
        "fontina gorgonzola mozzarella tomato stracchino",
        "anchovies mozzarella oil oregano tomato",
        "mozzarella oil oregano sausage tomato",
    ]

    assert build_index(pizzas) == {
        "anchovies":    {3},
        "artichokes":   {0},
        "fontina":      {2},
        "garlic":       {1},
        "gorgonzola":   {2},
        "mozzarella":   {0, 2, 3, 4},
        "mushrooms":    {0},
        "oil":          {0, 1, 3, 4},
        "olives":       {0},
        "oregano":      {1, 3, 4},
        "sausage":      {4},
        "stracchino":   {2},
        "tomato":       {0, 1, 2, 3, 4},
    }


def test_a_repeated_word_counts_once_per_document():
    """A small corpus written inline, for a case the pizzas do not have.

    Each document id is stored once however many times the word occurs.
    """
    assert build_index(["go go go"]) == {"go": {0}}


def test_no_documents_give_an_empty_index():
    """An empty corpus is a valid input: it has no words, not an error."""
    assert build_index([]) == {}


def test_an_empty_document_keeps_its_position():
    """The id is the position in the list, even for a document with no words.

    The empty document is id 0, so the words of the next one must be id 1.

    It also checks that the index is built with `tokenize`: "Hello World!" is
    indexed as "hello" and "world", lowercase and without the "!". The pizzas
    could not catch a plain `split()` instead, since they have no capitals and
    no punctuation. The tokenizing rules themselves are tested with `tokenize`.
    """
    assert build_index(["", "Hello World!"]) == {"hello": {1}, "world": {1}}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("mozzarella oregano", {3, 4}, id="two words"),
        pytest.param("mozzarella oil oregano", {3, 4}, id="three words"),
        pytest.param("garlic mozzarella", set(), id="no document has both"),
        pytest.param("basil oil", set(), id="a word that is not indexed"),
        pytest.param("tomato", {0, 1, 2, 3, 4}, id="a single word"),
        pytest.param("oil oil", {0, 1, 3, 4}, id="a repeated word"),
        pytest.param("Mozzarella, OREGANO!", {3, 4}, id="capitals and punctuation"),
        pytest.param("", set(), id="an empty query"),
        pytest.param(" ?! ", set(), id="a query with no words"),
    ],
)
def test_search_and_returns_the_documents_with_every_word(pizza_index, query, expected):
    """The result is the intersection of the documents of each word.

    - three words: all of them are intersected, not just the first two;
    - a word that is not indexed leaves no document, since none has it;
    - a repeated word changes nothing;
    - the query goes through `tokenize` like the documents did, so capitals
      and punctuation do not matter;
    - a query with no words finds nothing (and not every document, which is
      what an intersection of nothing would give). That is true of the empty
      string and also of a query made only of spaces and punctuation, which
      is not empty but has no words once tokenized.
    """
    assert search_and(pizza_index, query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("mozzarella oregano", {0, 1, 2, 3, 4}, id="overlapping words"),
        pytest.param("garlic sausage", {1, 4}, id="words with no document in common"),
        pytest.param("garlic sausage anchovies", {1, 3, 4}, id="three words"),
        pytest.param("basil oil", {0, 1, 3, 4}, id="a word that is not indexed"),
        pytest.param("basil", set(), id="only a word that is not indexed"),
        pytest.param("oregano", {1, 3, 4}, id="a single word"),
        pytest.param("oil oil", {0, 1, 3, 4}, id="a repeated word"),
        pytest.param("Garlic, SAUSAGE!", {1, 4}, id="capitals and punctuation"),
        pytest.param("", set(), id="an empty query"),
        pytest.param(" ?! ", set(), id="a query with no words"),
    ],
)
def test_search_or_returns_the_documents_with_any_word(pizza_index, query, expected):
    """The result is the union of the documents of each word.

    - three words: all of them are joined, not just the first two;
    - a word that is not indexed adds nothing, and does not stop the others;
    - a repeated word changes nothing;
    - the query goes through `tokenize` like the documents did, so capitals
      and punctuation do not matter;
    - a query with no words finds nothing. That is true of the empty string
      and also of a query made only of spaces and punctuation, which is not
      empty but has no words once tokenized.
    """
    assert search_or(pizza_index, query) == expected


@pytest.mark.parametrize("search", [search_and, search_or], ids=["and", "or"])
def test_changing_the_result_does_not_change_the_index(search, pizza_index):
    """The result is a new set, not the one stored in the index.

    A query with a single word is the risky one: returning the stored set as it
    is would let whoever adds to the result corrupt the index.
    """
    result = search(pizza_index, "oregano")

    result.add(99)

    assert pizza_index["oregano"] == {1, 3, 4}


@pytest.mark.parametrize("search", [search_and, search_or], ids=["and", "or"])
def test_a_search_leaves_the_index_as_it_was(search, pizza_index):
    """Combining sets must not be done in place on a set of the index.

    A query with several words is the risky one: `&=` or `|=` on the first
    word's set would change that entry of the index.
    """
    before = {word: set(documents) for word, documents in pizza_index.items()}

    search(pizza_index, "mozzarella oregano")

    assert pizza_index == before
