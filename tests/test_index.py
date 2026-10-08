from digsite.index import build_index


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
