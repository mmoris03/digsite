import pytest

from digsite.index import build_index, search_and, search_or


def test_builds_the_inverted_index_of_the_pizzas(pizza_index):
    """Every ingredient maps to the pizzas that have it, with how many times.

    The documents are the ingredients of five pizzas. A document's id is its
    position in the list.

    One assertion covers several cases, because the whole index is compared:

    - a token in a single document: "garlic" is only in pizza 1;
    - a token in several, with gaps: "oregano" is in 1, 3 and 4, not in 2;
    - a token in every document: "tomato";
    - a plural is another token: "olives" is indexed, "olive" is not;
    - no token is missing or left over, since a key that is not in the
      expected dict, or one that is only there, makes the comparison fail;
    - counts are kept per document: no pizza repeats an ingredient, so every
      count is 1, even for a token that is in all of them.
    """
    pizzas = [
        "artichokes mozzarella mushrooms oil olives tomato",
        "garlic oil oregano tomato",
        "fontina gorgonzola mozzarella tomato stracchino",
        "anchovies mozzarella oil oregano tomato",
        "mozzarella oil oregano sausage tomato",
    ]

    assert build_index(pizzas) == pizza_index


def test_a_repeated_word_is_counted():
    """A small corpus written inline, for a case the pizzas do not have.

    The count is how many times the token occurs in the document.
    """
    assert build_index(["go go go"]) == {"go": {0: 3}}


def test_no_documents_give_an_empty_index():
    """An empty corpus is a valid input: it has no tokens, not an error."""
    assert build_index([]) == {}


def test_an_empty_document_keeps_its_position():
    """The id is the position in the list, even for a document with no tokens.

    The empty document is id 0, so the tokens of the next one must be id 1.

    It also checks that the index is built with `tokenize`: "Hello World!" is
    indexed as "hello" and "world", lowercase and without the "!". The pizzas
    could not catch a plain `split()` instead, since they have no capitals and
    no punctuation. The tokenizing rules themselves are tested with `tokenize`.
    """
    assert build_index(["", "Hello World!"]) == {"hello": {1: 1}, "world": {1: 1}}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("mozzarella oregano", {3, 4}, id="two tokens"),
        pytest.param("mozzarella oil oregano", {3, 4}, id="three tokens"),
        pytest.param("garlic mozzarella", set(), id="no document has both"),
        pytest.param("basil oil", set(), id="a token that is not indexed"),
        pytest.param("tomato", {0, 1, 2, 3, 4}, id="a single token"),
        pytest.param("oil oil", {0, 1, 3, 4}, id="a repeated token"),
        pytest.param("Mozzarella, OREGANO!", {3, 4}, id="capitals and punctuation"),
        pytest.param("", set(), id="an empty query"),
        pytest.param(" ?! ", set(), id="a query with no tokens"),
    ],
)
def test_search_and_returns_the_documents_with_every_word(pizza_index, query, expected):
    """The result is the intersection of the documents of each token.

    - three tokens: all of them are intersected, not just the first two;
    - a token that is not indexed leaves no document, since none has it;
    - a repeated token changes nothing;
    - the query goes through `tokenize` like the documents did, so capitals
      and punctuation do not matter;
    - a query with no tokens finds nothing (and not every document, which is
      what an intersection of nothing would give). That is true of the empty
      string and also of a query made only of spaces and punctuation, which
      is not empty but has no tokens once tokenized.
    """
    assert search_and(pizza_index, query) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param("mozzarella oregano", {0, 1, 2, 3, 4}, id="overlapping tokens"),
        pytest.param("garlic sausage", {1, 4}, id="tokens with no document in common"),
        pytest.param("garlic sausage anchovies", {1, 3, 4}, id="three tokens"),
        pytest.param("basil oil", {0, 1, 3, 4}, id="a token that is not indexed"),
        pytest.param("basil", set(), id="only a token that is not indexed"),
        pytest.param("oregano", {1, 3, 4}, id="a single token"),
        pytest.param("oil oil", {0, 1, 3, 4}, id="a repeated token"),
        pytest.param("Garlic, SAUSAGE!", {1, 4}, id="capitals and punctuation"),
        pytest.param("", set(), id="an empty query"),
        pytest.param(" ?! ", set(), id="a query with no tokens"),
    ],
)
def test_search_or_returns_the_documents_with_any_word(pizza_index, query, expected):
    """The result is the union of the documents of each token.

    - three tokens: all of them are joined, not just the first two;
    - a token that is not indexed adds nothing, and does not stop the others;
    - a repeated token changes nothing;
    - the query goes through `tokenize` like the documents did, so capitals
      and punctuation do not matter;
    - a query with no tokens finds nothing. That is true of the empty string
      and also of a query made only of spaces and punctuation, which is not
      empty but has no tokens once tokenized.
    """
    assert search_or(pizza_index, query) == expected


@pytest.mark.parametrize("search", [search_and, search_or], ids=["and", "or"])
def test_changing_the_result_does_not_change_the_index(search, pizza_index):
    """The result is a new set, not the one stored in the index.

    A query with a single token is the risky one: returning the stored set as it
    is would let whoever adds to the result corrupt the index.
    """
    result = search(pizza_index, "oregano")

    result.add(99)

    assert pizza_index["oregano"] == {1: 1, 3: 1, 4: 1}


@pytest.mark.parametrize("search", [search_and, search_or], ids=["and", "or"])
def test_a_search_leaves_the_index_as_it_was(search, pizza_index):
    """Combining sets must not be done in place on a set of the index.

    A query with several tokens is the risky one: `&=` or `|=` on the first
    token's set would change that entry of the index.
    """
    before = {token: dict(postings) for token, postings in pizza_index.items()}

    search(pizza_index, "mozzarella oregano")

    assert pizza_index == before
