from digsite.text import tokenize


def build_index(docs: list[str]) -> dict[str, set[int]]:
    """Map each word to the ids of the documents that contain it."""
    index: dict[str, set[int]] = {}
    for i, doc in enumerate(docs):
        for token in tokenize(doc):
            index.setdefault(token, set())
            index[token].add(i)

    return index


def _postings_of_query(index: dict[str, set[int]], query: str) -> dict[str, set[int]]:
    """The postings of each word of the query, by word.

    A word that is not indexed stays in the result with no documents, so that
    a search can tell "nobody has it" from "it was not asked for". A word
    repeated in the query appears once: repetitions do not count.
    """
    return {word: index.get(word, set()) for word in tokenize(query)}


def search_and(index: dict[str, set[int]], query: str) -> set[int]:
    """Ids of the documents that contain every word of the query."""
    postings = _postings_of_query(index, query)
    if not postings:
        # An intersection needs at least one set, and a query with no words
        # should find nothing, not every document.
        return set()

    return set.intersection(*postings.values())


def search_or(index: dict[str, set[int]], query: str) -> set[int]:
    """Ids of the documents that contain at least one word of the query."""
    postings = _postings_of_query(index, query)

    return set().union(*postings.values())
