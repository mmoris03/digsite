from digsite.text import tokenize


def build_index(docs: list[str]) -> dict[str, dict[int, int]]:
    """Map each token to the documents that contain it, with its tf in each.

    [doc, ...] -> {token: {doc: tf, ...}, ...}

    A document is identified by its position in the list, and a token's tf
    (term frequency) in it is how many times the token occurs there.
    """
    index: dict[str, dict[int, int]] = {}
    for i, doc in enumerate(docs):
        for token in tokenize(doc):
            index.setdefault(token, dict())
            index[token].setdefault(i, 0)
            index[token][i] += 1

    return index


def postings_of_query(index: dict[str, dict[int, int]], query: str) -> dict[str, dict[int, int]]:
    """The postings of each token of the query, by token.

    index, query -> {token: {doc: tf, ...}, ...}

    A token that is not indexed stays in the result with no documents, so
    that a search can tell "nobody has it" from "it was not asked for". A
    token repeated in the query appears once.
    """
    return {
        token: index.get(token, {})
        for token in tokenize(query)
    }


def search_and(index: dict[str, dict[int, int]], query: str) -> set[int]:
    """The documents that contain every token of the query.

    index, query -> {doc, ...}
    """
    postings = postings_of_query(index, query)
    if not postings:
        # A query with no tokens finds nothing
        return set()

    return set.intersection(*(set(documents) for documents in postings.values()))


def search_or(index: dict[str, dict[int, int]], query: str) -> set[int]:
    """The documents that contain at least one token of the query.

    index, query -> {doc, ...}
    """
    postings = postings_of_query(index, query)
    if not postings:
        # A query with no tokens finds nothing
        return set()

    return set.union(*(set(documents) for documents in postings.values()))
