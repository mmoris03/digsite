from digsite.text import tokenize


def build_index(docs: list[str]) -> dict[str, set[int]]:
    """Map each word to the ids of the documents that contain it."""
    index: dict[str, set[int]] = {}
    for i, doc in enumerate(docs):
        for token in tokenize(doc):
            index.setdefault(token, set())
            index[token].add(i)

    return index
