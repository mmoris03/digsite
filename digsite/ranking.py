from digsite.index import postings_of_query


def rank(index: dict[str, dict[int, int]], query: str) -> list[tuple[int, float]]:
    """Documents that match the query, best first, with their score."""
    scores: dict[int, float] = {}
    for postings in postings_of_query(index, query).values():
        for document, count in postings.items():
            scores[document] = scores.get(document, 0) + count

    # Highest score first, and a tie goes to the smaller id
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
