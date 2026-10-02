"""Query expansion by pseudo-relevance feedback.

The best results of a query are assumed to be relevant, and the terms that are
unusually frequent in them, compared with the rest of the collection, are added
to the query. A document about the topic can then be found even if it uses
none of the words the user typed.

"Unusually frequent" is measured with Dunning's log-likelihood ratio (1993),
in the signed-root form popularised by Apache Mahout.
"""

import math
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

from digsite.index.inverted_index import InvertedIndex


@dataclass(frozen=True, slots=True)
class ExpansionSettings:
    """How a query is expanded.

    Attributes:
        feedback_documents: How many top results are taken as relevant.
        terms: How many terms are added to the query.
        weight: Weight of each added term, relative to 1 for the original ones.
        min_feedback_documents: In how many feedback documents a term must occur
            to be a candidate. Above 1, a term peculiar to a single document
            cannot be added.
    """

    # Defaults chosen by evaluation on two test collections; see docs/decisions.md.
    feedback_documents: int = 2
    terms: int = 40
    weight: float = 0.15
    min_feedback_documents: int = 1


def log_likelihood_ratio(k11: int, k12: int, k21: int, k22: int) -> float:
    """Dunning's G² statistic for a 2x2 table of counts.

    It measures how far the table is from what independence of rows and columns
    would give: 0 when they are independent, larger the stronger the association.

    Args:
        k11: Times the event occurred in the first group.
        k12: Times the event occurred in the second group.
        k21: Times anything else occurred in the first group.
        k22: Times anything else occurred in the second group.
    """
    row_entropy = _entropy(k11 + k12, k21 + k22)
    column_entropy = _entropy(k11 + k21, k12 + k22)
    matrix_entropy = _entropy(k11, k12, k21, k22)
    # Rounding can leave a tiny negative number where the true value is 0.
    return max(0.0, 2.0 * (row_entropy + column_entropy - matrix_entropy))


def signed_root_llr(k11: int, k12: int, k21: int, k22: int) -> float:
    """Square root of the log-likelihood ratio, with the direction of the association.

    Positive when the event is more frequent in the first group than in the
    second, negative when it is less frequent.
    """
    root = math.sqrt(log_likelihood_ratio(k11, k12, k21, k22))
    first = k11 / (k11 + k21) if k11 + k21 else 0.0
    second = k12 / (k12 + k22) if k12 + k22 else 0.0
    return -root if first < second else root


def expansion_terms[K](
    index: InvertedIndex[K],
    feedback: Collection[Counter[str]],
    exclude: Collection[str],
    settings: ExpansionSettings,
) -> list[tuple[str, float]]:
    """Choose the terms that best characterise the feedback documents.

    Args:
        index: The collection the feedback documents belong to.
        feedback: Term counts of each feedback document.
        exclude: Terms that must not be returned: those already in the query.
        settings: How many terms to return and which ones qualify.

    Returns:
        (term, signed root LLR) pairs, strongest first. Only terms that are more
        frequent in the feedback documents than in the rest are returned.
    """
    counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    for document in feedback:
        counts.update(document)
        document_counts.update(document.keys())

    feedback_total = counts.total()
    rest_total = index.total_length - feedback_total
    scored = []
    for term, frequency in counts.items():
        if term in exclude or document_counts[term] < settings.min_feedback_documents:
            continue
        elsewhere = max(0, index.collection_frequency(term) - frequency)
        score = signed_root_llr(
            frequency, elsewhere, feedback_total - frequency, max(0, rest_total - elsewhere)
        )
        if score > 0:
            scored.append((term, score))
    # Alphabetical among equals, so that results do not depend on dictionary order.
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[: settings.terms]


def _entropy(*counts: int) -> float:
    """Unnormalised Shannon entropy of a list of counts."""
    return _x_log_x(sum(counts)) - sum(_x_log_x(count) for count in counts)


def _x_log_x(x: int) -> float:
    return x * math.log(x) if x > 0 else 0.0
