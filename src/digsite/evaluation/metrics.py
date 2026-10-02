"""Evaluation metrics."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SetMetrics:
    """Agreement between a predicted set and the set that is actually correct."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """Fraction of the predictions that are correct."""
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        """Fraction of the correct items that were predicted."""
        relevant = self.true_positives + self.false_negatives
        return self.true_positives / relevant if relevant else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def compare_sets[T](truth: set[T], predicted: set[T]) -> SetMetrics:
    """Score a predicted set against the ground truth."""
    return SetMetrics(
        true_positives=len(truth & predicted),
        false_positives=len(predicted - truth),
        false_negatives=len(truth - predicted),
    )
