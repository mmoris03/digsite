# Decisions

Why things are the way they are. Each entry says what was decided, why, what
else was considered, and what would make it worth deciding again.

## 1. A word repeated in a query counts once

**Decision.** `search_and` and `search_or` ignore repetitions in the query:
`"oil oil"` finds the same documents as `"oil"`. The words of a query are
collected in a dict keyed by word (`_postings_of_query` in `digsite/index.py`),
so a repeated word is looked up once and its repetitions are not kept.

**Why.** Both searches ask a yes-or-no question about each document: does it
have every word of the query, or at least one? Asking twice about the same
word cannot change the answer (`p and p` is `p`, and so is `p or p`), so there
is nothing to count.

**Alternatives.** Keep how many times each word was typed, so that `"oil oil"`
weighs twice as much as `"oil"`. That only makes sense when results have a
score to weigh, and a boolean search has none: a document either matches or it
does not. Keeping the count now would be carrying data that nothing reads.

**Revisit when.** Results are ranked by a score and there is evidence, from a
measurement and not a hunch, that repeating a word should raise its weight.
Changing it means `_postings_of_query` also returns how many times each word
was typed. The current behaviour is pinned by the `a repeated word` cases in
`tests/test_index.py`.
