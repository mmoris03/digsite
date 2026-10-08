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

## 2. How digsite searches is something you can choose

**Decision.** digsite is meant to show how a search engine works inside, not
only to find things. So the way it searches is a setting: each way of
matching or ordering documents is kept as an option, and a new one is added
next to the ones that already exist instead of replacing them. Today the
options are finding the documents that have every word of the query
(`search_and`) or any of them (`search_or`).

**Why.** Running two approaches on the same documents with the same query,
and seeing where their results differ, explains each of them better than
either one alone. It also makes them easy to measure against each other.

**Alternatives.** A single, fixed way of searching: less code and a simpler
interface, but every improvement would erase the approach before it, and with
it the chance to compare them. Keeping the options has a cost too: each one
has to keep working and stay tested, and not every combination of settings
makes sense.

**Revisit when.** An option costs more to keep than it shows, for instance
one that never gives a result different from another option.
