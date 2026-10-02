"""Search results as people see them: documents, each shown by its best passage."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DocumentResult:
    """A document found by a search.

    Attributes:
        url: Address of the document.
        title: Title of the document.
        section: Headings the best passage sits under, below the title. Empty
            for a passage at the top of the document.
        passage: The passage of the document that matched best.
        score: The score of that passage. Its scale depends on the search mode.
        chunk_id: The chunk the passage is.
    """

    url: str
    title: str
    section: str
    passage: str
    score: float
    chunk_id: int


def section_of(context: str, title: str) -> str:
    """The headings of a passage below the document's title.

    Args:
        context: The full heading path of the passage, e.g. 'Logging > Handlers'.
        title: The title of its document, which the path usually starts with.
    """
    return context.removeprefix(title).removeprefix(" > ")
