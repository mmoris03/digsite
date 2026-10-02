"""The graph of links between documents, derived from the links of the crawl."""

from collections.abc import Collection, Iterable, Mapping

from digsite.models import Edge


def document_links(
    edges: Iterable[Edge],
    documents: Collection[str],
    *,
    redirects: Mapping[str, str] | None = None,
    duplicates: Mapping[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Reduce the links between pages to links between documents.

    The crawl records every link of every page. What says something about the
    importance of a document is narrower: links from one indexed document to a
    different one. So a link is followed through redirects to the page it ends
    at; a link from or to a duplicate counts for the document that stands for
    it; and links that leave the corpus, links of a document to itself and
    links marked `nofollow`, which the author does not vouch for, are dropped.

    Args:
        edges: Links between pages, as recorded by the crawl.
        documents: URLs of the documents that are indexed.
        redirects: For each URL that redirects, where it redirects to.
        duplicates: For each duplicate URL, the URL of the document that stands for it.

    Returns:
        The distinct (source, target) links between documents, sorted.
    """
    redirects = redirects or {}
    duplicates = duplicates or {}

    def document_of(url: str) -> str | None:
        seen = {url}
        while (target := redirects.get(url)) is not None and target not in seen:
            seen.add(target)
            url = target
        url = duplicates.get(url, url)
        return url if url in documents else None

    links = set()
    for edge in edges:
        if edge.nofollow:
            continue
        source = document_of(edge.source)
        target = document_of(edge.target)
        if source is not None and target is not None and source != target:
            links.add((source, target))
    return sorted(links)
