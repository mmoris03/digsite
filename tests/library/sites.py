"""A small website to build collections from in tests."""

import httpx

from fakes import HTML, FakeSite, article_html

SITE = "https://docs.example.com"
GUIDE = f"{SITE}/guide/"


def html(text: str) -> httpx.Response:
    return httpx.Response(200, headers=HTML, content=text.encode())


def guide_site() -> FakeSite:
    """A guide of three pages under /guide/, linking to a page outside it."""
    home = """<!DOCTYPE html><html lang="en"><head><title>Example Guide</title></head>
<body><main><h1>Example Guide</h1>
<p>This guide explains how the service stores data and how it records what happens.</p>
<a href="caching.html">Caching</a> <a href="logging.html">Logging</a>
<a href="/blog/news.html">News</a></main></body></html>"""
    return FakeSite(
        {
            GUIDE: html(home),
            f"{GUIDE}caching.html": html(
                article_html(
                    "Caching",
                    "The cache stores the result of expensive computations so that repeated "
                    "requests are answered without doing the work again.",
                    "When the cache is full, the entry that was used least recently is evicted.",
                )
            ),
            f"{GUIDE}logging.html": html(
                article_html(
                    "Logging",
                    "Every request is written to the access log with its duration and status.",
                    "Logs are rotated daily and kept for thirty days by default.",
                )
            ),
            f"{SITE}/blog/news.html": html(
                article_html("News", "The blog is not part of the guide.")
            ),
        }
    )
