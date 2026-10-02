"""Test doubles: an in-memory website, a scripted language model and a fake clock."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import httpx

HTML = {"content-type": "text/html; charset=utf-8"}

type Route = httpx.Response | Callable[[httpx.Request], httpx.Response]


def html_page(*hrefs: str, title: str = "Page", head: str = "") -> httpx.Response:
    """An HTML response with one link per href."""
    anchors = "".join(f'<a href="{href}">link to {href}</a>' for href in hrefs)
    body = f"<html><head><title>{title}</title>{head}</head><body>{anchors}</body></html>"
    return httpx.Response(200, headers=HTML, content=body.encode())


def article_html(heading: str, *paragraphs: str) -> str:
    """A realistic page: an article surrounded by navigation and a footer."""
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return f"""<!DOCTYPE html>
<html lang="en">
<head><title>{heading} — Example Docs</title></head>
<body>
  <nav><a href="/">Home</a> <a href="/docs/">Documentation</a> <a href="/pricing">Pricing</a></nav>
  <main><article><h1>{heading}</h1>{body}</article></main>
  <footer>Copyright 2026 Example Corporation. All rights reserved.</footer>
</body>
</html>"""


class FakeSite:
    """Serves canned responses by URL and records what was requested."""

    def __init__(self, routes: dict[str, Route], robots: str | None = None) -> None:
        self.routes = routes
        self.robots = robots
        self.requested: list[str] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requested.append(url)
        if request.url.path == "/robots.txt" and url not in self.routes:
            if self.robots is None:
                return httpx.Response(404)
            return httpx.Response(200, text=self.robots)
        route = self.routes.get(url)
        if route is None:
            return httpx.Response(404, headers=HTML, text="not found")
        if callable(route):
            return route(request)
        # httpx responses are single-use, so serve a copy.
        return httpx.Response(route.status_code, headers=route.headers, content=route.content)

    @property
    def pages_requested(self) -> list[str]:
        """Requested URLs, robots.txt files excluded."""
        return [url for url in self.requested if not url.endswith("/robots.txt")]


@dataclass(frozen=True, slots=True)
class ModelCall:
    """One request to a scripted model."""

    prompt: str
    system: str
    schema: Mapping[str, object] | None


class ScriptedModel:
    """A language model that replies what the test says, and records what it was asked.

    Args:
        replies: The replies to give, in order. A reply that is not a string is
            sent as JSON. An exception is raised instead of returned.
    """

    name = "scripted"

    def __init__(self, *replies: object) -> None:
        self._replies = list(replies)
        self.calls: list[ModelCall] = []

    def generate(
        self, prompt: str, *, system: str = "", schema: Mapping[str, object] | None = None
    ) -> str:
        self.calls.append(ModelCall(prompt, system, schema))
        if not self._replies:
            raise AssertionError(f"the model was asked more than it was scripted for: {prompt!r}")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


class FakeClock:
    """A clock that only advances when someone sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
