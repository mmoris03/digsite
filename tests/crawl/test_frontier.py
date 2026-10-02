from digsite.crawl.frontier import Frontier
from digsite.crawl.urls import Scope

SITE = "https://example.com"
SCOPE = Scope((f"{SITE}/",))


def drain(frontier: Frontier) -> list[tuple[str, int]]:
    return [frontier.pop() for _ in range(len(frontier))]


def test_yields_urls_in_insertion_order() -> None:
    frontier = Frontier(SCOPE, max_depth=5)

    for path in ("a", "b", "c"):
        assert frontier.add(f"{SITE}/{path}", 1)

    assert drain(frontier) == [(f"{SITE}/a", 1), (f"{SITE}/b", 1), (f"{SITE}/c", 1)]


def test_accepts_each_url_once_even_after_it_was_popped() -> None:
    frontier = Frontier(SCOPE, max_depth=5)
    frontier.add(f"{SITE}/a", 0)
    frontier.pop()

    assert not frontier.add(f"{SITE}/a", 3)
    assert len(frontier) == 0


def test_rejects_urls_seen_by_a_previous_crawl() -> None:
    frontier = Frontier(SCOPE, max_depth=5, seen=[f"{SITE}/old"])

    assert not frontier.add(f"{SITE}/old", 0)
    assert frontier.add(f"{SITE}/new", 0)


def test_rejects_urls_out_of_scope_too_deep_or_not_pages() -> None:
    frontier = Frontier(SCOPE, max_depth=2)

    assert not frontier.add("https://other.org/", 1)
    assert not frontier.add(f"{SITE}/deep", 3)
    assert not frontier.add(f"{SITE}/manual.pdf", 1)
    assert frontier.add(f"{SITE}/at-the-limit", 2)


def test_a_rejected_url_can_be_added_later_at_a_valid_depth() -> None:
    frontier = Frontier(SCOPE, max_depth=2)
    frontier.add(f"{SITE}/page", 3)

    assert frontier.add(f"{SITE}/page", 2)


def test_first_puts_the_url_at_the_front() -> None:
    frontier = Frontier(SCOPE, max_depth=5)
    frontier.add(f"{SITE}/a", 1)
    frontier.add(f"{SITE}/b", 1)

    frontier.add(f"{SITE}/urgent", 1, first=True)

    assert frontier.pop() == (f"{SITE}/urgent", 1)


def test_is_falsy_when_empty() -> None:
    frontier = Frontier(SCOPE, max_depth=5)

    assert not frontier
    frontier.add(f"{SITE}/a", 0)
    assert frontier
