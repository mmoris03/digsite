from digsite.ingest.content import extract_content

URL = "https://example.com/docs/caching.html"

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head><title>Caching — Example Docs 2.1</title></head>
<body>
  <nav class="navbar">
    <a href="/">Home</a> <a href="/docs/">Documentation</a> <a href="/pricing">Pricing plans</a>
  </nav>
  <div class="sidebar">
    <ul><li><a href="/docs/install.html">Installation guide</a></li>
        <li><a href="/docs/config.html">Configuration reference</a></li></ul>
  </div>
  <main>
    <article>
      <h1>Caching<a class="headerlink" href="#caching">¶</a></h1>
      <p>The cache stores the result of expensive computations so that repeated requests
      can be answered without doing the work again. Every entry is identified by a key
      and expires after a configurable amount of time.</p>
      <h2>Eviction policy<a class="headerlink" href="#eviction">¶</a></h2>
      <p>When the cache is full, the entry that was used least recently is removed first.
      This policy works well when recent requests are a good predictor of future ones,
      which is the common case for web traffic.</p>
      <p>You can inspect the state of the cache with the <code>stats</code> command, which
      reports the number of hits, misses and evictions since the process started.</p>
    </article>
  </main>
  <footer>Copyright 2026 Example Corporation. All rights reserved.</footer>
</body>
</html>
"""


def test_keeps_the_main_content_and_drops_the_boilerplate() -> None:
    content = extract_content(PAGE, URL)

    assert content is not None
    assert "used least recently is removed first" in content.text
    assert "hits, misses and evictions" in content.text
    assert "Pricing plans" not in content.text
    assert "Installation guide" not in content.text
    assert "All rights reserved" not in content.text


def test_keeps_headings_as_markdown_without_permalink_marks() -> None:
    content = extract_content(PAGE, URL)

    assert content is not None
    lines = content.text.splitlines()
    assert lines[0] == "# Caching"
    assert "## Eviction policy" in lines
    assert "¶" not in content.text


def test_title_is_the_first_top_level_heading() -> None:
    content = extract_content(PAGE, URL)

    assert content is not None
    assert content.title == "Caching"


def test_title_is_empty_when_there_is_no_top_level_heading() -> None:
    page = PAGE.replace('<h1>Caching<a class="headerlink" href="#caching">¶</a></h1>', "")

    content = extract_content(page, URL)

    assert content is not None
    assert content.title == ""


def test_title_drops_inline_markup() -> None:
    page = PAGE.replace("<h1>Caching<", "<h1><code>cache</code> — Caching<")

    content = extract_content(page, URL)

    assert content is not None
    assert content.title == "cache — Caching"


def test_a_page_without_text_yields_nothing() -> None:
    assert extract_content("<html><body><div><img src='logo.png'></div></body></html>", URL) is None
    assert extract_content("", URL) is None
