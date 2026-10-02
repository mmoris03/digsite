"use strict";

const form = document.getElementById("form");
const queryInput = document.getElementById("query");
const modeSelect = document.getElementById("mode");
const output = document.getElementById("output");
const statusLine = document.getElementById("status");
const collectionBar = document.getElementById("collection-bar");
const collectionSelect = document.getElementById("collection");
const addToggle = document.getElementById("add-toggle");
const addForm = document.getElementById("add-form");
const buildsBox = document.getElementById("builds");

let languageModel = "";
let collections = []; // as /api/collections lists them
let selected = ""; // id of the collection searched
const watched = new Set(); // builds asked for or seen running while this page is open
const dismissed = new Set(); // builds no longer to show: failures closed, websites announced
let pollTimer = null;
let current = null; // AbortController of the search or question in progress
let currentKey = ""; // the query string of what is shown
let timer = null;

// Builds an element. Strings among the children become text, never markup.
function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    node.setAttribute(name, value === true ? "" : String(value));
  }
  node.append(...children.flat().filter((child) => child !== null && child !== undefined));
  return node;
}

// Only web addresses become links: a website could hold anything.
function link(url, ...children) {
  if (!/^https?:\/\//i.test(url)) return el("span", {}, ...children);
  return el("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, ...children);
}

function message(text, kind = "") {
  return el("p", { class: `message ${kind}`.trim() }, text);
}

const count = (number) => number.toLocaleString("en");

async function getJSON(url, options = {}) {
  const response = await fetch(url, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    // Not JSON: the status code is all there is.
  }
  if (!response.ok) {
    const detail = body && body.detail;
    if (typeof detail === "string") throw new Error(detail);
    if (Array.isArray(detail)) throw new Error(detail.map((item) => item.msg).join("; "));
    throw new Error(`The server answered ${response.status}.`);
  }
  return body;
}

function stopTimer() {
  if (timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}

// --- Websites

async function refreshLibrary() {
  clearTimeout(pollTimer);
  let data;
  try {
    data = await getJSON("api/collections");
  } catch (error) {
    statusLine.textContent = `The server is not answering: ${error.message}`;
    pollTimer = setTimeout(refreshLibrary, 5000);
    return;
  }
  collections = data.collections;
  const builds = data.builds;
  for (const build of builds) {
    if (build.stage !== "done" && build.stage !== "failed") watched.add(build.id);
  }
  // A website added while the page is open becomes the one searched when it is ready.
  const ready = builds.find((build) => build.stage === "done" && watched.has(build.id)
    && !dismissed.has(`done:${build.collection}`)
    && collections.some((collection) => collection.id === build.collection));
  renderCollections(ready ? ready.collection : null);
  renderBuilds(builds);
  if (builds.some((build) => build.stage !== "done" && build.stage !== "failed")) {
    pollTimer = setTimeout(refreshLibrary, 1500);
  }
}

function renderCollections(newlyReady) {
  collectionSelect.replaceChildren(...collections.map((collection) =>
    el("option", { value: collection.id }, collection.title)));
  const empty = collections.length === 0;
  collectionBar.hidden = empty;
  form.hidden = empty;
  if (empty) {
    selected = "";
    showAddForm(true);
    statusLine.textContent = "No website yet: add one to start.";
    return;
  }
  const keep = collections.some((collection) => collection.id === selected);
  const choice = newlyReady || (keep ? selected : collections[0].id);
  if (newlyReady) dismissed.add(`done:${newlyReady}`);
  selectCollection(choice, newlyReady !== null && newlyReady !== selected);
}

function selectCollection(id, announce = false) {
  const collection = collections.find((item) => item.id === id);
  if (!collection) return;
  const changed = id !== selected;
  selected = id;
  collectionSelect.value = id;
  statusLine.replaceChildren(
    `${count(collection.documents)} documents · ${count(collection.chunks)} passages`,
    collection.source ? " · from " : "",
    collection.source ? link(collection.source, collection.source.replace(/^https?:\/\//, "")) : "",
    languageModel ? ` · answers by ${languageModel}` : "",
  );
  if (!collection.index_up_to_date) {
    statusLine.append(el("span", { class: "warning" }, " · the index is out of date"));
  }
  const previous = modeSelect.value;
  modeSelect.replaceChildren(...collection.modes.map((mode) => el("option", { value: mode }, mode)));
  modeSelect.value = collection.modes.includes(previous) && !changed ? previous : collection.default_mode;
  if (changed && announce) {
    output.replaceChildren(message(`${collection.title} is ready to search.`));
  }
}

function showAddForm(open) {
  addForm.hidden = !open;
  addToggle.setAttribute("aria-expanded", String(open));
  if (open && collections.length > 0) document.getElementById("add-url").focus();
}

function renderBuilds(builds) {
  const shown = builds.filter((build) => watched.has(build.id) && !dismissed.has(build.id)
    && !(build.stage === "done" && dismissed.has(`done:${build.collection}`)));
  buildsBox.replaceChildren(...shown.map(buildCard));
}

function buildCard(build) {
  const address = build.url.replace(/^https?:\/\//, "");
  const parts = [el("div", { class: "build-title" }, link(build.url, address))];
  if (build.stage === "failed") {
    parts.push(el("p", { class: "error" }, `Could not add it: ${build.error}`));
    const close = el("button", { type: "button", class: "secondary small-button" }, "Dismiss");
    close.addEventListener("click", () => {
      dismissed.add(build.id);
      refreshLibrary();
    });
    parts.push(close);
    return el("div", { class: "build failed" }, parts);
  }
  const elapsed = Math.max(0, Math.round(Date.now() / 1000 - build.submitted));
  parts.push(el("p", { class: "small" }, `${stageText(build)} · ${formatDuration(elapsed)}`));
  const bar = el("progress", { max: build.total || 1, "aria-label": "Progress" });
  if (build.total) bar.value = Math.min(build.done, build.total);
  else bar.removeAttribute("value");
  parts.push(bar);
  return el("div", { class: "build" }, parts);
}

function stageText(build) {
  switch (build.stage) {
    case "queued":
      return "Waiting to start";
    case "crawling":
      return `Downloading pages: ${count(build.done)} of at most ${count(build.total)}`;
    case "extracting":
      return `Extracting text: ${count(build.done)} of ${count(build.total)} pages`;
    case "indexing":
      return build.total
        ? `Indexing: ${count(build.done)} of ${count(build.total)} passages embedded`
        : "Indexing: splitting the pages into passages";
    default:
      return build.stage;
  }
}

function formatDuration(seconds) {
  if (seconds < 60) return `${seconds} s`;
  return `${Math.floor(seconds / 60)} min ${String(seconds % 60).padStart(2, "0")} s`;
}

async function addWebsite(event) {
  event.preventDefault();
  const button = addForm.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const build = await getJSON("api/collections", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: document.getElementById("add-url").value.trim(),
        max_pages: Number(document.getElementById("add-pages").value),
        language: document.getElementById("add-language").value || null,
      }),
    });
    watched.add(build.id);
    addForm.reset();
    setDefaultLanguage();
    if (collections.length > 0) showAddForm(false);
    await refreshLibrary();
  } catch (error) {
    buildsBox.prepend(el("p", { class: "message error" }, error.message));
  } finally {
    button.disabled = false;
  }
}

function setDefaultLanguage() {
  const language = (navigator.language || "").slice(0, 2);
  document.getElementById("add-language").value = language === "es" ? "spanish" : "english";
}

// --- Searching and asking

function run(action, query, mode, remember) {
  if (!query || !selected) return;
  if (current) current.abort();
  stopTimer();
  current = new AbortController();

  const params = new URLSearchParams({ c: selected, q: query, mode });
  if (action === "ask") params.set("ask", "1");
  currentKey = params.toString();
  if (remember) history.pushState(null, "", `?${currentKey}`);
  document.title = `${query} – Digsite`;

  const task = action === "ask"
    ? ask(selected, query, mode, current.signal)
    : search(selected, query, mode, current.signal);
  task.catch((error) => {
    if (error.name === "AbortError") return;
    stopTimer();
    output.replaceChildren(message(error.message, "error"));
  });
}

async function search(collection, query, mode, signal) {
  output.replaceChildren(message("Searching…"));
  const params = new URLSearchParams({ collection, q: query, mode, limit: "10" });
  const data = await getJSON(`api/search?${params}`, { signal });
  if (data.results.length === 0) {
    output.replaceChildren(message("No document matches."));
    return;
  }
  output.replaceChildren(
    el("p", { class: "muted small" }, `${data.results.length} documents · ${data.mode} search`),
    el("ol", { class: "results" }, data.results.map(resultItem)),
  );
}

function resultItem(result) {
  return el("li", { class: "result" },
    el("h2", {}, link(result.url, result.title || result.url)),
    el("div", { class: "url" }, result.url),
    result.section ? el("div", { class: "section" }, result.section) : null,
    // A snippet is a few lines of text: code fences would only take room.
    el("p", { class: "passage" }, inline(result.passage.replace(/^```.*$/gm, "").trim())),
  );
}

// --- Ask

async function ask(collection, query, mode, signal) {
  const started = Date.now();
  const progress = message("");
  const tick = () => {
    const seconds = Math.round((Date.now() - started) / 1000);
    progress.textContent =
      `Reading the question, searching and writing the answer… ${seconds} s. ` +
      "On a computer without a GPU this takes one to two minutes.";
  };
  tick();
  timer = setInterval(tick, 1000);
  output.replaceChildren(progress);

  const data = await getJSON("api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection, question: query, mode }),
    signal,
  });
  stopTimer();
  output.replaceChildren(...answerView(data));
}

function answerView(data) {
  const byNumber = new Map(data.passages.map((passage) => [passage.number, passage]));
  const cited = data.cited.map((number) => byNumber.get(number)).filter(Boolean);
  const others = data.passages.filter((passage) => !data.cited.includes(passage.number));
  const parts = [];

  if (data.answered) {
    parts.push(el("article", { class: "answer" }, formatText(data.answer, byNumber)));
    if (cited.length > 0) {
      parts.push(el("h2", {}, "Sources"), el("ul", { class: "passages" }, cited.map(passageItem)));
    } else {
      parts.push(message("The answer cites none of the passages it was written from.", "warning"));
    }
  } else {
    parts.push(message("The documents do not answer this question.", "warning"));
  }
  if (others.length > 0) {
    const label = data.answered
      ? `Other passages given to the model (${others.length})`
      : `Passages that were found (${others.length})`;
    parts.push(el("details", { open: !data.answered },
      el("summary", {}, label),
      el("ul", { class: "passages" }, others.map(passageItem))));
  }
  parts.push(el("p", { class: "muted small" },
    `Searched for: ${data.queries.join(" · ")} — ${Math.round(data.seconds)} s`));
  return parts;
}

function passageItem(passage) {
  const where = [passage.title, passage.section].filter(Boolean).join(" › ");
  return el("li", { id: `passage-${passage.number}`, class: "passage-card" },
    el("div", { class: "where" },
      el("span", { class: "number" }, `[${passage.number}]`), " ", link(passage.url, where)),
    el("div", { class: "passage" }, formatText(passage.text)),
  );
}

// Answers and passages are plain text with some Markdown: code blocks and `inline code`.
// Answers also cite passages, as in [1] or [1][2]; given the passages by number, the
// citations become links to them. Without, brackets are left alone: in a passage,
// [1, 2] is a list.
function formatText(text, byNumber = null) {
  const blocks = [];
  text.split("```").forEach((part, index) => {
    if (index % 2 === 1) {
      const code = part.replace(/^[\w+-]*\n/, "").replace(/\n$/, "");
      blocks.push(el("pre", {}, el("code", {}, code)));
      return;
    }
    for (const paragraph of part.split(/\n\s*\n/)) {
      if (paragraph.trim()) blocks.push(el("p", {}, inline(paragraph.trim(), byNumber)));
    }
  });
  return blocks;
}

// Same rule as the server: a bracket right after a name or a bracket is code, as in items[0].
const CITATIONS = /(?<![\w)\]])(?:\[\d+(?:\s*,\s*\d+)*\])+/g;

function inline(text, byNumber = null) {
  const nodes = [];
  text.split(/(`[^`\n]+`)/).forEach((piece, index) => {
    if (index % 2 === 1) {
      nodes.push(el("code", {}, piece.slice(1, -1)));
      return;
    }
    if (byNumber === null) {
      nodes.push(piece);
      return;
    }
    let last = 0;
    for (const match of piece.matchAll(CITATIONS)) {
      nodes.push(piece.slice(last, match.index));
      for (const digits of match[0].match(/\d+/g)) {
        const number = Number(digits);
        nodes.push(byNumber.has(number)
          ? el("a", { href: `#passage-${number}`, class: "cite", "data-passage": number }, `[${number}]`)
          : `[${number}]`);
      }
      last = match.index + match[0].length;
    }
    nodes.push(piece.slice(last));
  });
  return nodes;
}

// A citation scrolls to its passage without touching the address, which holds the query.
output.addEventListener("click", (event) => {
  const citation = event.target.closest("a.cite");
  if (!citation) return;
  event.preventDefault();
  const card = document.getElementById(`passage-${citation.dataset.passage}`);
  if (!card) return;
  const details = card.closest("details");
  if (details) details.open = true;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1500);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const action = event.submitter ? event.submitter.value : "search";
  run(action, queryInput.value.trim(), modeSelect.value, true);
});

collectionSelect.addEventListener("change", () => {
  if (current) current.abort();
  stopTimer();
  selectCollection(collectionSelect.value);
  output.replaceChildren();
  currentKey = new URLSearchParams({ c: selected }).toString();
  history.pushState(null, "", `?${currentKey}`);
});

addToggle.addEventListener("click", () => showAddForm(addForm.hidden));
addForm.addEventListener("submit", addWebsite);

// The address holds the website and the query, so that a search can be reloaded,
// shared or gone back to.
function runFromAddress() {
  const params = new URLSearchParams(location.search);
  const wanted = params.get("c");
  if (wanted && wanted !== selected) selectCollection(wanted);
  if (params.toString() === currentKey) return;
  const query = params.get("q") || "";
  const mode = params.get("mode");
  queryInput.value = query;
  if (mode && [...modeSelect.options].some((option) => option.value === mode)) {
    modeSelect.value = mode;
  }
  if (query) {
    run(params.get("ask") === "1" ? "ask" : "search", query, modeSelect.value, false);
  } else {
    currentKey = params.toString();
    output.replaceChildren();
  }
}

async function start() {
  setDefaultLanguage();
  try {
    languageModel = (await getJSON("api/status")).language_model;
  } catch {
    // The list of websites says what is wrong.
  }
  await refreshLibrary();
  runFromAddress();
}

window.addEventListener("popstate", runFromAddress);
start();
