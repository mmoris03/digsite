"use strict";

const form = document.getElementById("form");
const queryInput = document.getElementById("query");
const modeSelect = document.getElementById("mode");
const output = document.getElementById("output");
const statusLine = document.getElementById("status");

let current = null; // AbortController of the request in progress
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

// Only web addresses become links: a corpus could hold anything.
function link(url, ...children) {
  if (!/^https?:\/\//i.test(url)) return el("span", {}, ...children);
  return el("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, ...children);
}

function message(text, kind = "") {
  return el("p", { class: `message ${kind}`.trim() }, text);
}

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

async function loadStatus() {
  try {
    const status = await getJSON("api/status");
    const count = (number) => number.toLocaleString("en");
    statusLine.textContent =
      `${count(status.documents)} documents · ${count(status.chunks)} passages · ` +
      `answers by ${status.language_model}`;
    if (!status.index_up_to_date) {
      statusLine.append(el("span", { class: "warning" }, " · the index is out of date"));
    }
    modeSelect.replaceChildren(
      ...status.modes.map((mode) =>
        el("option", { value: mode, selected: mode === status.default_mode }, mode)),
    );
  } catch (error) {
    statusLine.textContent = `The server is not answering: ${error.message}`;
  }
}

function run(action, query, mode, remember) {
  if (!query) return;
  if (current) current.abort();
  stopTimer();
  current = new AbortController();

  const params = new URLSearchParams({ q: query, mode });
  if (action === "ask") params.set("ask", "1");
  currentKey = params.toString();
  if (remember) history.pushState(null, "", `?${currentKey}`);
  document.title = `${query} – Digsite`;

  const task = action === "ask" ? ask(query, mode, current.signal) : search(query, mode, current.signal);
  task.catch((error) => {
    if (error.name === "AbortError") return;
    stopTimer();
    output.replaceChildren(message(error.message, "error"));
  });
}

// --- Search

async function search(query, mode, signal) {
  output.replaceChildren(message("Searching…"));
  const params = new URLSearchParams({ q: query, mode, limit: "10" });
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

async function ask(query, mode, signal) {
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
    body: JSON.stringify({ question: query, mode }),
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

// The address holds the query, so that a search can be reloaded, shared or gone back to.
function runFromAddress() {
  const params = new URLSearchParams(location.search);
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
    currentKey = "";
    output.replaceChildren();
  }
}

window.addEventListener("popstate", runFromAddress);
loadStatus().then(runFromAddress);
