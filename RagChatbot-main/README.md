---
title: RagChatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 10000
pinned: false
---

# 🤖 Site-Aware RAG Chatbot

An embeddable, retrieval-augmented generation (RAG) chatbot backend and widget system. Designed as a lightweight, site-grounded assistant, this application allows host websites to provide grounded, domain-specific Q&A by processing local document uploads or crawling live web content.

---

## 📌 Project Overview

Unlike generic LLMs that rely solely on pre-trained parametric knowledge or risk hallucinating details, this system retrieves answers directly from a host site's owned content. It parses, chunks, and indexes documents and web pages to surface exact context before invoking a language model for response generation.

---

## ⚡ Core Features

* **Hybrid Retrieval Engine:** Combines dense semantic vector search (via Sentence Transformers) with lexical keyword matching (via Scikit-learn cosine similarity) to ensure exact term matches and contextual relevance are captured simultaneously.
* **Document & URL Ingestion:** Parses structure and text from structured files (`.pdf`, `.html`) and live web URLs using BeautifulSoup4 and PyPDF2.
* **Context-Aware Chunking:** Intelligently splits document text into manageable segments to optimize context window utilization and reduce source degradation.
* **Local Embedding Persistence:** Saves generated vector representations and document chunks to disk to eliminate redundant ingestion overhead across sessions.
* **Source-Grounded Response Generation:** Leverages Google's **Gemini 2.0 Flash API** to generate precise, conversational responses strictly grounded in the retrieved context.
* **Embeddable Frontend Interface:** Built to interface via a lightweight web UI or embeddable widget layer alongside standard REST API endpoints.

---

## 🛠️ Architecture & Stack

| Component | Technology |
| :--- | :--- |
| **Backend Framework** | FastAPI (ASGI Server via Uvicorn) |
| **Language Model** | Google Gemini 2.0 Flash API |
| **Embeddings & Search** | Sentence Transformers, Scikit-learn |
| **Parsing Engine** | BeautifulSoup4, PyPDF2 |
| **Runtime Environment** | Python 3.11 |

---

## 🕸️ Web Intelligence Platform (`webintel/`)

A second, additive pipeline for crawling and acting on a whole live site rather than a single uploaded document. New endpoints, all prefixed `/webintel`:

| Endpoint | What it does |
| :--- | :--- |
| `POST /webintel/crawl` | BFS-crawls a site from a seed URL. **Delta-crawling**: each page's content is hashed; on re-crawl, unchanged pages are skipped entirely (no re-parse, no re-embedding). Response includes measured `elapsed_sec`, `estimated_full_recrawl_sec`, and `delta_crawl_time_saved_pct` — run it twice on the same site to see a real number, not an assumed one. |
| `GET /webintel/search` | **True hybrid search**: BM25 (`rank_bm25`) + dense cosine, min-max normalized and combined (`alpha` param), then **cross-encoder re-ranked** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the shortlist. |
| `POST /webintel/resolve_action` | Gives the LLM a shortlist of interactive DOM nodes (buttons, links, inputs — each with a CSS selector built from the parsed DOM) and asks it to resolve a natural-language instruction to one of them. **Guardrails are enforced in code, not by the prompt**: action must be in an allow-list, the selector must exactly match a node we actually offered (hallucinated selectors are rejected), the action must be valid for that tag, and `fill` values are length-capped. Returns a *plan* — nothing is auto-executed; wire the plan into Playwright/Selenium or the frontend to actually run it. |
| `POST /webintel/eval/generate` | Bootstraps a site-specific eval set by sampling indexed nodes and asking the LLM to write a question each node answers. |
| `GET /webintel/eval/mrr` | Runs the eval set through the hybrid+rerank pipeline and reports **Mean Reciprocal Rank** — a measured number, so if you cite an MRR figure it's one you actually got on your own site, not a copied figure. |

### Where each resume-bullet claim comes from, concretely
- *"DOM node to vector embeddings"* → `webintel/crawler.py::extract_dom_nodes` parses per-element (not per-page) and embeds each node individually, keeping its CSS selector.
- *"Delta-crawling"* → `CrawlStore.is_unchanged` (content-hash skip) in `webintel/crawler.py::crawl_site`; the % saved is computed from real elapsed time, printed in the crawl response.
- *"Hybrid search (BM25 + Dense) with cross-encoder re-ranking"* → `webintel/search.py`.
- *"MRR on site-specific evaluations"* → `webintel/eval.py`; run `/webintel/eval/generate` then `/webintel/eval/mrr` against your actual crawled site to get your own number.
- *"LLMs resolve DOM selectors and execute automated actions with guardrails"* → `webintel/actions.py::resolve_action`.

### Setup notes
- Add `rank_bm25` and `pydantic==1.10.13` (already in `requirements.txt`).
- The cross-encoder model downloads from Hugging Face on first use — needs normal internet access (it wasn't reachable in the sandbox this was built in, so it falls back to hybrid-only order if the download fails; test it in your real environment).
- `/webintel/resolve_action` returns a plan object, not a browser action. If you want it to *actually* click/fill, add Playwright and execute `plan["selector"]` / `plan["action"]` there — keep the guardrail checks server-side either way, don't trust a browser extension or frontend to re-validate them.
- There's a pre-existing bug in `app.py` I fixed while in there: `re` was used in `extract_key_entities` but never imported — added `import re, json` at the top.
- Heads up: `app.py` defines `/stream_query` **twice** (once with the basic prompt builder, once with the "enhanced" intent-aware one). FastAPI matches routes in registration order, so the first one always wins and the enhanced version is currently dead code — worth deciding which one you actually want and deleting the other.

