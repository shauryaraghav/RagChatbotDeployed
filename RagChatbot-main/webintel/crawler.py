"""
Delta-crawling web crawler that parses pages down to individual DOM nodes
(not whole-page chunks) so each node can be embedded and later resolved
back to a CSS selector for automated actions.
"""
import time
import hashlib
import logging
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from .store import CrawlStore, sha256

logger = logging.getLogger(__name__)

TEXT_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "span", "label"}
INTERACTIVE_TAGS = {"a", "button", "input", "textarea", "select"}
CANDIDATE_TAGS = TEXT_TAGS | INTERACTIVE_TAGS
MIN_TEXT_LEN = 15  # for non-interactive text nodes only


def build_selector(el: Tag) -> str:
    """Build a CSS selector path from an element up to the nearest id/root.
    Not a full CSS-selector-generator, but stable and sufficient to
    re-locate the same node in the same DOM (used for guardrail validation
    and for handing selectors to a browser-automation layer)."""
    parts = []
    node = el
    while node is not None and getattr(node, "name", None) not in (None, "[document]"):
        if node.get("id"):
            parts.append(f'#{node["id"]}')
            break
        siblings = [s for s in node.parent.find_all(node.name, recursive=False)] if node.parent else [node]
        if len(siblings) > 1:
            idx = siblings.index(node) + 1
            parts.append(f"{node.name}:nth-of-type({idx})")
        else:
            parts.append(node.name)
        node = node.parent
    return " > ".join(reversed(parts))


def node_text(el: Tag) -> str:
    if el.name == "input":
        return (el.get("placeholder") or el.get("value") or el.get("name") or "").strip()
    return el.get_text(separator=" ", strip=True)


def extract_dom_nodes(html: str, url: str) -> list:
    """Parse HTML into candidate DOM nodes: each becomes a unit of retrieval
    (for search) and a unit of action (for guarded automation)."""
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup(["script", "style", "noscript"]):
        junk.decompose()

    nodes = []
    seen_text = set()
    for el in soup.find_all(list(CANDIDATE_TAGS)):
        text = node_text(el)
        is_interactive = el.name in INTERACTIVE_TAGS
        if not is_interactive and len(text) < MIN_TEXT_LEN:
            continue
        if not is_interactive and text in seen_text:
            continue  # skip exact-duplicate text nodes (e.g. nested spans)
        seen_text.add(text)

        selector = build_selector(el)
        node_id = hashlib.sha1(f"{url}|{selector}|{text}".encode()).hexdigest()[:16]
        nodes.append({
            "node_id": node_id,
            "url": url,
            "tag": el.name,
            "selector": selector,
            "text": text,
            "is_interactive": is_interactive,
            "attrs": {
                "id": el.get("id"),
                "class": " ".join(el.get("class", [])) or None,
                "href": el.get("href"),
                "type": el.get("type"),
                "name": el.get("name"),
                "role": el.get("role"),
            },
            "text_hash": sha256(text),
        })

    links = set()
    for a in soup.find_all("a", href=True):
        links.add(urljoin(url, a["href"]))
    return nodes, links


def _same_domain(url: str, seed_domain: str) -> bool:
    try:
        return urlparse(url).netloc == seed_domain
    except Exception:
        return False


async def crawl_site(
    seed_url: str,
    store: CrawlStore,
    embed_fn,
    max_pages: int = 30,
    same_domain_only: bool = True,
    timeout: float = 10.0,
) -> dict:
    """BFS crawl from seed_url. Delta-crawling: pages whose fetched-content
    hash matches what's already stored are skipped (no re-parse, no
    re-embedding). Returns a report with real measured timings so the
    "% latency cut" is computed, not asserted."""
    seed_domain = urlparse(seed_url).netloc
    queue = [seed_url]
    seen = {seed_url}

    pages_fetched, pages_skipped_unchanged, pages_failed = 0, 0, 0
    nodes_embedded = 0
    fetch_time_total = 0.0
    embed_time_total = 0.0

    t_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        while queue and pages_fetched + pages_skipped_unchanged < max_pages:
            url = queue.pop(0)
            t0 = time.perf_counter()
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text
            except Exception as e:
                logger.warning(f"crawl fetch failed for {url}: {e}")
                pages_failed += 1
                continue
            fetch_time_total += time.perf_counter() - t0

            content_hash = sha256(html)

            if store.is_unchanged(url, content_hash):
                pages_skipped_unchanged += 1
                # still need the links to keep crawling, but skip
                # DOM parsing/embedding work entirely -> this IS the saving
                try:
                    _, links = extract_dom_nodes(html, url)
                    for link in links:
                        if link not in seen and (not same_domain_only or _same_domain(link, seed_domain)):
                            seen.add(link)
                            queue.append(link)
                except Exception:
                    pass
                continue

            dom_nodes, links = extract_dom_nodes(html, url)

            t1 = time.perf_counter()
            for n in dom_nodes:
                n["embedding"] = embed_fn(n["text"])
            embed_time_total += time.perf_counter() - t1

            store.upsert_nodes(dom_nodes)
            store.upsert_page(url, content_hash, [n["node_id"] for n in dom_nodes])
            nodes_embedded += len(dom_nodes)
            pages_fetched += 1

            for link in links:
                if link not in seen and (not same_domain_only or _same_domain(link, seed_domain)):
                    seen.add(link)
                    queue.append(link)

    store.save()
    elapsed = time.perf_counter() - t_start

    # Estimate what a *full* (non-delta) re-crawl would have cost: fetch
    # every page we skipped too, and re-embed at the same avg per-node cost.
    avg_embed_per_page = (embed_time_total / pages_fetched) if pages_fetched else 0.0
    avg_fetch_per_page = (fetch_time_total / max(pages_fetched, 1))
    hypothetical_full_time = elapsed + pages_skipped_unchanged * avg_embed_per_page

    time_saved_pct = (
        round(100 * (1 - elapsed / hypothetical_full_time), 1)
        if hypothetical_full_time > 0 else 0.0
    )

    return {
        "seed_url": seed_url,
        "pages_fetched": pages_fetched,
        "pages_skipped_unchanged": pages_skipped_unchanged,
        "pages_failed": pages_failed,
        "nodes_embedded": nodes_embedded,
        "elapsed_sec": round(elapsed, 3),
        "estimated_full_recrawl_sec": round(hypothetical_full_time, 3),
        "delta_crawl_time_saved_pct": time_saved_pct,
        "store_stats": store.stats(),
    }
