"""
Persistent storage for the web-intelligence crawler.

Design note: delta-crawling only works if we can cheaply answer "has this
page changed since I last indexed it?" without re-embedding everything.
We do that by keeping a content hash per URL. On re-crawl, a page whose
hash is unchanged is skipped entirely (no fetch of children work, no
re-embedding) -- that's where the latency win comes from.
"""
import json
import time
import hashlib
from pathlib import Path
from typing import Optional


class CrawlStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.pages: dict = {}   # url -> {content_hash, last_crawled, node_ids}
        self.nodes: dict = {}   # node_id -> node dict (text, embedding, selector, ...)
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.pages = data.get("pages", {})
                self.nodes = data.get("nodes", {})
            except Exception:
                self.pages, self.nodes = {}, {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"pages": self.pages, "nodes": self.nodes}, f)

    def page_hash(self, url: str) -> Optional[str]:
        return self.pages.get(url, {}).get("content_hash")

    def is_unchanged(self, url: str, content_hash: str) -> bool:
        return self.page_hash(url) == content_hash

    def upsert_page(self, url: str, content_hash: str, node_ids: list):
        self.pages[url] = {
            "content_hash": content_hash,
            "last_crawled": time.time(),
            "node_ids": node_ids,
        }

    def upsert_nodes(self, nodes: list):
        for n in nodes:
            self.nodes[n["node_id"]] = n

    def all_nodes(self) -> list:
        return list(self.nodes.values())

    def clear(self):
        self.pages, self.nodes = {}, {}
        self.save()

    def stats(self) -> dict:
        return {
            "pages_indexed": len(self.pages),
            "nodes_indexed": len(self.nodes),
        }


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
