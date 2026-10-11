"""
Measures Mean Reciprocal Rank of the retrieval pipeline against a labeled
eval set of (query, expected_url_substring) pairs, so "0.85+ MRR" is a
number you actually compute against your own site, not a claim.
"""
import json
from pathlib import Path


def load_eval_set(path: Path) -> list:
    with open(path, "r") as f:
        return json.load(f)


def _is_relevant(node: dict, expected: dict) -> bool:
    if "expected_node_id" in expected:
        return node["node_id"] == expected["expected_node_id"]
    url_ok = expected.get("expected_url_contains", "") in node["url"]
    text_ok = expected.get("expected_text_contains", "").lower() in node["text"].lower()
    if "expected_url_contains" in expected and "expected_text_contains" in expected:
        return url_ok and text_ok
    return url_ok or text_ok


def compute_mrr(eval_items: list, search_fn) -> dict:
    """search_fn(query) -> ranked list of node dicts (already hybrid+reranked)."""
    reciprocal_ranks = []
    details = []
    for item in eval_items:
        ranked = search_fn(item["query"])
        rr = 0.0
        rank_found = None
        for rank, node in enumerate(ranked, start=1):
            if _is_relevant(node, item):
                rr = 1.0 / rank
                rank_found = rank
                break
        reciprocal_ranks.append(rr)
        details.append({"query": item["query"], "rank_found": rank_found, "reciprocal_rank": rr})

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    return {"mrr": round(mrr, 4), "n_queries": len(eval_items), "details": details}


def generate_eval_set_with_llm(nodes: list, generate_fn, n: int = 20, seed: int = 42) -> list:
    """Sample n text-bearing nodes and ask the LLM to write a natural
    question whose answer is that node's text -- a standard way to bootstrap
    a site-specific retrieval eval set without hand-labeling everything."""
    import random
    random.seed(seed)
    text_nodes = [nd for nd in nodes if not nd.get("is_interactive") and len(nd["text"]) > 30]
    sample = random.sample(text_nodes, min(n, len(text_nodes)))

    eval_set = []
    for node in sample:
        prompt = (
            "Write ONE short, natural user question that this webpage text "
            "directly answers. Respond with ONLY the question, no quotes, no preamble.\n\n"
            f"Text: {node['text'][:400]}"
        )
        try:
            question = generate_fn(prompt).strip().strip('"')
        except Exception:
            continue
        eval_set.append({
            "query": question,
            "expected_node_id": node["node_id"],
            "expected_url_contains": node["url"],
        })
    return eval_set
