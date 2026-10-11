"""
Lets the LLM resolve a natural-language instruction ("open pricing", "search
for X") to a concrete DOM node from the crawled+indexed site, and propose an
action -- but the LLM never gets to invent a selector or an action type.

Guardrails (all enforced server-side, independent of what the LLM outputs):
  1. action must be in ALLOWED_ACTIONS
  2. selector must exactly match one of the candidate nodes we handed the
     LLM in this call (no hallucinated selectors reach the caller)
  3. action must be compatible with that node's tag (can't "fill" a <button>)
  4. "fill" requires a non-empty value; free-text is capped in length
  5. every resolution is logged for audit; nothing is auto-executed here --
     this returns a *plan* that a browser-automation layer (e.g. Playwright)
     or the frontend must explicitly execute
"""
import json
import logging

logger = logging.getLogger(__name__)

ALLOWED_ACTIONS = {"click", "fill", "navigate", "none"}
ACTION_TAG_COMPAT = {
    "click": {"a", "button", "input"},
    "fill": {"input", "textarea", "select"},
    "navigate": {"a"},
    "none": set(),  # always allowed
}
MAX_FILL_VALUE_LEN = 500


def build_action_prompt(instruction: str, candidates: list) -> str:
    catalog = [
        {"selector": c["selector"], "tag": c["tag"], "text": c["text"][:120],
         "attrs": c.get("attrs", {})}
        for c in candidates
    ]
    return f"""You are resolving a user instruction to ONE element on a webpage.

User instruction: "{instruction}"

Candidate elements (JSON, choose ONLY from this exact list, do not invent selectors):
{json.dumps(catalog, indent=2)}

Respond with ONLY a JSON object, no prose, no markdown fences:
{{"selector": "<one selector copied exactly from the list above, or null if none fit>",
  "action": "<one of: click, fill, navigate, none>",
  "value": "<text to fill, or null>",
  "reasoning": "<one short sentence>"}}"""


def _parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    return json.loads(raw)


def resolve_action(instruction: str, candidates: list, generate_fn) -> dict:
    """generate_fn: callable(prompt: str) -> str, e.g. wraps the Gemini call
    already used elsewhere in the app."""
    if not candidates:
        return {"status": "blocked", "reason": "no candidate elements provided"}

    prompt = build_action_prompt(instruction, candidates)
    try:
        raw = generate_fn(prompt)
        proposal = _parse_llm_json(raw)
    except Exception as e:
        logger.error(f"action resolution LLM call/parse failed: {e}")
        return {"status": "blocked", "reason": f"could not parse model output: {e}"}

    action = proposal.get("action")
    selector = proposal.get("selector")
    value = proposal.get("value")

    # Guardrail 1: action allow-list
    if action not in ALLOWED_ACTIONS:
        return {"status": "blocked", "reason": f"action '{action}' not in allow-list", "raw": proposal}

    if action == "none" or selector is None:
        return {"status": "no_action", "reasoning": proposal.get("reasoning")}

    # Guardrail 2: selector must be one we actually offered (no hallucinated targets)
    match = next((c for c in candidates if c["selector"] == selector), None)
    if match is None:
        return {"status": "blocked", "reason": "selector not in candidate set (rejected as hallucinated)",
                "raw": proposal}

    # Guardrail 3: action/tag compatibility
    if match["tag"] not in ACTION_TAG_COMPAT.get(action, set()):
        return {"status": "blocked",
                "reason": f"action '{action}' not valid on tag <{match['tag']}>",
                "raw": proposal}

    # Guardrail 4: fill payload sanity
    if action == "fill":
        if not value or not str(value).strip():
            return {"status": "blocked", "reason": "fill action requires a non-empty value"}
        if len(str(value)) > MAX_FILL_VALUE_LEN:
            return {"status": "blocked", "reason": "fill value exceeds max length guardrail"}

    plan = {
        "status": "ready_for_execution",
        "action": action,
        "selector": selector,
        "value": value,
        "target_url": match["url"],
        "reasoning": proposal.get("reasoning"),
    }
    logger.info(f"action plan resolved: {plan}")
    return plan
