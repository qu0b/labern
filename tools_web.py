"""Web-search tools for the labern dictation agent, backed by the Exa API.

Two tools the model can call during the Anthropic tool-use loop:
  exa_search   — semantic/keyword web search → numbered list of results.
  exa_contents — fetch the full(er) text of specific URLs.

Same contract as voice_input_tools.py: each TOOLS entry is an Anthropic `schema`
plus a `run(args, vi)` callable returning a compact string. Errors are returned
as "error: ..." strings, never raised. Output is capped so the model can re-read
results cheaply every turn.

The Exa API key is read from config ([tools].exa_api_key) or $EXA_API_KEY — it is
NEVER hardcoded here, because this repo is public. See https://docs.exa.ai.
"""

import os

import requests

_BASE = "https://api.exa.ai"
_TIMEOUT = 30
_MAX_OUT = 8000  # ~8KB cap on returned strings


def _api_key(vi):
    """Resolve the Exa key from labern config, then the environment."""
    cfg = getattr(vi, "tools_config", {}) or {}
    return cfg.get("exa_api_key") or os.environ.get("EXA_API_KEY")


def _post(path, key, payload):
    """POST JSON to the Exa API; return (data, None) or (None, err_string)."""
    try:
        resp = requests.post(
            f"{_BASE}{path}",
            headers={"x-api-key": key, "Content-Type": "application/json"},
            json=payload,
            timeout=_TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return None, f"error: Exa request to {path} timed out"
    except requests.exceptions.RequestException as e:
        return None, f"error: Exa request failed: {e}"
    if resp.status_code != 200:
        return None, f"error: Exa {path} returned {resp.status_code}: {resp.text[:300]}"
    try:
        return resp.json(), None
    except ValueError:
        return None, f"error: Exa {path} returned non-JSON: {resp.text[:300]}"


def _trim(text, n):
    """Collapse whitespace and clip to n chars (with an ellipsis if clipped)."""
    text = " ".join((text or "").split())
    return text[: n - 1] + "…" if len(text) > n else text


def _exa_search(args, vi=None):
    """Exa /search → compact numbered list of title / url / snippet."""
    key = _api_key(vi)
    if not key:
        return "error: no Exa API key (set [tools].exa_api_key or $EXA_API_KEY)"
    query = (args.get("query") or "").strip()
    if not query:
        return "error: 'query' is required"
    try:
        num = max(1, min(int(args.get("num_results") or 5), 10))
    except (TypeError, ValueError):
        num = 5
    payload = {
        "query": query,
        "numResults": num,
        "type": "auto",
        "contents": {"text": {"maxCharacters": 800}},
    }
    category = (args.get("category") or "").strip()
    if category:
        payload["category"] = category

    data, err = _post("/search", key, payload)
    if err:
        return err
    results = data.get("results") or []
    if not results:
        return "(no results)"

    lines = []
    for i, r in enumerate(results[:num], 1):
        title = _trim(r.get("title") or "(untitled)", 150)
        url = r.get("url") or ""
        snippet = _trim(r.get("text") or "", 200)
        block = f"{i}. {title}\n   {url}"
        if snippet:
            block += f"\n   {snippet}"
        lines.append(block)
    return "\n".join(lines)[:_MAX_OUT]


def _exa_contents(args, vi=None):
    """Exa /contents → title / url / text for each requested URL."""
    key = _api_key(vi)
    if not key:
        return "error: no Exa API key (set [tools].exa_api_key or $EXA_API_KEY)"
    urls = args.get("urls")
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list) or not urls:
        return "error: 'urls' is required (a list of URL strings)"
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]
    if not urls:
        return "error: 'urls' is required (a list of URL strings)"

    data, err = _post(
        "/contents", key, {"urls": urls, "text": {"maxCharacters": 1500}}
    )
    if err:
        return err
    results = data.get("results") or []
    if not results:
        return "(no contents)"

    blocks = []
    for r in results:
        title = _trim(r.get("title") or "(untitled)", 200)
        url = r.get("url") or ""
        text = _trim(r.get("text") or "", 1500)
        block = f"{title}\n{url}"
        if text:
            block += f"\n{text}"
        blocks.append(block)
    return "\n\n".join(blocks)[:_MAX_OUT]


TOOLS = {
    "exa_search": {
        "schema": {
            "name": "exa_search",
            "description": (
                "Search the web with Exa and return a numbered list of results "
                "(title, url, and a short text snippet). Use this to find current "
                "information, pages, papers, or companies before answering."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "what to search the web for"},
                    "num_results": {"type": "integer",
                                    "description": "max results, 1-10 (default 5)"},
                    "category": {"type": "string",
                                 "description": ("optional focus: e.g. 'company', "
                                                 "'research paper', 'news', "
                                                 "'github', 'pdf'")},
                },
                "required": ["query"],
            },
        },
        "run": _exa_search,
    },
    "exa_contents": {
        "schema": {
            "name": "exa_contents",
            "description": (
                "Fetch the full(er) page text for specific URLs via Exa. Use this "
                "after exa_search to read a result in more detail before answering."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "urls": {"type": "array",
                             "items": {"type": "string"},
                             "description": "list of URLs to fetch text for"},
                },
                "required": ["urls"],
            },
        },
        "run": _exa_contents,
    },
}


if __name__ == "__main__":
    # Self-test: key comes from $EXA_API_KEY (never written into this file).
    print("=== exa_search ===")
    out = _exa_search({"query": "what is the Exa API", "num_results": 3})
    print(out)

    # Pull a URL out of the search result and read it back via exa_contents.
    first_url = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("http"):
            first_url = line
            break

    print("\n=== exa_contents ===")
    if first_url:
        print(_exa_contents({"urls": [first_url]}))
    else:
        print("(no URL found in search output to fetch)")
