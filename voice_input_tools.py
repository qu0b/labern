"""Local tools the dictation agent can call (Hermes-style: tools are plain
functions executed by the host, the model only emits intentions).

A tool is an entry in TOOLS: an Anthropic tool `schema` (name/description/
input_schema) plus a `run(args, vi)` callable returning a string the model reads
back as a tool_result. `vi` is the VoiceInput instance — tools read config off it
(e.g. the search root). Keep results small; the model re-reads them every turn.

First tool: semantic_search → colgrep (semantic code search, ColBERT-backed).
"""

import json
import os
import subprocess


def _semantic_search(args, vi=None):
    """colgrep semantic search → trimmed JSON of matching code units."""
    query = (args.get("query") or "").strip()
    if not query:
        return "error: 'query' is required"
    root = (args.get("path")
            or (getattr(vi, "tools_config", {}) or {}).get("root")
            or os.getcwd())
    try:
        k = max(1, min(int(args.get("k") or 8), 20))
    except (TypeError, ValueError):
        k = 8
    cmd = ["colgrep", "--json", "-k", str(k), query, root]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return "error: colgrep is not installed"
    except subprocess.TimeoutExpired:
        return "error: colgrep timed out"
    if proc.returncode != 0:
        return f"error: colgrep exit {proc.returncode}: {proc.stderr.strip()[:300]}"
    try:
        results = json.loads(proc.stdout or "[]")
    except ValueError:
        return (proc.stdout or "(no output)")[:1500]
    # Trim to what the model needs to reason + cite; drop the call-graph noise.
    hits = []
    for r in results[:k]:
        u = r.get("unit", {}) if isinstance(r, dict) else {}
        snippet = (u.get("code") or u.get("signature") or "")
        hits.append({
            "file": u.get("file"),
            "line": u.get("line"),
            "name": u.get("qualified_name") or u.get("name"),
            "signature": u.get("signature"),
            "snippet": snippet[:600],
        })
    return json.dumps(hits or [{"note": "no matches"}])


TOOLS = {
    "semantic_search": {
        "schema": {
            "name": "semantic_search",
            "description": (
                "Search a codebase by meaning (not just keywords) using colgrep, "
                "and return matching code units with their file, line, signature, "
                "and a code snippet. Use this to locate where something is "
                "implemented before answering."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string",
                              "description": "natural-language description of what to find"},
                    "path": {"type": "string",
                             "description": "directory or file to search (default: configured project root)"},
                    "k": {"type": "integer",
                          "description": "max results, 1-20 (default 8)"},
                },
                "required": ["query"],
            },
        },
        "run": _semantic_search,
    },
}
