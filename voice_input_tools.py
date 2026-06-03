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
import tempfile


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


def _png_path(prefix):
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".png")
    os.close(fd)
    return path


def _browser_use(args, vi=None):
    """Ephemeral browsing via the browser-use CLI: navigate, screenshot, read text.
    Stateful agentic browser — best for one-off 'go here and grab this'. Returns
    {text, images}. Needs IN_DOCKER=true (no-sandbox) on this AppArmor host."""
    url = (args.get("url") or "").strip()
    if not url:
        return "error: 'url' is required"
    env = {**os.environ, "IN_DOCKER": "true"}
    try:
        nav = subprocess.run(["browser-use", "open", url],
                             capture_output=True, text=True, timeout=60, env=env)
    except FileNotFoundError:
        return "error: browser-use is not installed"
    except subprocess.TimeoutExpired:
        return "error: browser-use open timed out"
    if nav.returncode != 0:
        return f"error: browser-use open failed: {nav.stderr.strip()[:300]}"
    png = _png_path("labern_bu_")
    shot = subprocess.run(["browser-use", "screenshot"] + (["--full"] if args.get("full") else []) + [png],
                          capture_output=True, text=True, timeout=60, env=env)
    if shot.returncode != 0 or not os.path.exists(png):
        return f"error: screenshot failed: {shot.stderr.strip()[:300]}"
    text = ""
    state = subprocess.run(["browser-use", "--json", "state"],
                           capture_output=True, text=True, timeout=30, env=env)
    try:
        text = json.loads(state.stdout).get("data", {}).get("_raw_text", "")
    except ValueError:
        pass
    return {"text": f"browser-use screenshot of {url}\n{text[:1500]}", "images": [png]}


def _playwright(args, vi=None):
    """Deterministic browsing via Playwright (Microsoft): navigate at an exact
    viewport size, screenshot, read text. Best for repeatable captures and
    specific page sizes. Returns {text, images}."""
    url = (args.get("url") or "").strip()
    if not url:
        return "error: 'url' is required"
    try:
        width = max(320, min(int(args.get("width") or 1280), 3840))
        height = max(240, min(int(args.get("height") or 800), 2160))
    except (TypeError, ValueError):
        width, height = 1280, 800
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return "error: playwright not installed (uv pip install playwright)"
    png = _png_path("labern_pw_")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])  # AppArmor host
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(url, wait_until="load", timeout=30000)
            title, body = page.title(), ""
            try:
                body = page.inner_text("body")[:1500]
            except Exception:
                pass
            page.screenshot(path=png, full_page=bool(args.get("full")))
            browser.close()
    except Exception as e:
        return f"error: playwright failed: {e}"
    return {"text": f"playwright screenshot of {url} ({title}) at {width}x{height}\n{body}",
            "images": [png]}


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
    "browser_use": {
        "schema": {
            "name": "browser_use",
            "description": (
                "Open a URL in a real browser (browser-use) and capture a screenshot "
                "plus the visible page text. Best for one-off, ad-hoc browsing — "
                "'go to this page and grab what's there'."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to open"},
                    "full": {"type": "boolean",
                             "description": "capture the full scrollable page (default: viewport only)"},
                },
                "required": ["url"],
            },
        },
        "run": _browser_use,
    },
    "playwright": {
        "schema": {
            "name": "playwright",
            "description": (
                "Open a URL with Playwright at an EXACT viewport size and capture a "
                "screenshot plus visible text. Deterministic and repeatable — best "
                "for specific page sizes and reproducible captures."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to open"},
                    "width": {"type": "integer", "description": "viewport width px (default 1280)"},
                    "height": {"type": "integer", "description": "viewport height px (default 800)"},
                    "full": {"type": "boolean",
                             "description": "capture the full scrollable page (default: viewport only)"},
                },
                "required": ["url"],
            },
        },
        "run": _playwright,
    },
}

# Aggregate optional tool modules (each is a self-contained tools_*.py exposing
# its own TOOLS dict). A module whose backend/dep is missing fails to import and
# is simply skipped — it disables only its own tools, never the core registry.
import importlib  # noqa: E402

for _mod in ("tools_files", "tools_treesitter", "tools_lsp",
             "tools_web", "tools_shell", "tools_explore"):
    try:
        TOOLS.update(getattr(importlib.import_module(_mod), "TOOLS", {}))
    except Exception as _e:  # missing optional dep, import error, etc.
        import sys
        print(f"[tools: {_mod} unavailable — {_e}]", file=sys.stderr)
