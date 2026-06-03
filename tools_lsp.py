"""Code-intelligence tool for the dictation agent, backed by the Language Server
Protocol. A minimal one-shot LSP stdio client (no client library): for each call
we spawn a language server, drive the handshake + a single query over JSON-RPC
with `Content-Length` framing, then shut it down.

Exposes the standard TOOLS dict: an Anthropic `schema` plus a `run(args, vi)`
callable that returns a compact string the model reads back. Errors never raise —
they come back as "error: ...".

Languages are registered in SERVERS by file extension; add an entry to support
more. python-lsp-server (pylsp) ships with labern, so .py works out of the box.
"""

import json
import os
import subprocess
import threading
import time

# Where to find the bundled python-lsp-server. Prefer the venv binary next to this
# file; fall back to whatever `pylsp` is on PATH.
_VENV_PYLSP = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "pylsp")
_PYLSP = _VENV_PYLSP if os.path.exists(_VENV_PYLSP) else "pylsp"

# extension -> server launch command. One per language; extend to add more.
SERVERS = {
    ".py": [_PYLSP],
}

_TIMEOUT = 15.0  # seconds to wait for any single server response

# LSP DiagnosticSeverity (1-based) -> short label.
_SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def _path_to_uri(path):
    return "file://" + os.path.abspath(path)


def _uri_to_path(uri):
    return uri[len("file://"):] if uri.startswith("file://") else uri


class _LspClient:
    """One short-lived LSP conversation over a subprocess's stdin/stdout.

    A background reader thread parses framed messages and routes them: responses
    (have an `id`) land in a dict keyed by id; notifications (have a `method`)
    are appended to a list. Callers block on those with a timeout so a wedged
    server can never hang the agent."""

    def __init__(self, cmd, root):
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._root = root
        self._next_id = 1
        self._lock = threading.Lock()
        self._responses = {}             # id -> result-or-error message
        self._notifications = []         # list of notification messages
        self._cond = threading.Condition(self._lock)
        self._alive = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # ---- framed I/O ---------------------------------------------------------

    def _write(self, msg):
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + body)
        self._proc.stdin.flush()

    def _read_message(self):
        """Read one Content-Length-framed JSON message. None on EOF."""
        out = self._proc.stdout
        headers = {}
        while True:
            line = out.readline()
            if not line:
                return None                      # EOF
            line = line.rstrip(b"\r\n")
            if line == b"":
                break                            # blank line ends the header block
            if b":" in line:
                key, _, val = line.partition(b":")
                headers[key.strip().lower()] = val.strip()
        length = int(headers.get(b"content-length", b"0"))
        body = b""
        while len(body) < length:
            chunk = out.read(length - len(body))
            if not chunk:
                return None
            body += chunk
        try:
            return json.loads(body.decode("utf-8"))
        except ValueError:
            return {}

    def _read_loop(self):
        while True:
            msg = self._read_message()
            if msg is None:
                with self._cond:
                    self._alive = False
                    self._cond.notify_all()
                return
            with self._cond:
                if "id" in msg and ("result" in msg or "error" in msg):
                    self._responses[msg["id"]] = msg
                elif msg.get("method"):
                    self._notifications.append(msg)
                self._cond.notify_all()

    # ---- JSON-RPC primitives ------------------------------------------------

    def request(self, method, params):
        """Send a request and block for its response. Returns the result, or
        raises TimeoutError / RuntimeError so callers can map to 'error: ...'."""
        with self._lock:
            rid = self._next_id
            self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        deadline = time.monotonic() + _TIMEOUT
        with self._cond:
            while rid not in self._responses:
                if not self._alive:
                    raise RuntimeError("lsp server exited")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(method)
                self._cond.wait(remaining)
            msg = self._responses.pop(rid)
        if "error" in msg:
            raise RuntimeError(msg["error"].get("message", "lsp error"))
        return msg.get("result")

    def notify(self, method, params):
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def wait_notification(self, method, predicate=None):
        """Block until a notification with `method` (and matching `predicate`)
        arrives, or raise TimeoutError. Drains already-buffered ones first."""
        deadline = time.monotonic() + _TIMEOUT
        seen = 0
        with self._cond:
            while True:
                while seen < len(self._notifications):
                    msg = self._notifications[seen]
                    seen += 1
                    if msg.get("method") == method and (predicate is None or predicate(msg)):
                        return msg
                if not self._alive:
                    raise RuntimeError("lsp server exited")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(method)
                self._cond.wait(remaining)

    # ---- lifecycle ----------------------------------------------------------

    def initialize(self):
        self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": _path_to_uri(self._root),
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {},
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {},
                    "references": {},
                },
            },
            "clientInfo": {"name": "labern-lsp", "version": "1"},
        })
        self.notify("initialized", {})

    def did_open(self, path, text, language_id):
        self.notify("textDocument/didOpen", {
            "textDocument": {
                "uri": _path_to_uri(path),
                "languageId": language_id,
                "version": 1,
                "text": text,
            },
        })

    def shutdown(self):
        try:
            self.request("shutdown", None)
            self.notify("exit", {})
        except Exception:
            pass

    def close(self):
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


# ---- result formatting ------------------------------------------------------

def _hover_text(result):
    if not result:
        return "(no hover info)"
    contents = result.get("contents")
    if contents is None:
        return "(no hover info)"
    if isinstance(contents, str):
        return contents.strip() or "(no hover info)"
    if isinstance(contents, dict):                  # MarkupContent or {language,value}
        return (contents.get("value") or "").strip() or "(no hover info)"
    if isinstance(contents, list):                  # list of strings/MarkedString
        parts = []
        for c in contents:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                parts.append(c.get("value", ""))
        return "\n".join(p for p in parts if p).strip() or "(no hover info)"
    return str(contents)


def _locations(result, root):
    """Normalize a Location | Location[] | LocationLink[] into relpath:line:col."""
    if not result:
        return []
    items = result if isinstance(result, list) else [result]
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        uri = it.get("uri") or it.get("targetUri")
        rng = it.get("range") or it.get("targetSelectionRange") or it.get("targetRange")
        if not uri or not rng:
            continue
        path = _uri_to_path(uri)
        try:
            rel = os.path.relpath(path, root)
        except ValueError:
            rel = path
        start = rng.get("start", {})
        line = start.get("line", 0) + 1            # back to 1-based for humans
        col = start.get("character", 0) + 1
        out.append(f"{rel}:{line}:{col}")
    return out


# ---- the tool itself --------------------------------------------------------

def _run(args, vi=None):
    action = (args.get("action") or "").strip()
    if action not in ("hover", "definition", "references", "diagnostics"):
        return ("error: action must be one of "
                "hover, definition, references, diagnostics")

    raw_path = (args.get("path") or "").strip()
    if not raw_path:
        return "error: 'path' is required"

    root = (getattr(vi, "tools_config", {}) or {}).get("root") or os.getcwd()
    path = raw_path if os.path.isabs(raw_path) else os.path.join(root, raw_path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return f"error: file not found: {path}"

    ext = os.path.splitext(path)[1].lower()
    cmd = SERVERS.get(ext)
    if not cmd:
        return f"error: no language server configured for {ext or '(no extension)'}"

    # Position (1-based from the model) -> LSP 0-based. Ignored for diagnostics.
    if action != "diagnostics":
        try:
            line = int(args.get("line"))
        except (TypeError, ValueError):
            return "error: 'line' (1-based integer) is required"
        try:
            character = int(args.get("character", 1))
        except (TypeError, ValueError):
            character = 1
        position = {"line": max(0, line - 1), "character": max(0, character - 1)}

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return f"error: cannot read {path}: {e}"

    language_id = {".py": "python"}.get(ext, "plaintext")

    client = None
    try:
        client = _LspClient(cmd, root)
        client.initialize()
        client.did_open(path, text, language_id)
        uri = _path_to_uri(path)
        td = {"textDocument": {"uri": uri}}

        if action == "hover":
            result = client.request("textDocument/hover", {**td, "position": position})
            return _hover_text(result)

        if action == "definition":
            result = client.request("textDocument/definition", {**td, "position": position})
            locs = _locations(result, root)
            return "\n".join(locs) if locs else "(no definition found)"

        if action == "references":
            result = client.request("textDocument/references", {
                **td, "position": position,
                "context": {"includeDeclaration": True},
            })
            locs = _locations(result, root)
            return "\n".join(locs) if locs else "(no references found)"

        # diagnostics: wait for the publishDiagnostics notification for this file.
        msg = client.wait_notification(
            "textDocument/publishDiagnostics",
            predicate=lambda m: (m.get("params") or {}).get("uri") == uri,
        )
        diags = (msg.get("params") or {}).get("diagnostics") or []
        if not diags:
            return "(no diagnostics)"
        lines = []
        for d in diags:
            sev = _SEVERITY.get(d.get("severity"), "info")
            ln = (d.get("range", {}).get("start", {}).get("line", 0)) + 1
            lines.append(f"{sev} L{ln}: {(d.get('message') or '').strip()}")
        return "\n".join(lines)
    except TimeoutError:
        return "error: lsp timeout"
    except FileNotFoundError:
        return f"error: language server not found: {cmd[0]}"
    except Exception as e:
        return f"error: {e}"
    finally:
        if client is not None:
            client.shutdown()
            client.close()


TOOLS = {
    "lsp": {
        "schema": {
            "name": "lsp",
            "description": (
                "Answer code-intelligence questions via the Language Server "
                "Protocol: hover (type/docstring at a position), definition (where "
                "a symbol is defined), references (all uses), or diagnostics (errors "
                "and warnings in a file). Positions are 1-based line and character."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["hover", "definition", "references", "diagnostics"],
                        "description": "which query to run",
                    },
                    "path": {
                        "type": "string",
                        "description": "file to query (absolute, or relative to the project root)",
                    },
                    "line": {
                        "type": "integer",
                        "description": "1-based line number (required except for diagnostics)",
                    },
                    "character": {
                        "type": "integer",
                        "description": "1-based column on that line (default 1; ignored for diagnostics)",
                    },
                },
                "required": ["action", "path"],
            },
        },
        "run": _run,
    },
}


if __name__ == "__main__":
    # Self-test: exercise all four actions against a real symbol in voice_input.py.
    # `self._transcribe` is the call site on line 246, col 38 (1-based); its
    # definition is `def _transcribe` on line 254.
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_input.py")

    class _FakeVI:
        tools_config = {"root": os.path.dirname(os.path.abspath(__file__))}

    vi = _FakeVI()
    LINE, CHAR = 246, 38  # at `_transcribe` in `self._transcribe`

    for action, kwargs in [
        ("hover", {"line": LINE, "character": CHAR}),
        ("definition", {"line": LINE, "character": CHAR}),
        ("references", {"line": LINE, "character": CHAR}),
        ("diagnostics", {}),
    ]:
        out = _run({"action": action, "path": target, **kwargs}, vi)
        print(f"===== {action} (line={kwargs.get('line')}, char={kwargs.get('character')}) =====")
        print(out)
        print()
