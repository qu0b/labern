"""Filesystem tools the dictation agent can call (stdlib only).

Same TOOLS contract as voice_input_tools.py: each entry has an Anthropic `schema`
(name/description/input_schema) plus a `run(args, vi)` callable returning a string
the model reads back as a tool_result. `vi` is the VoiceInput instance — tools read
the search root off `vi.tools_config`. Keep results small: the model re-reads them
every turn, so large output is capped with a truncation note.

Tools: file_list (enumerate files), tree (directory tree), read_file (cat -n).
"""

import os
import subprocess

# Directories we never want to descend into when walking a tree by hand.
_NOISE_DIRS = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}

# Compaction limits (the model re-reads tool output every turn).
_FILE_LIST_MAX = 200
_TREE_MAX_LINES = 300
_READ_MAX_LINES = 400
_READ_MAX_BYTES = 20 * 1024


def _root(vi):
    """Default project root from [tools] config, falling back to cwd."""
    return (getattr(vi, "tools_config", {}) or {}).get("root") or os.getcwd()


def _resolve(path, vi):
    """Resolve a possibly-relative path against the configured root."""
    if not path:
        return _root(vi)
    if os.path.isabs(path):
        return path
    return os.path.join(_root(vi), path)


def _git_ls_files(root):
    """Tracked + untracked-but-not-ignored files relative to root, or None.

    None means root is not inside a git work tree (caller falls back to os.walk).
    """
    try:
        inside = subprocess.run(
            ["git", "-C", root, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", root, "ls-files",
             "--cached", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return [ln for ln in proc.stdout.splitlines() if ln]


def _walk_files(root):
    """Relative file paths under root via os.walk, skipping noise dirs."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _NOISE_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            out.append(os.path.relpath(full, root))
    return out


def _file_list(args, vi=None):
    """List files under a path, preferring git (respects .gitignore)."""
    root = _resolve(args.get("path"), vi)
    if not os.path.exists(root):
        return f"error: path not found: {root}"
    if os.path.isfile(root):
        return f"error: path is a file, not a directory: {root}"
    pattern = (args.get("pattern") or "").strip()
    try:
        cap = max(1, int(args.get("max") or _FILE_LIST_MAX))
    except (TypeError, ValueError):
        cap = _FILE_LIST_MAX

    files = _git_ls_files(root)
    if files is None:
        try:
            files = _walk_files(root)
        except OSError as e:
            return f"error: walk failed: {e}"
    else:
        files = sorted(files)

    if pattern:
        import fnmatch
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), pattern)]

    if not files:
        return "(no matching files)"

    total = len(files)
    shown = files[:cap]
    text = "\n".join(shown)
    if total > cap:
        text += f"\n... ({total - cap} more)"
    return text


def _tree(args, vi=None):
    """Indented directory tree, depth-limited, skipping noise dirs."""
    root = _resolve(args.get("path"), vi)
    if not os.path.exists(root):
        return f"error: path not found: {root}"
    if not os.path.isdir(root):
        return f"error: path is not a directory: {root}"
    try:
        depth = max(0, int(args.get("depth") if args.get("depth") is not None else 2))
    except (TypeError, ValueError):
        depth = 2

    lines = [os.path.basename(os.path.normpath(root)) or root]
    truncated = False

    def walk(d, prefix, level):
        nonlocal truncated
        if truncated or level > depth:
            return
        try:
            entries = sorted(os.listdir(d))
        except OSError as e:
            lines.append(prefix + f"[error: {e}]")
            return
        entries = [e for e in entries if e not in _NOISE_DIRS]
        dirs = [e for e in entries if os.path.isdir(os.path.join(d, e))]
        rest = [e for e in entries if not os.path.isdir(os.path.join(d, e))]
        ordered = dirs + rest
        for i, name in enumerate(ordered):
            if len(lines) >= _TREE_MAX_LINES:
                truncated = True
                return
            last = i == len(ordered) - 1
            connector = "└── " if last else "├── "
            full = os.path.join(d, name)
            is_dir = os.path.isdir(full)
            lines.append(prefix + connector + name + ("/" if is_dir else ""))
            if is_dir:
                extension = "    " if last else "│   "
                walk(full, prefix + extension, level + 1)

    walk(root, "", 1)
    if truncated:
        lines.append(f"... (truncated at {_TREE_MAX_LINES} lines)")
    return "\n".join(lines)


def _read_file(args, vi=None):
    """Read a file with cat -n style line-number prefixes, honoring a range."""
    raw = args.get("path")
    if not raw or not str(raw).strip():
        return "error: 'path' is required"
    path = _resolve(str(raw).strip(), vi)
    if not os.path.exists(path):
        return f"error: file not found: {path}"
    if os.path.isdir(path):
        return f"error: path is a directory, not a file: {path}"

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    start = _int(args.get("start_line"))
    end = _int(args.get("end_line"))

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError as e:
        return f"error: read failed: {e}"

    n = len(all_lines)
    has_range = start is not None or end is not None
    if has_range:
        lo = max(1, start if start is not None else 1)
        hi = min(n, end if end is not None else n)
        if lo > n:
            return f"error: start_line {lo} is past end of file ({n} lines)"
        if hi < lo:
            return f"error: end_line {hi} is before start_line {lo}"
        sel = list(range(lo, hi + 1))
        note = None
    else:
        # No range: cap at line and byte budget, whichever comes first.
        sel = []
        size = 0
        for idx in range(n):
            if len(sel) >= _READ_MAX_LINES:
                break
            size += len(all_lines[idx].encode("utf-8", errors="replace"))
            if size > _READ_MAX_BYTES and sel:
                break
            sel.append(idx + 1)
        capped = len(sel) < n
        note = (f"... (truncated: showing {len(sel)} of {n} lines; "
                f"pass start_line/end_line for more)") if capped else None

    width = len(str(sel[-1])) if sel else 1
    body = "".join(
        f"{ln:>{width}}\t{all_lines[ln - 1]}" for ln in sel
    )
    if body and not body.endswith("\n"):
        body += "\n"
    if note:
        body += note + "\n"
    return body.rstrip("\n") or "(empty file)"


TOOLS = {
    "file_list": {
        "schema": {
            "name": "file_list",
            "description": (
                "List files under a directory, one path per line (relative to the "
                "search root). Respects .gitignore when inside a git repo. Use this "
                "to discover what files exist before reading them."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "directory to list (default: configured project root)"},
                    "pattern": {"type": "string",
                                "description": "glob matched on the file basename, e.g. '*.py' (optional)"},
                    "max": {"type": "integer",
                            "description": "max paths to return (default 200)"},
                },
                "required": [],
            },
        },
        "run": _file_list,
    },
    "tree": {
        "schema": {
            "name": "tree",
            "description": (
                "Render a directory as an indented tree (like the `tree` command), "
                "skipping noise dirs (.git, node_modules, .venv, etc). Use this to "
                "understand a project's layout at a glance."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "directory to render (default: configured project root)"},
                    "depth": {"type": "integer",
                              "description": "how many levels deep to descend (default 2)"},
                },
                "required": [],
            },
        },
        "run": _tree,
    },
    "read_file": {
        "schema": {
            "name": "read_file",
            "description": (
                "Read a text file and return its contents with line-number prefixes "
                "(like `cat -n`). Pass start_line/end_line to read a specific range; "
                "otherwise output is capped and truncation is noted."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string",
                             "description": "file to read (relative paths resolve against the project root)"},
                    "start_line": {"type": "integer",
                                   "description": "1-based first line to include (optional)"},
                    "end_line": {"type": "integer",
                                 "description": "last line to include, inclusive (optional)"},
                },
                "required": ["path"],
            },
        },
        "run": _read_file,
    },
}


if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))

    class _VI:
        tools_config = {"root": _here}

    vi = _VI()

    def _show(title, out):
        print("=" * 70)
        print(title)
        print("-" * 70)
        print(out)
        print()

    _show("file_list (all, default root)",
          _file_list({}, vi))
    _show("file_list (pattern '*.py')",
          _file_list({"pattern": "*.py"}, vi))
    _show("file_list (max 3)",
          _file_list({"max": 3}, vi))
    _show("file_list (error: missing path)",
          _file_list({"path": "does-not-exist"}, vi))

    _show("tree (depth 2, default root)",
          _tree({}, vi))
    _show("tree (depth 1)",
          _tree({"depth": 1}, vi))

    _show("read_file (full, capped)",
          _read_file({"path": "voice_input_tools.py"}, vi))
    _show("read_file (range 1-10)",
          _read_file({"path": "tools_files.py", "start_line": 1, "end_line": 10}, vi))
    _show("read_file (error: missing path)",
          _read_file({}, vi))
    _show("read_file (error: not found)",
          _read_file({"path": "nope.txt"}, vi))
