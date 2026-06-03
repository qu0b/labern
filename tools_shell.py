"""A strictly-allowlisted shell tool for the dictation agent (stdlib only).

Same TOOLS contract as voice_input_tools.py / tools_files.py: one entry with an
Anthropic `schema` (name/description/input_schema) plus a `run(args, vi)` callable
returning a string the model reads back as a tool_result. `vi` is the VoiceInput
instance — the working dir comes off `vi.tools_config` ([tools].root, else cwd).

SAFETY IS THE POINT. The command text ultimately originates from speech, so this
tool refuses anything not explicitly allowed:
  * The command is parsed with shlex.split and run with shell=False (no shell is
    ever involved), so no quoting, globbing, or substitution is interpreted.
  * The program (argv[0]) must appear in an allowlist; everything else is rejected.
  * Shell metacharacters in the raw text (chaining/redirection/substitution) are
    rejected up front so the model learns to send one simple command.
The allowlist is read-only / inspection only by design; it deliberately omits
anything that mutates the filesystem, escalates privileges, or hits the network.
"""

import os
import shlex
import subprocess

# Read-only / inspection-only programs. Nothing here mutates the filesystem,
# escalates privileges, spawns a shell, or reaches the network. Anything NOT in
# this set is rejected, so destructive tools (rm, mv, dd, sudo, curl, bash, ...)
# are blocked by virtue of simply not being listed.
_DEFAULT_ALLOW = frozenset({
    "git", "ls", "cat", "head", "tail", "wc", "rg", "grep", "egrep", "fgrep",
    "find", "tree", "pwd", "env", "printenv", "ps", "df", "du", "stat", "file",
    "which", "type", "date", "echo", "uname", "hostname", "sed", "awk", "sort",
    "uniq", "cut", "jq", "colgrep", "dig", "host", "ip", "uptime", "whoami", "id",
})

# Raw-text characters that imply chaining / redirection / substitution. Even
# though shell=False would never interpret them, we refuse so the model doesn't
# bother trying to compose pipelines.
_SHELL_METACHARS = set(";|&><`$(){}\n")

# Output budget — the model re-reads tool output every turn, so keep it compact.
_MAX_BYTES = 8 * 1024
_TIMEOUT = 30


def _root(vi):
    """Working directory: [tools].root from config, falling back to cwd."""
    return (getattr(vi, "tools_config", {}) or {}).get("root") or os.getcwd()


def _allowlist(vi):
    """Effective allowlist: a non-empty [tools].shell_allow override, else default."""
    override = (getattr(vi, "tools_config", {}) or {}).get("shell_allow")
    if isinstance(override, (list, tuple)) and override:
        return frozenset(str(p) for p in override)
    return _DEFAULT_ALLOW


def _cap(text):
    """Trim text to the byte budget, appending a truncation note if needed."""
    if text is None:
        return ""
    data = text.encode("utf-8", errors="replace")
    if len(data) <= _MAX_BYTES:
        return text
    clipped = data[:_MAX_BYTES].decode("utf-8", errors="replace")
    return clipped + f"\n... (truncated at {_MAX_BYTES} bytes)"


def _shell(args, vi=None):
    """Run ONE allowlisted command without a shell and return a compact result."""
    command = args.get("command")
    if not command or not str(command).strip():
        return "error: 'command' is required"
    command = str(command)

    # Refuse chaining / redirection / substitution before we even parse.
    if any(c in command for c in _SHELL_METACHARS) or "&&" in command or "||" in command:
        return "error: shell operators are not allowed (run one simple command)"

    try:
        argv = shlex.split(command)
    except ValueError as e:
        return f"error: could not parse command: {e}"
    if not argv:
        return "error: 'command' is required"

    prog = argv[0]
    allow = _allowlist(vi)
    if prog not in allow:
        return (f"error: command not allowed: {prog} "
                f"(allowed: {', '.join(sorted(allow))})")

    cwd = _root(vi)
    try:
        proc = subprocess.run(
            argv, shell=False, cwd=cwd,
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
    except FileNotFoundError:
        return f"error: program not found: {prog}"
    except subprocess.TimeoutExpired:
        return "error: command timed out"
    except OSError as e:
        return f"error: failed to run command: {e}"

    parts = [f"exit code: {proc.returncode}"]
    out = (proc.stdout or "").rstrip("\n")
    if out:
        parts.append("stdout:\n" + out)
    err = (proc.stderr or "").rstrip("\n")
    if err:
        parts.append("stderr:\n" + err)
    if not out and not err:
        parts.append("(no output)")
    return _cap("\n".join(parts))


TOOLS = {
    "shell": {
        "schema": {
            "name": "shell",
            "description": (
                "Run ONE simple, read-only shell command and return its output "
                "(exit code, stdout, stderr). Runs without a shell, so no pipes, "
                "redirection, chaining (; | && >), or substitution — send a single "
                "command. Only inspection programs are allowed (git, ls, cat, grep, "
                "find, wc, jq, ...); anything that writes, deletes, escalates, or "
                "hits the network is refused."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "a single command line, e.g. \"git status --short\" or "
                            "\"wc -l voice_input.py\" (no pipes/redirection/chaining)"
                        ),
                    },
                },
                "required": ["command"],
            },
        },
        "run": _shell,
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

    print("### ALLOWED COMMANDS (should run) ###\n")
    _show('shell: git status --short', _shell({"command": "git status --short"}, vi))
    _show('shell: ls', _shell({"command": "ls"}, vi))
    _show('shell: wc -l voice_input.py', _shell({"command": "wc -l voice_input.py"}, vi))

    print("### DISALLOWED COMMANDS (should be refused) ###\n")
    _show('shell: rm -rf /tmp/x', _shell({"command": "rm -rf /tmp/x"}, vi))
    _show('shell: echo hi > /tmp/x', _shell({"command": "echo hi > /tmp/x"}, vi))
    _show('shell: curl http://x', _shell({"command": "curl http://x"}, vi))
