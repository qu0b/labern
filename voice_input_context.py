#!/usr/bin/env python3
"""AT-SPI focused-widget snapshot for labern (pull-based).

Run once, prints a JSON snapshot of the currently-focused widget to stdout, and
exits:

    /usr/bin/python3 voice_input_context.py --once
    {"app": "Code", "role": "text", "name": "...", "window": "labern - Code"}

labern spawns this at dictation time (when a transform pipeline runs) to pick a
context-appropriate pipeline and project root. It is invoked on demand rather
than run as a standing listener, so the desktop pays no continuous accessibility
IPC cost — only one quick scan per dictation.

Runs under the system /usr/bin/python3 so it can import gi.repository.Atspi (the
labern uv venv intentionally has no gi). Fail-open: any error → prints `{}` and
labern falls back to its pipeline defaults.

Needs `gsettings set org.gnome.desktop.interface toolkit-accessibility true` so
the a11y bus exposes the focused widget.
"""

import json
import sys

import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi  # noqa: E402

# Roles that represent a toplevel container we'd call "the window".
_WINDOW_ROLES = (Atspi.Role.FRAME, Atspi.Role.WINDOW, Atspi.Role.DIALOG)


def _find_window(obj):
    """Walk up to the nearest frame/window ancestor; obj itself on failure."""
    cur = obj
    for _ in range(20):  # depth cap; AT-SPI trees can have cycles on bad apps
        if cur is None:
            return obj
        try:
            if cur.get_role() in _WINDOW_ROLES:
                return cur
            cur = cur.get_parent()
        except Exception:
            return obj
    return obj


def _snapshot(obj):
    """Best-effort dict of {app, role, name, window} for one accessible."""
    try:
        app = obj.get_application()
        win = _find_window(obj)
        return {
            "app":    (app.get_name() if app else "") or "",
            "role":   obj.get_role_name() or "",
            "name":   obj.get_name() or "",
            "window": (win.get_name() if win else "") or "",
        }
    except Exception:
        return {}


def snapshot_focused():
    """Scan the active window for the focused widget and return its snapshot dict
    ({} if none found). One pass over the active app's window subtree — the
    pull-based replacement for subscribing to every global focus change."""
    try:
        desktop = Atspi.get_desktop(0)
        for i in range(desktop.get_child_count()):
            app = desktop.get_child_at_index(i)
            if app is None:
                continue
            for j in range(app.get_child_count()):
                win = app.get_child_at_index(j)
                if win is None:
                    continue
                try:
                    if not win.get_state_set().contains(Atspi.StateType.ACTIVE):
                        continue
                except Exception:
                    continue
                # DFS for a focused descendant; fall back to the window itself.
                stack = [win]
                while stack:
                    node = stack.pop()
                    try:
                        if node.get_state_set().contains(Atspi.StateType.FOCUSED):
                            return _snapshot(node)
                        for k in range(node.get_child_count()):
                            child = node.get_child_at_index(k)
                            if child is not None:
                                stack.append(child)
                    except Exception:
                        pass
                return _snapshot(win)
    except Exception as e:
        print(f"labern-context: scan failed: {e}", file=sys.stderr)
    return {}


def main():
    # Single-shot: print one snapshot as JSON and exit. The `--once` flag is
    # accepted for explicitness but this is the only mode.
    if Atspi.init() != 0:
        print("{}")  # a11y bus unavailable → fail-open, labern uses defaults
        return
    print(json.dumps(snapshot_focused()))


if __name__ == "__main__":
    main()
