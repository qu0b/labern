#!/usr/bin/env python3
"""AT-SPI focus-change listener for labern.

Subscribes once to ``object:state-changed:focused`` events on the session
accessibility bus. On each event, atomically writes a JSON snapshot of the
focused widget to ~/.cache/labern/context.json. The main labern process reads
that file at dictation time to pick a context-appropriate transform pipeline.

Runs as a daemon spawned by labern with the system /usr/bin/python3 so it can
import gi.repository.Atspi (the labern uv venv intentionally has no gi).
Fail-open: any error → cache stays as-is; labern uses pipeline defaults.

Needs `gsettings set org.gnome.desktop.interface toolkit-accessibility true`
or focus events will not be emitted on GNOME.
"""

import json
import os
import sys
import tempfile

import gi
gi.require_version("Atspi", "2.0")
from gi.repository import Atspi  # noqa: E402

CACHE = os.path.expanduser("~/.cache/labern/context.json")

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


def _write(d):
    """Atomic write: tmp file in the same dir, then rename. Silent on failure."""
    try:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CACHE), prefix=".ctx.")
        try:
            os.write(fd, json.dumps(d).encode())
        finally:
            os.close(fd)
        os.rename(tmp, CACHE)
    except OSError as e:
        print(f"labern-context: write failed: {e}", file=sys.stderr)


def _seed_initial():
    """Find the currently-focused widget and seed the cache once at startup."""
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
                            _write(_snapshot(node))
                            return
                        for k in range(node.get_child_count()):
                            child = node.get_child_at_index(k)
                            if child is not None:
                                stack.append(child)
                    except Exception:
                        pass
                _write(_snapshot(win))
                return
    except Exception as e:
        print(f"labern-context: initial scan failed: {e}", file=sys.stderr)


def _on_focus(event):
    # detail1 == 1 → focus gained; 0 → lost. Only act on gain.
    if event.detail1:
        _write(_snapshot(event.source))


def main():
    if Atspi.init() != 0:
        print("labern-context: Atspi.init() failed; is the a11y bus running?",
              file=sys.stderr)
        return
    _seed_initial()
    listener = Atspi.EventListener.new(_on_focus)
    listener.register("object:state-changed:focused")
    Atspi.event_main()  # blocks; SIGTERM unblocks cleanly


if __name__ == "__main__":
    main()
