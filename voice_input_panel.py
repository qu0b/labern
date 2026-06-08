#!/usr/bin/env python3
"""Interactive agent panel for labern (GTK 3, runs under the SYSTEM python).

labern spawns this when a tool-using ('agent') pipeline fires on a key. The main
process streams progress events to our stdin (one JSON object per line) and we
render them live: the spoken request, an activity log of each tool call, and the
agent's answer in an EDITABLE box. The user then Inserts (labern types it into the
app they dictated from), Copies, Refines (sends an instruction back for another
pass), or Cancels — we print the chosen action as one JSON line to stdout and the
main process carries it out.

Why GTK under the system python: running as a subprocess means a GUI crash can
never take down dictation, and labern launches us with /usr/bin/python3, which has
PyGObject/GTK — the same dependency surface as the AT-SPI context helper. (The uv
venv's bundled Tk is unusable here: its threaded Tcl notifier aborts Xlib under a
running mainloop.) Stdin is read event-driven via a GLib fd watch — no threads.

Protocol — main → us (stdin), one JSON per line:
  {"type":"request","text":"..."}              the spoken request (header)
  {"type":"status","text":"..."}               a status line in the log
  {"type":"tool","name":"...","input":{...}}   a tool call started
  {"type":"result","text":"..."}               agent finished; show editable answer
  {"type":"error","text":"..."}                something failed; show it
us → main (stdout), one JSON per line:
  {"action":"insert","text":"<edited answer>"} type it into the previous window
  {"action":"copy","text":"<edited answer>"}   put it on the clipboard
  {"action":"refine","text":"<instruction>"}   run another pass, then a new result
  {"action":"cancel"}                          discard
"""

import json
import os
import sys

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk, Pango  # noqa: E402

_CSS = b"""
window { background-color: #1e1f29; }
textview, textview text, entry { background-color: #15161e; color: #e6e6e6; }
entry { caret-color: #e6e6e6; }
label { color: #9aa0b4; }
.mono text { font-family: monospace; font-size: 9pt; }
"""


class Panel:
    def __init__(self, request=""):
        self.busy = True
        self.buf = ""
        self._closed = False

        self.win = Gtk.Window(title="labern · agent")
        self.win.set_default_size(580, 560)
        self.win.set_position(Gtk.WindowPosition.CENTER)
        self.win.set_keep_above(True)
        self.win.connect("destroy", self._on_destroy)

        try:
            prov = Gtk.CssProvider()
            prov.load_from_data(_CSS)
            Gtk.StyleContext.add_provider_for_screen(
                self.win.get_screen(), prov,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        except GLib.Error:
            pass  # styling is cosmetic; never fail over it

        self._build(request)
        self.win.show_all()

        fd = sys.stdin.fileno()
        try:
            os.set_blocking(fd, False)
        except (OSError, ValueError):
            pass
        GLib.io_add_watch(fd, GLib.IO_IN | GLib.IO_HUP, self._on_stdin)

    # ---- layout -------------------------------------------------------------

    def _build(self, request):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        for m in ("set_margin_top", "set_margin_bottom", "set_margin_start",
                  "set_margin_end"):
            getattr(box, m)(10)
        self.win.add(box)

        self.head = Gtk.Label(label=(request or "…"), xalign=0.0)
        self.head.set_line_wrap(True)
        self.head.set_selectable(True)
        attrs = Pango.AttrList()
        attrs.insert(Pango.attr_style_new(Pango.Style.ITALIC))
        self.head.set_attributes(attrs)
        box.pack_start(self.head, False, False, 0)

        box.pack_start(Gtk.Label(label="activity", xalign=0.0), False, False, 0)
        self.log = Gtk.TextView()
        self.log.set_editable(False)
        self.log.set_cursor_visible(False)
        self.log.set_wrap_mode(Gtk.WrapMode.WORD)
        self.log.get_style_context().add_class("mono")
        self.log_buf = self.log.get_buffer()
        sw1 = Gtk.ScrolledWindow()
        sw1.set_min_content_height(150)
        sw1.add(self.log)
        box.pack_start(sw1, False, False, 0)

        box.pack_start(Gtk.Label(label="result (editable)", xalign=0.0),
                       False, False, 0)
        self.out = Gtk.TextView()
        self.out.set_wrap_mode(Gtk.WrapMode.WORD)
        self.out_buf = self.out.get_buffer()
        sw2 = Gtk.ScrolledWindow()
        sw2.add(self.out)
        box.pack_start(sw2, True, True, 0)

        rf = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.refine_entry = Gtk.Entry()
        self.refine_entry.set_placeholder_text("refine: e.g. make it shorter — press Enter")
        self.refine_entry.connect("activate", lambda w: self._refine())
        rf.pack_start(self.refine_entry, True, True, 0)
        rb = Gtk.Button(label="Refine ↻")
        rb.connect("clicked", lambda w: self._refine())
        rf.pack_start(rb, False, False, 0)
        box.pack_start(rf, False, False, 0)

        br = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.insert_btn = Gtk.Button(label="Insert")
        self.insert_btn.connect("clicked", lambda w: self._insert())
        br.pack_start(self.insert_btn, False, False, 0)
        self.copy_btn = Gtk.Button(label="Copy")
        self.copy_btn.connect("clicked", lambda w: self._copy())
        br.pack_start(self.copy_btn, False, False, 0)
        cancel = Gtk.Button(label="Cancel")
        cancel.connect("clicked", lambda w: self._cancel())
        br.pack_end(cancel, False, False, 0)
        box.pack_start(br, False, False, 0)

        self.status = Gtk.Label(label="working…", xalign=0.0)
        box.pack_start(self.status, False, False, 0)
        self._set_busy(True)

    # ---- stdin -> UI (GLib fd watch, single-threaded) -----------------------

    def _on_stdin(self, fd, condition):
        if condition & GLib.IO_IN:
            try:
                chunk = os.read(fd, 65536)
            except (BlockingIOError, OSError):
                return True
            if not chunk:
                return False  # EOF: main closed the pipe; keep the window up
            self.buf += chunk.decode("utf-8", "replace")
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    self._handle(json.loads(line))
                except ValueError:
                    pass
            return True
        if condition & GLib.IO_HUP:
            return False
        return True

    def _handle(self, ev):
        t = ev.get("type")
        if t == "request":
            self.head.set_text(ev.get("text") or "…")
        elif t == "status":
            self._logline("· " + ev.get("text", ""))
            self.status.set_text(ev.get("text", ""))
        elif t == "tool":
            ip = ev.get("input") or {}
            arg = ", ".join(f"{k}={v!r}" for k, v in list(ip.items())[:3])
            self._logline(f"→ {ev.get('name')}({arg})")
        elif t == "result":
            self.out_buf.set_text(ev.get("text", ""))
            self._set_busy(False)
            self.status.set_text("ready — edit, then Insert / Copy / Refine")
            self.out.grab_focus()
        elif t == "error":
            self._logline("✗ " + ev.get("text", ""))
            self._set_busy(False)
            self.status.set_text("error — see activity")

    def _logline(self, s):
        self.log_buf.insert(self.log_buf.get_end_iter(), s + "\n")
        self.log.scroll_to_iter(self.log_buf.get_end_iter(), 0.0, False, 0, 0)

    def _set_busy(self, busy):
        self.busy = busy
        self.insert_btn.set_sensitive(not busy)
        self.copy_btn.set_sensitive(not busy)

    # ---- UI -> main ---------------------------------------------------------

    def _emit(self, obj):
        try:
            sys.stdout.write(json.dumps(obj) + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass

    def _result_text(self):
        s, e = self.out_buf.get_bounds()
        return self.out_buf.get_text(s, e, True).strip()

    def _insert(self):
        if self.busy:
            return
        self._closed = True
        self._emit({"action": "insert", "text": self._result_text()})
        self._quit()

    def _copy(self):
        if self.busy:
            return
        self._emit({"action": "copy", "text": self._result_text()})
        self.status.set_text("copied to clipboard")

    def _refine(self):
        instr = self.refine_entry.get_text().strip()
        if not instr or self.busy:
            return
        self.refine_entry.set_text("")
        self._logline(f"↻ refine: {instr}")
        self._set_busy(True)
        self.status.set_text("refining…")
        self._emit({"action": "refine", "text": instr})

    def _cancel(self):
        self._closed = True
        self._emit({"action": "cancel"})
        self._quit()

    def _on_destroy(self, *_):
        if not self._closed:
            self._emit({"action": "cancel"})
        Gtk.main_quit()

    def _quit(self):
        Gtk.main_quit()


def main():
    request = ""
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "--request":
        request = argv[1]
    Panel(request)
    Gtk.main()


if __name__ == "__main__":
    main()
