#!/usr/bin/env python3
"""
Push-to-talk voice input: hold a key, speak, release — text appears at the cursor.

Three keys (see BINDINGS below), all whisper-large-v3-turbo (multilingual, EN+DE):
    Right Ctrl  -> raw transcript, typed as-is
    Right Shift -> "clean" pass: an LLM fixes transcription slips (run-together
                   words, grammar, spelling) and normalizes technical terms from a
                   glossary, then types the result — no panel, no tools
    Right Alt   -> "agent" pass: tool-using agent carries out a spoken request

Transcription runs on a remote OpenAI-compatible endpoint when one is configured
(VOICE_INPUT_REMOTE_URL + an API key in ~/.config/voice-input/api_key or
$VOICE_INPUT_API_KEY); otherwise — or if the endpoint is unreachable — it uses a
local faster-whisper model. Works fully offline out of the box.

Setup:
    uv sync                           # build .venv from pyproject.toml
    sudo apt install xdotool          # X11 keystroke injection
    # Wayland: sudo apt install wtype

Usage:
    uv run voice_input.py --no-tray              # headless
    uv run voice_input.py --no-remote            # force the local model
    uv run voice_input.py --model small          # local fallback size
    uv run voice_input.py --language en          # force one language on every key

Hotkey notes (Linux/X11):
    Modifiers (ctrl_r, shift_r, alt_r) are ideal — single press, no Fn, emit no
    character, and don't auto-repeat while held. The laptop Fn key is invisible
    to software and can't be used. Avoid 'cmd'/Super (opens the GNOME overview).
    Edit BINDINGS to change keys/engines.
"""

import argparse
import faulthandler
import io
import json
import os
import signal
import platform
import queue
import re
import subprocess
import sys
import threading
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover — py3.9/3.10 fallback
    import tomli as tomllib

SAMPLE_RATE = 16000
SOUNDS = "/usr/share/sounds/freedesktop/stereo"
# Hard ceiling on a single push-to-talk recording. pynput drops modifier
# key-release events on X11 often enough that a stuck key is a real failure
# mode: without this, the mic stream stays open and `frames` grows unbounded.
# Well above any real dictation (longest observed ~53s) so it never clips speech.
MAX_RECORD_SECONDS = 120

# Per-key bindings: hold a key, transcribe, then run that key's pipeline (if any).
# language=None lets whisper auto-detect (EN+DE). All three keys share the remote
# endpoint + local fallback. ctrl_r types the raw transcript; shift_r runs the
# inline "refine" clean-up pass; alt_r runs the tool-using "agent" pass (panel).
# Parakeet (English-only, native punctuation) is retired from the default layout
# because the clean-up/agent passes must handle German too — re-add a binding with
# model="parakeet-tdt-0.6b-v2", language="en" if you want a fast English-only key.
BINDINGS = [
    {"key": "ctrl_r",  "model": "whisper-large-v3-turbo", "language": None, "label": "raw"},
    {"key": "shift_r", "model": "whisper-large-v3-turbo", "language": None, "label": "clean",
     "pipeline": "refine"},
    {"key": "alt_r",   "model": "whisper-large-v3-turbo", "language": None, "label": "agent",
     "pipeline": "agent"},
]

# The AT-SPI helper that snapshots the focused widget; run on demand per dictation.
CONTEXT_HELPER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voice_input_context.py")

# The interactive agent panel (Tkinter), spawned when a tool-using pipeline fires.
PANEL_HELPER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voice_input_panel.py")


def _make_icon(color):
    """Build a simple mic icon in the given color."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((24, 8, 40, 38), radius=8, fill=color)
    d.arc((18, 22, 46, 50), start=0, end=180, fill=color, width=4)
    d.line((32, 46, 32, 56), fill=color, width=4)
    d.line((24, 56, 40, 56), fill=color, width=4)
    return img


_SESSION = None


def _http():
    """Process-wide requests.Session: keep-alive reuses one TCP+TLS connection
    across STT and the up-to-max_steps agent round-trips instead of handshaking
    afresh every call. Lazily built so importing this module stays cheap."""
    global _SESSION
    if _SESSION is None:
        import requests
        _SESSION = requests.Session()
    return _SESSION


class VoiceInput:
    def __init__(self, bindings, model_size, device, use_tray, initial_prompt,
                 remote_url=None, api_key=None, local_beam_size=1,
                 local_cpu_threads=None,
                 agent=None, pipelines=None, context_rules=None, tools_config=None,
                 run_context_listener=True, glossary=None):
        self.use_tray = use_tray
        self.initial_prompt = initial_prompt
        # Technical-term dictionary for the LLM passes: substituted into any step
        # prompt containing the literal {glossary} token (see _render_prompt).
        self.glossary = glossary or ""
        self.recording = False
        self.frames = []
        self.stream = None
        self._watchdog = None  # auto-stop timer; guards a missed key-release
        self._lock = threading.Lock()
        self.tray = None
        self._icons = {}
        self._active = None  # binding currently recording

        # resolve each binding's key once; index by the resolved key object
        self.bindings = []
        self._keymap = {}
        for b in bindings:
            rb = dict(b, keyobj=self._resolve_key(b["key"]))
            self.bindings.append(rb)
            self._keymap[rb["keyobj"]] = rb

        # GPU-hosted endpoint preferred when a key is present; local is a lazy fallback.
        self.remote_url = remote_url
        self.api_key = api_key
        self.use_remote = bool(remote_url and api_key)
        self._model_size = model_size
        self._device = device
        self._compute = "int8" if device == "cpu" else "float16"
        self._beam_size = max(1, local_beam_size)  # greedy by default: ~3x faster on CPU
        # Cap CPU inference threads so a local-fallback transcription can't pin
        # every core and freeze the desktop. Default: leave half the machine free.
        self._cpu_threads = (local_cpu_threads if local_cpu_threads is not None
                             else max(1, (os.cpu_count() or 4) // 2))
        self.model = None

        # Transform-pipeline + context-routing state. All optional; absent config
        # → feature inert and behavior matches pre-pipeline labern exactly.
        agent = agent or {}
        self.agent_url = agent.get("url")
        self.agent_model = agent.get("model")
        self.agent_timeout = agent.get("timeout", 60)
        self.agent_max_tokens = agent.get("max_tokens", 2048)  # required by /v1/messages
        self.agent_max_steps = agent.get("max_steps", 6)       # tool-loop iteration cap
        # Reasoning depth for the endpoint ("thinking on + high"). Sent as the
        # OpenAI-style `reasoning_effort` field, which the LiteLLM proxy maps to
        # minimax's native reasoning. Empty/unset → no field sent (thinking off).
        self.reasoning_effort = (agent.get("reasoning_effort") or "").strip()
        self.vision = bool(agent.get("vision", False))         # send tool images to the model
        self.image_output = bool(agent.get("image_output", True))  # tool images → clipboard
        self.tools_config = dict(tools_config or {})
        self._base_tools_config = dict(self.tools_config)
        self.projects_dir = os.path.expanduser(
            self.tools_config.get("projects_dir") or "~/repos")
        self.pipelines = pipelines or {}
        self.context_rules = context_rules or []
        self._compiled_rules = self._compile_context_rules(self.context_rules)
        # Context is captured on demand at dictation time (see _context), not via a
        # standing AT-SPI listener — keeps the desktop free of continuous a11y IPC.
        self._consult_context = run_context_listener
        self._warn_pipeline_misconfig()

        if not self.use_remote:
            self._ensure_local_model()

        if self.initial_prompt:
            print(f"vocab bias: {self.initial_prompt.count(',') + 1} terms")
        if self.glossary:
            print(f"glossary: {self.glossary.count(chr(10)) + 1} terms (LLM passes)")
        if self.reasoning_effort:
            print(f"reasoning_effort: {self.reasoning_effort}")
        where = "remote+local-fallback" if self.use_remote else f"local {self._model_size}"
        keys = ", ".join(f"[{b['key']}]={b['label']}" for b in self.bindings)
        print(f"ready ({where}) — hold {keys}")

    @staticmethod
    def _resolve_key(name):
        try:
            return getattr(keyboard.Key, name)
        except AttributeError:
            if len(name) == 1:
                return keyboard.KeyCode.from_char(name)
            raise SystemExit(f"unknown key: {name}")

    def _set_state(self, state):
        if self.tray and state in self._icons:
            self.tray.icon = self._icons[state]
            self.tray.title = f"voice-input - {state}"

    @staticmethod
    def _cue(fname):
        """Play a short, non-blocking sound cue (audible push-to-talk feedback)."""
        try:
            subprocess.Popen(
                ["pw-play", f"{SOUNDS}/{fname}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass

    @staticmethod
    def _notify(text, source):
        """Toast the transcript titled with its backend; sync hint replaces the prior toast."""
        try:
            subprocess.Popen(
                ["notify-send", "-a", "voice-input", "-t", "2500",
                 "-h", "string:x-canonical-private-synchronous:voice-input",
                 f"🎤 {source}", text],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass

    def _start(self, binding):
        with self._lock:
            if self.recording:
                return
            self.recording = True
            self.frames = []
            self._active = binding

        sys.stdout.write(f"[rec {binding['label']}] ")
        sys.stdout.flush()
        self._set_state("recording")
        self._cue("bell.oga")  # mic is hot

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            callback=self._audio_cb,
            blocksize=1024,
        )
        self.stream.start()
        # Backstop a missed key-release: stop on our own after MAX_RECORD_SECONDS
        # so a dropped modifier can never leave the mic open / frames unbounded.
        self._watchdog = threading.Timer(MAX_RECORD_SECONDS, self._auto_stop)
        self._watchdog.daemon = True
        self._watchdog.start()

    def _audio_cb(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def _auto_stop(self):
        """Watchdog fire: a key-release was never seen. Stop as if released."""
        if self.recording:
            print(f"[auto-stop after {MAX_RECORD_SECONDS}s — key-release missed?]")
            self._stop()

    def _stop(self):
        # Runs in the pynput listener thread on key-release. Do the bare minimum
        # here (flip the flag, detach the stream) and hand the rest — including the
        # potentially-blocking PortAudio close() — to a worker, so a wedged audio
        # device can never stall the keyboard listener and freeze input.
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            binding = self._active
            stream, self.stream = self.stream, None
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        self._cue("message.oga")  # mic off, transcribing
        # The audio callback no longer appends once `recording` is False, so the
        # frames are frozen and safe to read from the worker.
        threading.Thread(target=self._finish, args=(stream, binding),
                         daemon=True).start()

    def _finish(self, stream, binding):
        """Off the listener thread: close the stream, then transcribe."""
        try:
            stream.stop()
            stream.close()
        except Exception as e:  # a wedged device must not take down the app
            print(f"[audio close failed: {e}]")

        if not self.frames:
            print("(empty)")
            self._set_state("idle")
            return

        audio = np.concatenate(self.frames).flatten()
        duration = len(audio) / SAMPLE_RATE

        if duration < 0.3:
            print(f"{duration:.1f}s — too short, skipped")
            self._set_state("idle")
            return

        sys.stdout.write(f"{duration:.1f}s ")
        sys.stdout.flush()
        self._set_state("busy")
        self._transcribe(audio, binding)

    def _ensure_local_model(self):
        if self.model is None:
            print(f"loading whisper '{self._model_size}' on {self._device} "
                  f"({self._cpu_threads} threads)...")
            self.model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute,
                cpu_threads=self._cpu_threads)

    def _transcribe(self, audio, binding):
        try:
            model, language, label = binding["model"], binding["language"], binding["label"]
            text, source = None, None
            if self.use_remote:
                try:
                    text = self._transcribe_remote(audio, model, language)
                    source = f"{label} · GPU"
                except Exception as e:  # endpoint down / network → offline fallback
                    print(f"[remote STT failed: {e} — using local]")
            if text is None:
                text = self._transcribe_local(audio, language)
                source = (f"⚠ {label}→local · {self._model_size}" if self.use_remote
                          else f"{label} · local {self._model_size}")
            text = (text or "").strip()

            if not text:
                print("(no speech)")
                return

            # Optional context-aware transform. A tool-using ('agent') pipeline is
            # interactive: it runs behind the panel, which streams progress and owns
            # the result (edit/copy/refine/insert). Plain pipelines (grammar/chat)
            # run inline and are typed like raw text. Fail-open throughout.
            images = []
            if binding.get("pipeline") or self.context_rules:
                name, steps, note = self._plan_pipeline(binding)
                if steps and any(s.get("tools") for s in steps):
                    self._set_state("refining")
                    print(f"-> [{source} → {name}] (agent panel)")
                    self._agent_panel_session(text, steps, note, f"{source} → {name}")
                    return
                if steps:  # plain inline pipeline (grammar / chat)
                    self._set_state("refining")
                    text, imgs = self._agent_steps_run(text, steps, note)
                    images.extend(imgs)
                    source = f"{source} → {name}"
                    if not text.strip() and not images:
                        print("(empty after pipeline)")
                        return

            print(f"-> [{source}] {text}")
            if text.strip():
                self._type(text)
            if images and self.image_output:
                self._emit_image(images[-1])  # most recent screenshot → clipboard
            self._notify(text or "(image)", source)
            if source.startswith("⚠"):
                self._cue("dialog-warning.oga")  # audible degraded-mode cue
        finally:
            self._set_state("idle")

    def _transcribe_local(self, audio, language):
        self._ensure_local_model()
        audio_f32 = audio.astype(np.float32) / 32768.0
        opts = {"beam_size": self._beam_size}
        if language:
            opts["language"] = language
        if self.initial_prompt:
            opts["initial_prompt"] = self.initial_prompt
        segments, _ = self.model.transcribe(audio_f32, **opts)
        return " ".join(seg.text for seg in segments)

    def _transcribe_remote(self, audio, model, language):
        """POST the clip to an OpenAI-compatible /v1/audio/transcriptions endpoint."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # int16 PCM
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())
        buf.seek(0)
        data = {"model": model}
        if language:
            data["language"] = language
        if self.initial_prompt:
            data["prompt"] = self.initial_prompt  # OpenAI's term-bias field
        resp = _http().post(
            self.remote_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data=data,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

    # ---- transform pipeline + context routing -------------------------------

    def _warn_pipeline_misconfig(self):
        """Surface common config gotchas at startup, but never crash."""
        referenced = {b.get("pipeline") for b in self.bindings if b.get("pipeline")}
        referenced |= {r.get("pipeline") for r in self.context_rules if r.get("pipeline")}
        referenced -= {None, "raw"}
        missing = referenced - set(self.pipelines)
        if missing:
            print(f"[warn: pipelines referenced but not defined: "
                  f"{sorted(missing)} — those keys will run raw]")
        if (referenced - missing) and not self.agent_url:
            print("[warn: pipelines configured but agent.url is unset — "
                  "all pipelines will be skipped (raw transcript typed)]")
        from voice_input_tools import TOOLS
        used = {t for steps in self.pipelines.values() for step in steps
                for t in (step.get("tools") or [])}
        unknown = used - set(TOOLS)
        if unknown:
            print(f"[warn: pipeline steps reference unknown tools {sorted(unknown)} "
                  f"— available: {sorted(TOOLS)}]")

    def _context(self):
        """Snapshot the focused widget on demand: spawn the AT-SPI helper for a
        single reading at dictation time (pull-based — no standing listener taxing
        the desktop). {} when disabled or on any failure (fail-open)."""
        if not self._consult_context or not os.path.exists(CONTEXT_HELPER):
            return {}
        try:
            proc = subprocess.run(
                ["/usr/bin/python3", CONTEXT_HELPER, "--once"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=2.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        try:
            return json.loads(proc.stdout or b"{}") or {}
        except ValueError:
            return {}

    @staticmethod
    def _compile_context_rules(rules):
        """Pre-compile rule regexes so bad patterns fail fast at startup, not mid-dictation."""
        compiled = []
        for i, rule in enumerate(rules):
            matchers = []
            for field, pat in rule.items():
                if field in ("pipeline", "root"):  # non-regex directives
                    continue
                try:
                    matchers.append((field, re.compile(pat)))
                except (re.error, TypeError) as e:
                    raise SystemExit(f"bad context rule #{i} field '{field}': {e}")
            compiled.append((matchers, rule))
        return compiled

    def _select_pipeline(self, binding, ctx):
        """First context rule whose every regex matches ctx wins; else binding default."""
        for matchers, rule in self._compiled_rules:
            if all(rx.search(ctx.get(field, "") or "") for field, rx in matchers):
                return rule.get("pipeline")
        return binding.get("pipeline")

    def _resolve_root(self, ctx):
        """Project root for code/file tools, from the focused window: a context
        rule's `root`, else a repo under projects_dir whose name appears in the
        window title, else the configured default. Lets the agent search the repo
        you're actually focused on instead of a fixed path."""
        wl = (ctx.get("window") or "").lower()
        for matchers, rule in self._compiled_rules:
            if rule.get("root") and all(
                    rx.search(ctx.get(f, "") or "") for f, rx in matchers):
                return os.path.expanduser(rule["root"])
        if wl and os.path.isdir(self.projects_dir):
            try:
                names = [d for d in os.listdir(self.projects_dir)
                         if os.path.isdir(os.path.join(self.projects_dir, d))]
            except OSError:
                names = []
            hits = sorted((d for d in names if d.lower() in wl), key=len, reverse=True)
            if hits:
                return os.path.join(self.projects_dir, hits[0])
        return self._base_tools_config.get("root") or os.getcwd()

    @staticmethod
    def _context_note(ctx, root):
        """Preamble telling the agent where the user is dictating + which repo."""
        bits = []
        if ctx.get("app"):
            bits.append(f"app: {ctx['app']}")
        if ctx.get("window"):
            bits.append(f"window: {ctx['window']!r}")
        if ctx.get("role"):
            bits.append(f"field: {ctx['role']}")
        note = ("The user is dictating into — " + "; ".join(bits) + ".") if bits else ""
        if root:
            note += (f"\nTheir active project root is {root}; pass that path to the "
                     "code/file tools (semantic_search, explore, file_list, read_file, "
                     "tree_sitter, lsp) unless the request clearly points elsewhere.")
        return note

    def _render_prompt(self, prompt):
        """Substitute the {glossary} token in a step prompt with the technical-term
        dictionary so the LLM normalizes mis-transcribed jargon to canonical forms.
        Prompts without the token are returned unchanged (glossary is opt-in)."""
        if "{glossary}" in prompt:
            return prompt.replace("{glossary}", self.glossary or "(none)")
        return prompt

    def _agent_step(self, text, step, ctx_note=""):
        """POST one transform step to the agent endpoint. Prefers the Anthropic
        /v1/messages API (system top-level, max_tokens required); falls back to
        the OpenAI /v1/chat/completions shape when the URL points there. A step
        with `tools` runs the agentic tool-use loop instead. ctx_note (app/window/
        repo) is appended to the system prompt of tool steps so the agent knows
        where it is. Returns (reply_text, image_paths)."""
        model, system = step.get("model") or self.agent_model, self._render_prompt(step["prompt"])
        if ctx_note and step.get("tools"):  # situational awareness for agent steps
            system = f"{system}\n\n{ctx_note}"
        is_messages = self.agent_url.rstrip("/").endswith("/messages")
        if step.get("tools") and is_messages:
            return self._agent_tool_loop(text, system, model, step["tools"])
        if is_messages:  # Anthropic Messages
            data = self._messages_call(model, system, [{"role": "user", "content": text}])
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks
                           if b.get("type") == "text").strip(), []
        # OpenAI chat-completions
        resp = _http().post(
            self.agent_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": model,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": text}]},
            timeout=self.agent_timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip(), []

    def _agent_tool_loop(self, text, system, model, tool_names, emit=None, cancel=None):
        """Gather-then-synthesize. Phase 1: let the model call tools (tool_choice
        auto), collecting their text results AND any images they produce; if it
        answers directly, use that. Phase 2 (on hitting max_steps without an
        answer): a fresh, TOOL-FREE call that hands the model the findings and
        asks for prose — search-eager models (minimax) keep searching when tools
        are present but answer cleanly when they aren't. `emit` (when set) receives
        progress events for the panel; `cancel` (an Event) stops it between steps.
        Returns (text, images)."""
        from voice_input_tools import TOOLS
        tools = [TOOLS[n]["schema"] for n in tool_names if n in TOOLS]
        messages = [{"role": "user", "content": text}]
        findings, images = [], []
        for _ in range(self.agent_max_steps):
            if cancel is not None and cancel.is_set():
                break
            data = self._messages_call(model, system, messages, tools=tools)
            blocks = data.get("content") or []
            if data.get("stop_reason") != "tool_use":
                answer = self._strip_tool_markup(
                    "".join(b.get("text", "") for b in blocks if b.get("type") == "text"))
                if answer.strip():
                    return answer.strip(), images
                break  # leaked markup only → fall through to synthesis
            messages.append({"role": "assistant", "content": blocks})
            results = []
            for b in blocks:
                if b.get("type") == "tool_use":
                    if emit:
                        emit({"type": "tool", "name": b.get("name"),
                              "input": b.get("input") or {}})
                    out_text, out_imgs = self._tool_result(b.get("name"), b.get("input") or {})
                    images.extend(out_imgs)
                    findings.append(f"{b.get('name')}({b.get('input')}):\n{out_text}")
                    results.append({"type": "tool_result", "tool_use_id": b.get("id"),
                                    "content": self._render_tool_content(out_text, out_imgs)})
            messages.append({"role": "user", "content": results})
        if emit:
            emit({"type": "status", "text": "composing answer…"})
        return self._agent_synthesize(model, system, text, findings), images

    def _agent_synthesize(self, model, system, question, findings):
        """Fresh tool-free call: write the final answer from gathered findings."""
        if not findings:
            return ""
        digest = "\n\n".join(findings)[:8000]
        msg = (f"Spoken request: {question}\n\nTool findings:\n{digest}\n\n"
               "Using only the findings above, produce the final text to insert at "
               "the cursor now. Do not call tools, do not mention tools — output only "
               "the answer.")
        data = self._messages_call(model, system, [{"role": "user", "content": msg}])
        blocks = data.get("content") or []
        return self._strip_tool_markup(
            "".join(b.get("text", "") for b in blocks if b.get("type") == "text")).strip()

    def _messages_call(self, model, system, messages, tools=None):
        """One Anthropic /v1/messages POST; returns the parsed JSON."""
        payload = {"model": model, "max_tokens": self.agent_max_tokens,
                   "system": system, "messages": messages}
        if tools:
            payload["tools"] = tools
        if self.reasoning_effort:  # "thinking on + high" — proxy maps to minimax reasoning
            payload["reasoning_effort"] = self.reasoning_effort
        resp = _http().post(
            self.agent_url,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "anthropic-version": "2023-06-01"},
            json=payload, timeout=self.agent_timeout,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _strip_tool_markup(s):
        """Safety net: remove any native tool-call markup a model leaks as text."""
        s = re.sub(r"<[^>]*tool_call>.*?</[^>]*tool_call>", "", s, flags=re.S)
        s = re.sub(r"<invoke\b.*?</invoke>", "", s, flags=re.S)
        return s

    def _tool_result(self, name, args):
        """Run one tool. Returns (text, image_paths). A tool may return a plain
        string or a dict {text, images}. Errors come back as text for the model."""
        from voice_input_tools import TOOLS
        tool = TOOLS.get(name)
        if not tool:
            return f"error: unknown tool '{name}'", []
        print(f"  [tool] {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        try:
            raw = tool["run"](args, self)
        except Exception as e:
            return f"error running {name}: {e}", []
        if isinstance(raw, dict):
            return str(raw.get("text", "")), [p for p in (raw.get("images") or []) if p]
        return str(raw), []

    def _render_tool_content(self, text, images):
        """Build the Anthropic tool_result content. With vision on, attach the
        images as base64 blocks so a vision model sees them; with vision off (the
        default — minimax is text-only), send text and just note the image paths."""
        if images and self.vision:
            content = [{"type": "text", "text": text}] if text else []
            for path in images:
                b64 = self._b64_image(path)
                if b64:
                    content.append({"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": b64}})
            return content or text
        if images:  # vision off: keep the text path so a text model still reasons
            return f"{text}\n[captured image(s): {', '.join(images)}]"
        return text

    @staticmethod
    def _b64_image(path):
        import base64
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except OSError:
            return None

    def _plan_pipeline(self, binding):
        """Pick the pipeline for the current focus and return (name, steps, note),
        scoping the code/file tools to the focused repo. (name, None, None) when
        there is nothing to run (no/raw pipeline, undefined steps, or no agent URL)."""
        ctx = self._context()
        name = self._select_pipeline(binding, ctx)
        if not name or name == "raw":
            return None, None, None
        steps = self.pipelines.get(name)
        if not steps or not self.agent_url:
            return name, None, None  # warning was already printed at startup
        root = self._resolve_root(ctx)
        self.tools_config = {**self._base_tools_config, "root": root}
        return name, steps, self._context_note(ctx, root)

    def _agent_steps_run(self, text, steps, note, emit=None, cancel=None):
        """Chain a pipeline's steps, threading the running text through each. Tool
        steps run the agentic loop (with optional panel `emit`/`cancel`); plain
        steps are a single LLM call. Fail-open: keep the best text on error.
        Returns (text, images)."""
        out, images = text, []
        for i, step in enumerate(steps):
            try:
                if step.get("tools"):
                    system = self._render_prompt(step["prompt"]) + (f"\n\n{note}" if note else "")
                    model = step.get("model") or self.agent_model
                    new, imgs = self._agent_tool_loop(
                        out, system, model, step["tools"], emit=emit, cancel=cancel)
                else:
                    new, imgs = self._agent_step(out, step, note)
            except Exception as e:
                if emit:
                    emit({"type": "error", "text": str(e)})
                print(f"[pipeline step {i} failed: {e} — keeping prior]")
                self._cue("dialog-warning.oga")
                break
            images.extend(imgs)
            if new:
                out = new
            if cancel is not None and cancel.is_set():
                break
        return out, images

    def _agent_panel_session(self, text, steps, note, source):
        """Run the agent behind the interactive panel: stream each tool call, show
        the answer in an editable box, and let the user Insert / Copy / Refine /
        Cancel before anything is typed. Owns typing + clipboard for this dictation.
        Fail-open: if the panel can't be spawned, fall back to the clipboard."""
        target = self._active_window_id()
        try:
            # System python (GTK lives there, like the AT-SPI helper); the uv
            # venv's bundled Tk is unusable under a mainloop.
            proc = subprocess.Popen(
                ["/usr/bin/python3", PANEL_HELPER, "--request", text[:300]],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        except OSError as e:
            print(f"[panel spawn failed: {e} — running headless, copying result]")
            answer, _imgs = self._agent_steps_run(text, steps, note)
            if answer:
                self._copy_text(answer)
            self._notify(answer or "(no result)", f"{source} → clipboard")
            return

        actions = queue.Queue()
        cancel = threading.Event()

        def _reader():
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    a = json.loads(line)
                except ValueError:
                    continue
                if a.get("action") == "cancel":
                    cancel.set()
                actions.put(a)
            actions.put({"action": "_eof"})

        threading.Thread(target=_reader, daemon=True).start()

        def emit(ev):
            try:
                proc.stdin.write(json.dumps(ev) + "\n")
                proc.stdin.flush()
            except (OSError, ValueError):
                pass

        emit({"type": "status", "text": "transcribed — thinking…"})
        answer, images = self._agent_steps_run(text, steps, note, emit=emit, cancel=cancel)
        emit({"type": "result", "text": answer or "(no answer)"})
        last = answer or ""

        dismissed = False
        while True:
            a = actions.get()
            act = a.get("action")
            if act in ("cancel", "_eof"):
                dismissed = (act == "cancel")
                break
            if act == "copy":
                self._copy_text(a.get("text") or last)
                self._notify("copied to clipboard", source)
                continue
            if act == "refine":
                cancel.clear()
                emit({"type": "status", "text": "refining…"})
                revised = (f"Previous answer:\n{last}\n\nRevise it per this "
                           f"instruction: {a.get('text') or ''}")
                answer, imgs = self._agent_steps_run(
                    revised, steps, note, emit=emit, cancel=cancel)
                images.extend(imgs)
                last = answer or last
                emit({"type": "result", "text": last})
                continue
            if act == "insert":
                final = a.get("text") or last
                self._wait_proc(proc)          # let the panel close so focus returns
                self._focus_window(target)
                if final.strip():
                    self._type(final)
                if images and self.image_output:
                    self._emit_image(images[-1])
                self._notify(final or "(image)", f"{source} → inserted")
                return
        self._terminate(proc)
        if not dismissed and last.strip():
            # The panel went away without an explicit choice (e.g. no display or
            # GTK missing) — don't lose the answer; leave it on the clipboard.
            self._copy_text(last)
            self._notify(last, f"{source} → clipboard")

    @staticmethod
    def _active_window_id():
        """X11 id of the focused window (to refocus before Insert), or None."""
        if os.environ.get("WAYLAND_DISPLAY"):
            return None
        try:
            r = subprocess.run(["xdotool", "getactivewindow"],
                               capture_output=True, text=True, timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() or None

    @staticmethod
    def _focus_window(wid):
        """Re-activate the window captured before the panel stole focus (X11)."""
        if not wid:
            return
        try:
            subprocess.run(["xdotool", "windowactivate", "--sync", wid],
                           check=False, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _copy_text(self, text):
        """Put plain text on the clipboard (wl-copy on Wayland, else xclip)."""
        try:
            if os.environ.get("WAYLAND_DISPLAY"):
                subprocess.run(["wl-copy"], input=text.encode(), check=False)
            else:
                subprocess.run(["xclip", "-selection", "clipboard"],
                               input=text.encode(), check=False)
        except OSError as e:
            print(f"[copy failed: {e}]")

    @staticmethod
    def _wait_proc(proc):
        try:
            proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass

    @staticmethod
    def _terminate(proc):
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass

    def _emit_image(self, path):
        """Put a tool-produced image on the clipboard so it can be pasted straight
        into the downstream prompt — the 'pass the image to the output, don't
        transform it' path. PNG via wl-copy (Wayland) or xclip (X11)."""
        try:
            if os.environ.get("WAYLAND_DISPLAY"):
                with open(path, "rb") as f:
                    subprocess.run(["wl-copy", "--type", "image/png"],
                                   stdin=f, check=False)
            else:
                subprocess.run(["xclip", "-selection", "clipboard",
                                "-t", "image/png", "-i", path], check=False)
            print(f"  [image → clipboard] {path}")
            self._notify(f"on clipboard: {os.path.basename(path)}", "📷 screenshot")
        except OSError as e:
            print(f"[image clipboard failed: {e}]")

    # ---- typing + key handlers ----------------------------------------------

    @staticmethod
    def _type(text):
        if platform.system() == "Darwin":
            escaped = text.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e", f'tell application "System Events" to keystroke "{escaped}"'],
                check=False,
            )
        elif os.environ.get("WAYLAND_DISPLAY"):
            subprocess.run(["wtype", "--", text], check=False)
        else:
            subprocess.run(
                ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
                check=False,
            )

    def _on_press(self, key):
        binding = self._keymap.get(key)
        if binding:
            self._start(binding)

    def _on_release(self, key):
        if self._active is not None and key == self._active["keyobj"]:
            self._stop()

    def _quit(self, icon=None, item=None):
        if self.tray:
            self.tray.stop()
        os._exit(0)

    def run(self):
        listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        listener.start()

        if not self.use_tray:
            listener.join()
            return

        import pystray
        self._icons = {
            "idle": _make_icon("#aaaaaa"),
            "recording": _make_icon("#e53935"),
            "busy": _make_icon("#fb8c00"),
            "refining": _make_icon("#3949ab"),
        }
        keys = " · ".join(f"[{b['key']}]={b['label']}" for b in self.bindings)
        menu = pystray.Menu(
            pystray.MenuItem(lambda _: f"hold {keys}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("quit", self._quit),
        )
        self.tray = pystray.Icon(
            "voice-input", self._icons["idle"],
            "voice-input - idle", menu=menu,
        )
        self.tray.run()


def _load_vocab(path):
    """Read a one-term-per-line glossary into a comma-joined initial_prompt."""
    try:
        with open(path) as f:
            terms = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except OSError:
        return None
    return ", ".join(terms) if terms else None


def _load_glossary(path):
    """Read a technical-term dictionary into a prompt block for the LLM passes.

    One entry per line; '#' starts a comment. A bare line is a canonical term; a
    line of the form `heard form, other form => Canonical` also lists common
    mis-transcriptions to fold back to the canonical spelling. Returns a formatted
    bullet list (or None when the file is missing/empty)."""
    try:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    except OSError:
        return None
    out = []
    for ln in lines:
        if "=>" in ln:
            heard, canon = ln.split("=>", 1)
            variants = ", ".join(v.strip() for v in heard.split(",") if v.strip())
            canon = canon.strip()
            out.append(f"- {canon}" + (f" (often mis-transcribed as: {variants})"
                                       if variants else ""))
        else:
            out.append(f"- {ln}")
    return "\n".join(out) if out else None


def _load_config(path):
    """Parse the optional TOML config. Missing → {}; malformed → SystemExit (fail-fast)."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise SystemExit(f"bad config {path}: {e}")


def _install_crash_logging():
    """Make a future freeze diagnosable. Without this the process vanished with no
    trace; now a native crash dumps a traceback, an uncaught Python exception is
    logged, and a kill signal says so — all to stderr, captured by the journal."""
    faulthandler.enable()  # dump C-level traceback on SIGSEGV/SIGABRT etc.

    def _on_signal(sig, _frame):
        print(f"[exiting on signal {signal.Signals(sig).name}]", flush=True)
        os._exit(0)

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _on_signal)

    def _thread_excepthook(args):
        import traceback
        print(f"[uncaught in thread {args.thread.name if args.thread else '?'}]",
              flush=True)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)

    threading.excepthook = _thread_excepthook


def main():
    _install_crash_logging()
    p = argparse.ArgumentParser(description="Push-to-talk voice input (two engines on two keys)")
    p.add_argument("-m", "--model", default="small",
                   help="LOCAL fallback whisper model size (tiny/base/small/medium/large-v3)")
    p.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("-l", "--language", default=None,
                   help="force this language on EVERY key (default: per-binding)")
    p.add_argument("-p", "--initial-prompt", default=None,
                   help="bias decoding toward this text (overrides --vocab)")
    p.add_argument("--vocab", default=None,
                   help="glossary file (one term per line); defaults to vocab.txt next to this script")
    p.add_argument("--remote-url",
                   default=os.environ.get("VOICE_INPUT_REMOTE_URL"),
                   help="OpenAI-compatible /v1/audio/transcriptions endpoint "
                        "(or set $VOICE_INPUT_REMOTE_URL, or [stt].url in --config). "
                        "Unset = local model only.")
    p.add_argument("--no-remote", action="store_true",
                   help="force local transcription even if an API key is available")
    p.add_argument("--no-tray", dest="tray", action="store_false",
                   help="disable system tray icon")
    p.add_argument("--config",
                   default=os.path.expanduser("~/.config/voice-input/config.toml"),
                   help="TOML config: [stt], [agent], [[pipeline.*]], [[context]] tables")
    p.add_argument("--agent-url",
                   default=os.environ.get("VOICE_INPUT_AGENT_URL"),
                   help="override [agent].url (OpenAI-compatible /v1/chat/completions)")
    p.add_argument("--no-pipeline", action="store_true",
                   help="run every key raw, even if a pipeline is configured (debug)")
    p.add_argument("--no-context-listener", action="store_true",
                   help="skip the AT-SPI focus-listener subprocess (debug)")
    args = p.parse_args()

    prompt = args.initial_prompt
    if prompt is None:
        vocab_path = args.vocab or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vocab.txt")
        prompt = _load_vocab(vocab_path)

    # API key: env wins, else the key file. Reused for STT, TTS, and chat endpoints.
    api_key = os.environ.get("VOICE_INPUT_API_KEY")
    if not api_key:
        try:
            with open(os.path.expanduser("~/.config/voice-input/api_key")) as f:
                api_key = f.read().strip()
        except OSError:
            api_key = None

    config = _load_config(args.config)

    # STT endpoint precedence: --remote-url / $VOICE_INPUT_REMOTE_URL > [stt].url in config.
    stt = config.get("stt") or {}
    remote_url = args.remote_url or stt.get("url")
    if args.no_remote:
        remote_url = None
    try:
        local_beam_size = int(stt.get("beam_size", 1))  # local-fallback decode width
    except (TypeError, ValueError):
        local_beam_size = 1
    try:
        local_cpu_threads = stt.get("cpu_threads")  # None → half the cores (headroom)
        local_cpu_threads = int(local_cpu_threads) if local_cpu_threads is not None else None
    except (TypeError, ValueError):
        local_cpu_threads = None

    bindings = [dict(b) for b in BINDINGS]
    if args.language:  # global override across all keys
        for b in bindings:
            b["language"] = args.language

    agent = dict(config.get("agent") or {})
    if args.agent_url:
        agent["url"] = args.agent_url

    # Technical-term dictionary for the LLM passes (separate from the whisper vocab
    # above): [agent].glossary, else terms.txt next to this script. Injected wherever
    # a pipeline prompt contains the {glossary} token.
    glossary_path = agent.get("glossary") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "terms.txt")
    glossary = _load_glossary(os.path.expanduser(glossary_path))
    pipelines = config.get("pipeline") or {}        # {name: [step, ...]}
    context_rules = config.get("context") or []    # [{field: regex, ..., pipeline: name}, ...]
    tools_config = config.get("tools") or {}        # {root: ..., ...} for local tools
    if args.no_pipeline:                            # debug escape hatch
        for b in bindings:
            b.pop("pipeline", None)
        context_rules = []

    VoiceInput(bindings, args.model, args.device, args.tray,
               prompt, remote_url, api_key, local_beam_size, local_cpu_threads,
               agent=agent, pipelines=pipelines, context_rules=context_rules,
               tools_config=tools_config,
               run_context_listener=not args.no_context_listener,
               glossary=glossary).run()


if __name__ == "__main__":
    main()
