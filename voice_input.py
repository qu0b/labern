#!/usr/bin/env python3
"""
Push-to-talk voice input: hold a key, speak, release — text appears at the cursor.

Two engines, one per key (see BINDINGS below):
    Right Ctrl  -> whisper-large-v3-turbo  (multilingual, EN+DE)
    Right Shift -> parakeet-tdt-0.6b-v2    (English, faster, native punctuation)

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
import io
import os
import platform
import subprocess
import sys
import threading
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard

SAMPLE_RATE = 16000
SOUNDS = "/usr/share/sounds/freedesktop/stereo"

# Per-key engine bindings: hold a key, its engine transcribes. language=None lets
# the engine auto-detect (whisper is multilingual; parakeet is English-only so it
# is pinned to en). Both engines share the remote endpoint + local fallback.
BINDINGS = [
    {"key": "ctrl_r",  "model": "whisper-large-v3-turbo", "language": None, "label": "whisper"},
    {"key": "shift_r", "model": "parakeet-tdt-0.6b-v2",   "language": "en", "label": "parakeet"},
]


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


class VoiceInput:
    def __init__(self, bindings, model_size, device, use_tray, initial_prompt,
                 remote_url=None, api_key=None):
        self.use_tray = use_tray
        self.initial_prompt = initial_prompt
        self.recording = False
        self.frames = []
        self.stream = None
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
        self.model = None

        if not self.use_remote:
            self._ensure_local_model()

        if self.initial_prompt:
            print(f"vocab bias: {self.initial_prompt.count(',') + 1} terms")
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
            self.tray.title = f"voice-input — {state}"

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

    def _audio_cb(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def _stop(self):
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            binding = self._active

        self._cue("message.oga")  # mic off, transcribing
        self.stream.stop()
        self.stream.close()

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
        threading.Thread(target=self._transcribe, args=(audio, binding), daemon=True).start()

    def _ensure_local_model(self):
        if self.model is None:
            print(f"loading whisper '{self._model_size}' on {self._device}...")
            self.model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute)

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

            print(f"-> [{source}] {text}")
            self._type(text)
            self._notify(text, source)
            if source.startswith("⚠"):
                self._cue("dialog-warning.oga")  # audible degraded-mode cue
        finally:
            self._set_state("idle")

    def _transcribe_local(self, audio, language):
        self._ensure_local_model()
        audio_f32 = audio.astype(np.float32) / 32768.0
        opts = {"beam_size": 5}
        if language:
            opts["language"] = language
        if self.initial_prompt:
            opts["initial_prompt"] = self.initial_prompt
        segments, _ = self.model.transcribe(audio_f32, **opts)
        return " ".join(seg.text for seg in segments)

    def _transcribe_remote(self, audio, model, language):
        """POST the clip to an OpenAI-compatible /v1/audio/transcriptions endpoint."""
        import requests
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
        resp = requests.post(
            self.remote_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            files={"file": ("audio.wav", buf, "audio/wav")},
            data=data,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")

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
        }
        keys = " · ".join(f"[{b['key']}]={b['label']}" for b in self.bindings)
        menu = pystray.Menu(
            pystray.MenuItem(lambda _: f"hold {keys}", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("quit", self._quit),
        )
        self.tray = pystray.Icon(
            "voice-input", self._icons["idle"],
            "voice-input — idle", menu=menu,
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


def main():
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
                        "(or set $VOICE_INPUT_REMOTE_URL). Unset = local model only.")
    p.add_argument("--no-remote", action="store_true",
                   help="force local transcription even if an API key is available")
    p.add_argument("--no-tray", dest="tray", action="store_false",
                   help="disable system tray icon")
    args = p.parse_args()

    prompt = args.initial_prompt
    if prompt is None:
        vocab_path = args.vocab or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vocab.txt")
        prompt = _load_vocab(vocab_path)

    # API key for the remote endpoint: env wins, else the key file.
    api_key = os.environ.get("VOICE_INPUT_API_KEY")
    if not api_key:
        try:
            with open(os.path.expanduser("~/.config/voice-input/api_key")) as f:
                api_key = f.read().strip()
        except OSError:
            api_key = None
    remote_url = None if args.no_remote else args.remote_url

    bindings = [dict(b) for b in BINDINGS]
    if args.language:  # global override across all keys
        for b in bindings:
            b["language"] = args.language

    VoiceInput(bindings, args.model, args.device, args.tray,
               prompt, remote_url, api_key).run()


if __name__ == "__main__":
    main()
