#!/usr/bin/env python3
"""End-to-end integration test for labern's transform pipeline.

Uses an OpenAI-compatible audio API both to *generate* test data (Kokoro TTS) and
to run the real path under test: it drives labern's own `_transcribe_remote` and
`_refine` methods — not reimplementations — over synthesized dictation clips.

Flow per sample:
    text  --Kokoro TTS-->  wav  --resample 16k-->  int16 audio
          --labern._transcribe_remote-->  raw transcript
          --labern._refine (chat pipeline)-->  refined text

Endpoints come from your local (gitignored) ~/.config/voice-input/config.toml —
`[stt].url` and `[agent].url`/`.model` — so nothing host-specific is committed.
The TTS URL is derived from the STT URL (`…/transcriptions` → `…/speech`). Any of
these can be overridden via env:

    LABERN_KEY        (default: /tmp/labern_key, else ~/.config/voice-input/api_key)
    LABERN_CONFIG     (default: ~/.config/voice-input/config.toml)
    LABERN_STT_URL    (default: [stt].url, else http://localhost:8080/v1/audio/transcriptions)
    LABERN_CHAT_URL   (default: [agent].url, else http://localhost:8080/v1/messages)
    LABERN_CHAT_MODEL (default: [agent].model, else minimax-m2.7)
    LABERN_TTS_URL    (default: STT URL with /speech)
    LABERN_TTS_MODEL  (default: kokoro)
    LABERN_STT_MODEL  (default: whisper-large-v3-turbo)

Run:  uv run tests/e2e_pipeline.py     (or .venv/bin/python tests/e2e_pipeline.py)
"""

import os
import sys
import wave

import numpy as np
import requests

# import labern itself (parent dir) — we test the real methods
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voice_input  # noqa: E402

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def _cfg():
    """Load the same config.toml labern uses; {} if absent — keeps endpoints out of git."""
    path = os.environ.get("LABERN_CONFIG", os.path.expanduser("~/.config/voice-input/config.toml"))
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


_C = _cfg()
STT_URL = (os.environ.get("LABERN_STT_URL") or (_C.get("stt") or {}).get("url")
           or "http://localhost:8080/v1/audio/transcriptions")
CHAT_URL = (os.environ.get("LABERN_CHAT_URL") or (_C.get("agent") or {}).get("url")
            or "http://localhost:8080/v1/messages")
CHAT_MODEL = (os.environ.get("LABERN_CHAT_MODEL") or (_C.get("agent") or {}).get("model")
              or "minimax-m2.7")
TTS_URL = os.environ.get("LABERN_TTS_URL") or STT_URL.replace("/transcriptions", "/speech")
TTS_MODEL = os.environ.get("LABERN_TTS_MODEL", "kokoro")
STT_MODEL = os.environ.get("LABERN_STT_MODEL", "whisper-large-v3-turbo")
CACHE = "/tmp/labern_testdata"

# Sample dataset: (label, spoken text, voice, pipeline). The "refine" rows exercise
# prose cleanup; the "agent" rows exercise the semantic_search tool loop over the
# labern repo (skipped if colgrep is absent).
DATASET = [
    ("grammar1", "so i was thinkin we shud add a retry to the conection pool becuase it keeps droping under load", "af_heart", "refine"),
    ("grammar2", "the function dont return nothing when the input is empty it just crash", "am_michael", "refine"),
    ("command1", "create a new branch called fix slash auth and open a pull request", "af_bella", "refine"),
    ("agent1", "how does labern fall back to a local model when the remote transcription endpoint fails", "bf_emma", "agent"),
    ("agent2", "what keybindings does labern use and which engine does each one run", "am_michael", "agent"),
]

HAVE_COLGREP = bool(__import__("shutil").which("colgrep"))


def _key():
    k = os.environ.get("LABERN_KEY")
    if k:
        return k.strip()
    for path in ("/tmp/labern_key", os.path.expanduser("~/.config/voice-input/api_key")):
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            continue
    sys.exit("no API key: set $LABERN_KEY or write /tmp/labern_key")


def synth(label, text, voice, key):
    """Kokoro TTS via LiteLLM → wav file (cached). Returns the path."""
    path = os.path.join(CACHE, f"{label}.wav")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    os.makedirs(CACHE, exist_ok=True)
    r = requests.post(TTS_URL, headers={"Authorization": f"Bearer {key}"},
                      json={"model": TTS_MODEL, "input": text,
                            "voice": voice, "response_format": "wav"}, timeout=90)
    r.raise_for_status()
    if not r.content[:4] == b"RIFF":
        sys.exit(f"TTS for {label} did not return a WAV: {r.content[:200]!r}")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def load_16k_int16(path):
    """Read a WAV and return mono int16 @16kHz — what labern's mic capture yields."""
    with wave.open(path, "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)
    if sr != voice_input.SAMPLE_RATE:  # 24k -> 16k linear resample (fine for STT)
        tgt = int(len(audio) * voice_input.SAMPLE_RATE / sr)
        xp = np.linspace(0, 1, len(audio), endpoint=False)
        x = np.linspace(0, 1, tgt, endpoint=False)
        audio = np.interp(x, xp, audio.astype(np.float32)).astype(np.int16)
    return audio


def main():
    key = _key()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # A VoiceInput wired for remote STT + two pipelines: prose "refine" and a
    # tool-using "agent" (semantic_search over this repo).
    vi = voice_input.VoiceInput(
        bindings=[{"key": "shift_r", "model": STT_MODEL, "language": None,
                   "label": "test", "pipeline": "refine"}],
        model_size="small", device="cpu", use_tray=False, initial_prompt=None,
        remote_url=STT_URL, api_key=key,
        agent={"url": CHAT_URL, "model": CHAT_MODEL, "timeout": 90, "max_steps": 4},
        pipelines={
            "refine": [{"prompt":
                "Fix spelling, grammar, and punctuation in this dictated text. "
                "Output ONLY the corrected text, nothing else."}],
            "agent": [{"tools": ["semantic_search"], "prompt":
                "You are a dictation assistant with tools. Use semantic_search to "
                "locate relevant code, then write a concise answer for the cursor."}],
        },
        context_rules=[], tools_config={"root": repo_root},
        run_context_listener=False,
    )
    binding = vi.bindings[0]

    print(f"\n{'='*72}\nlabern e2e: TTS → _transcribe_remote → _refine ({CHAT_MODEL})\n{'='*72}")
    failures = 0
    for label, text, voice, pipeline in DATASET:
        if pipeline == "agent" and not HAVE_COLGREP:
            print(f"\n[{label}]  SKIPPED (colgrep not installed)")
            continue
        wav = synth(label, text, voice, key)
        audio = load_16k_int16(wav)
        raw = (vi._transcribe_remote(audio, STT_MODEL, None) or "").strip()
        binding["pipeline"] = pipeline
        out, pipe, _images = vi._refine(raw, binding)
        print(f"\n[{label}]  voice={voice}  pipeline={pipe}")
        print(f"  spoken : {text}")
        print(f"  STT    : {raw}")
        print(f"  output : {out}")
        if not raw:
            print("  !! STT returned empty"); failures += 1
        if not out.strip():
            print("  !! pipeline returned empty"); failures += 1

    print(f"\n{'='*72}\n{'FAILED' if failures else 'OK'}: "
          f"{len(DATASET)} samples, {failures} problem(s)\n{'='*72}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
