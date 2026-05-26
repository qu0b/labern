# labern

Push-to-talk voice input for your desktop: **hold a key, speak, release — the text
appears at your cursor.** Works in any focused window (editor, terminal, browser,
chat box). Transcribes locally out of the box, or offloads to a remote
OpenAI-compatible speech endpoint when you configure one.

```
[hold Right Ctrl]  "create a new branch called fix slash auth"
[release]          → create a new branch called fix-slash-auth   ⌨ typed at cursor
```

## How it works

Two engines, one per key — hold the key for as long as you speak:

| Key           | Engine                  | Notes                                   |
|---------------|-------------------------|-----------------------------------------|
| **Right Ctrl**  | `whisper-large-v3-turbo` | Multilingual (e.g. EN + DE), auto-detect |
| **Right Shift** | `parakeet-tdt-0.6b-v2`   | English only, faster, native punctuation |

A short beep marks the mic going hot; a second marks "transcribing." The result is
typed at your cursor and shown as a desktop notification titled with the backend
that produced it (so you can see at a glance whether it ran remote or local).

- **No remote endpoint configured →** everything runs locally via
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU by default).
  Fully offline.
- **Remote endpoint configured →** clips are POSTed to an OpenAI-compatible
  `/v1/audio/transcriptions` endpoint, with automatic fallback to the local model
  if the endpoint is unreachable.

The two key/engine mappings, including the remote model names, live in the
`BINDINGS` list at the top of `voice_input.py` — edit it to taste.

## Install

### Linux (X11 or Wayland)

```bash
git clone https://git.starflinger.eu/starflinger/labern.git
cd labern
./install.sh        # apt deps + venv + desktop launcher + login autostart
```

`install.sh` installs `xdotool` (X11 keystroke injection), PortAudio, and an
AppIndicator tray extension, creates a `.venv`, and registers a GNOME autostart
entry so it runs on login. Re-running it is safe (idempotent).

> **Wayland:** keystroke injection uses `wtype` instead of `xdotool` —
> `sudo apt install wtype`. (`xdotool` only works under X11.)

### macOS

```bash
brew install portaudio
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python voice_input.py
```

Typing is done via AppleScript, so grant your terminal (or Python) **Accessibility**
*and* **Input Monitoring** permission in System Settings → Privacy & Security — the
hotkey listener and keystroke injection both require it.

### Manual (any platform)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python voice_input.py
```

## Configuration

### Local only (default)

Nothing to configure — the first run downloads a small Whisper model and you're
typing by voice. Pick a bigger/smaller local model with `--model`:

```bash
python voice_input.py --model small      # tiny | base | small | medium | large-v3
python voice_input.py --device cuda       # use an NVIDIA GPU for the local model
```

### Remote endpoint (optional, faster + better quality)

Point at **any** OpenAI-compatible `/v1/audio/transcriptions` endpoint — OpenAI
itself, a self-hosted [speaches](https://github.com/speaches-ai/speaches) server,
Groq, etc. Set the URL and an API key:

```bash
export VOICE_INPUT_REMOTE_URL="https://api.openai.com/v1/audio/transcriptions"
export VOICE_INPUT_API_KEY="sk-..."        # or write it to ~/.config/voice-input/api_key
python voice_input.py
```

```bash
# Or persist the key instead of exporting it:
mkdir -p ~/.config/voice-input
printf '%s' "sk-..." > ~/.config/voice-input/api_key
```

Then make the per-key **model names match what your endpoint serves** by editing
`BINDINGS` in `voice_input.py`. For OpenAI, for example, both keys would use
`whisper-1` (OpenAI doesn't serve Parakeet); for a self-hosted speaches/Groq server
the defaults (`whisper-large-v3-turbo`, `parakeet-tdt-0.6b-v2`) work as-is.

| Setting              | Env var                  | File                              | Flag           |
|----------------------|--------------------------|-----------------------------------|----------------|
| Remote endpoint URL  | `VOICE_INPUT_REMOTE_URL` | —                                 | `--remote-url` |
| API key              | `VOICE_INPUT_API_KEY`    | `~/.config/voice-input/api_key`   | —              |

Force local even when a remote is configured with `--no-remote`.

### Vocabulary bias (Whisper engine only)

Domain jargon — usernames, product names, acronyms — transcribes better if you bias
the decoder toward it. Copy the example and edit:

```bash
cp vocab.example.txt vocab.txt    # vocab.txt is gitignored; one term per line
```

`vocab.txt` next to the script is picked up automatically; override with `--vocab
<path>` or pass freeform text with `--initial-prompt`. The startup line
`vocab bias: N terms` confirms it loaded. Keep it tight — Whisper only reads the
last ~223 tokens of the prompt and an over-long prompt hurts quality. (Parakeet
ignores the vocab.)

### Keybindings

Edit the `BINDINGS` list at the top of `voice_input.py`. Modifier keys
(`ctrl_r`, `shift_r`, `alt_r`) are ideal: single press, no character emitted, no
auto-repeat while held. Avoid `cmd`/Super (opens the desktop overview). The
laptop **Fn key is invisible to software** and cannot be bound.

## Usage

```
python voice_input.py [options]

  -m, --model SIZE      local fallback model (tiny|base|small|medium|large-v3)  [small]
  -d, --device DEV      cpu | cuda                                              [cpu]
  -l, --language LANG   force one language on EVERY key (default: per-binding)
  -p, --initial-prompt  bias decoding toward this text (overrides --vocab)
      --vocab PATH      glossary file (defaults to vocab.txt next to the script)
      --remote-url URL  OpenAI-compatible endpoint (or $VOICE_INPUT_REMOTE_URL)
      --no-remote       force local transcription even if an API key is present
      --no-tray         run headless, no system-tray icon
```

## Troubleshooting

- **No tray icon (GNOME):** install/enable an AppIndicator extension, then log out
  and back in. Or just run with `--no-tray`.
- **Nothing typed on Wayland:** install `wtype` (`xdotool` is X11-only).
- **macOS types nothing / hotkey ignored:** grant Accessibility + Input Monitoring
  to your terminal/Python.
- **First run is slow:** it's downloading the local Whisper model once; subsequent
  runs are instant.

## License

MIT — see [LICENSE](LICENSE).
