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

### Switching engine without a restart

Right-click the tray icon → pick a key → pick an engine. It applies to that
key's **next** dictation; nothing restarts, no config is edited, and the menu
radio-checks whichever engine the key is actually on.

```
🎤 voice-input
   hold [ctrl_r]=raw · [shift_r]=clean · [alt_r]=agent
   ─────────────
   raw   [ctrl_r]   ▸   ● whisper (multilingual)
   clean [shift_r]  ▸   ○ parakeet (EN, fast)
   agent [alt_r]    ▸   ○ ink-2 (Cartesia)
   ─────────────
   quit
```

The list comes from `[stt.catalog]` in `config.toml` (same entry shape as
`[stt.models]`); omit it for the built-in whisper + parakeet pair. Whatever a
key starts on is always listed, so a `[stt.models]` override stays visible and
switchable.

## Install

### Linux (X11 or Wayland)

```bash
git clone https://github.com/qu0b/labern.git
cd labern
./install.sh        # uv env + apt deps + desktop launcher + login autostart
```

`install.sh` installs [uv](https://docs.astral.sh/uv/) (if missing), `xdotool`
(X11 keystroke injection), PortAudio, and an AppIndicator tray extension, runs
`uv sync` to build the `.venv` from `pyproject.toml`, and registers a GNOME
autostart entry so it runs on login. Re-running it is safe (idempotent).

> **Wayland:** keystroke injection uses `wtype` instead of `xdotool` —
> `sudo apt install wtype`. (`xdotool` only works under X11.)

### macOS

```bash
brew install uv portaudio
uv run voice_input.py        # uv builds the .venv from pyproject.toml on first run
```

Typing is done via AppleScript, so grant your terminal (or Python) **Accessibility**
*and* **Input Monitoring** permission in System Settings → Privacy & Security — the
hotkey listener and keystroke injection both require it.

### Manual (any platform)

Requires [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
uv run voice_input.py        # resolves + installs deps into .venv, then runs
```

## Configuration

### Local only (default)

Nothing to configure — the first run downloads a small Whisper model and you're
typing by voice. Pick a bigger/smaller local model with `--model`:

```bash
uv run voice_input.py --model small      # tiny | base | small | medium | large-v3
uv run voice_input.py --device cuda       # use an NVIDIA GPU for the local model
```

### Remote endpoint (optional, faster + better quality)

Point at **any** OpenAI-compatible `/v1/audio/transcriptions` endpoint — OpenAI
itself, a self-hosted [speaches](https://github.com/speaches-ai/speaches) server,
Groq, etc. Set the URL and an API key:

```bash
export VOICE_INPUT_REMOTE_URL="https://api.openai.com/v1/audio/transcriptions"
export VOICE_INPUT_API_KEY="sk-..."        # or write it to ~/.config/voice-input/api_key
uv run voice_input.py
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

### Transform pipeline & context routing (optional)

The transcript can flow through a configurable **pipeline of LLM steps** before
being typed — to clean grammar, reformat as a chat message, translate, or hand
off to a tool-using agent. Each step calls an LLM endpoint: the **Anthropic
Messages API (`/v1/messages`) is preferred**, with OpenAI `/v1/chat/completions`
also supported (labern picks the request shape from the URL). A *context layer*
then picks which pipeline runs based on the focused widget (Slack vs. email vs.
terminal). Without a config file the feature is fully inert and labern behaves
exactly as before.

```bash
mkdir -p ~/.config/voice-input
cp config.example.toml ~/.config/voice-input/config.toml
# edit ~/.config/voice-input/config.toml to taste
```

The example config documents every field; the shape is:

```toml
[stt]                                # remote transcription endpoint (optional;
url     = "..."                      # else --remote-url/$VOICE_INPUT_REMOTE_URL, or local)

[agent]                              # Anthropic /v1/messages (or OpenAI
url     = "..."                      # /v1/chat/completions). Reuses ~/.config/voice-input/api_key.
model   = "minimax-m2.7"
timeout = 60

[[pipeline.refine]]                  # named pipelines: lists of steps. Each step
prompt = "Fix grammar."              # POSTs system=prompt, user=running-text;
                                     # the reply becomes the next step's input.

[[context]]                          # context rules: first match wins; no match
app  = "slack|mattermost"            # → the binding's default pipeline. Fields
role = "text"                        # are Python regexes (re.search), matched
pipeline = "refine"                  # against the focused-widget snapshot.
```

**Two kinds of step.** A plain step POSTs text and pastes the reply (grammar,
formatting, translation). A step with `tools = [...]` runs an **agentic
tool-use loop**: labern sends the transcript plus tool schemas, the model
requests a tool, labern runs it *locally* and feeds the result back, repeating
until the model answers (capped by `[agent].max_steps`, then a tool-free
synthesis call forces a final answer). Tools are registered in
`voice_input_tools.py` and run on your machine:

| Tool | What it does | Needs |
|------|--------------|-------|
| `explore` | sub-agent that maps **where** relevant code lives (delegate broad "how/where is X" investigation) | — |
| `semantic_search` | semantic code search over `[tools].root` via colgrep | `colgrep` |
| `file_list` / `tree` | list files (gitignore-aware) / directory tree | — |
| `read_file` | read a file or line range | — |
| `tree_sitter` | symbol/structure outline of a source file | `--extra code` |
| `lsp` | hover / definition / references / diagnostics | `--extra code` |
| `exa_search` / `exa_contents` | web search + page content via Exa | `[tools].exa_api_key` |
| `browser_use` | ad-hoc browser screenshot + page text | `browser-use` CLI |
| `playwright` | screenshot a URL at an **exact viewport size** | `--extra browser` |
| `shell` | run ONE allowlisted, read-only command (no shell operators) | — |

Tools live in `voice_input_tools.py` + `tools_*.py`; a missing backend disables
only that tool. Install tool deps per group: `uv sync --extra code --extra browser`.

So "find where we handle auth errors and summarize" searches your code and types
a grounded answer; "screenshot example.com at 1024×768" captures the page. (Tool
execution uses the Anthropic `tool_use` blocks of `/v1/messages`.)

**Images: vision-ready + clipboard pass-through.** Tools that produce images
(the browser tools) feed them back two ways, both configurable under `[agent]`:

- `vision = true` attaches screenshots to the model as image blocks — set this
  only with a vision model (e.g. `qwen3-vl`); text-only models (minimax) keep it
  `false` and receive the page text instead.
- `image_output = true` (default) copies the screenshot to your **clipboard**, so
  you can paste it straight into whatever you're prompting — the image reaches the
  output without the pipeline model having to describe it.

**Per-key default.** `shift_r` defaults to the `agent` pipeline; `ctrl_r`
stays raw. Override either by editing `BINDINGS` in `voice_input.py` or by
writing context rules. The reserved pipeline name `"raw"` disables the pipeline
for a context.

**Context signal.** A small daemon (`voice_input_context.py`, system Python)
subscribes to AT-SPI `object:state-changed:focused` events and writes
`~/.cache/labern/context.json` on every focus change — push-based, ~5–50 ms
latency, X11 and Wayland. Snapshot fields available to rules:

| Field | Meaning |
|---|---|
| `app`    | AT-SPI application name (e.g. `slack`, `gnome-terminal-server`) |
| `role`   | Focused widget's role (`text`, `terminal`, `push button`, …) |
| `name`   | Focused widget's accessible name (often the field label) |
| `window` | Toplevel window's accessible name (usually the window title) |

**Requirement:** AT-SPI must be enabled — `install.sh` sets
`gsettings set org.gnome.desktop.interface toolkit-accessibility true` for you.
Without it apps don't emit focus events and rules stay inert (binding defaults
still run).

**Fail-open everywhere.** Endpoint down, helper crash, malformed step reply,
no matching rule, AT-SPI bus missing → labern types the best text it has and
plays a warning cue. Dictation is never lost.

**Per-repo terminal routing (optional).** gnome-terminal's per-tab cwd lives
inside VTE and isn't published externally. If you want repo-precise routing
inside a terminal, set the terminal *title* to your `$PWD` (or the git-root
basename) from your prompt — AT-SPI then surfaces it as the window's
accessible name, which a context rule can match:

```sh
# zsh (~/.zshrc):
precmd() { print -Pn "\e]2;${PWD:t}\a"; }
# bash (~/.bashrc):
PROMPT_COMMAND='printf "\e]2;%s\a" "${PWD##*/}"'
```

Or use a terminal that publishes per-pane cwd as a first-class API —
`kitty @ ls` or `wezterm cli list --format json` both ship that out of the box.

**Privacy.** The context cache holds only widget identity (app/role/name/window)
— not text content, not screenshots, not surrounding text. The chat endpoint
sees only the transcript (plus any system prompts you author).

### Keybindings

Edit the `BINDINGS` list at the top of `voice_input.py`. Modifier keys
(`ctrl_r`, `shift_r`, `alt_r`) are ideal: single press, no character emitted, no
auto-repeat while held. Avoid `cmd`/Super (opens the desktop overview). The
laptop **Fn key is invisible to software** and cannot be bound.

## Usage

```
uv run voice_input.py [options]

  -m, --model SIZE       local fallback model (tiny|base|small|medium|large-v3) [small]
  -d, --device DEV       cpu | cuda                                             [cpu]
  -l, --language LANG    force one language on EVERY key (default: per-binding)
  -p, --initial-prompt   bias decoding toward this text (overrides --vocab)
      --vocab PATH       glossary file (defaults to vocab.txt next to the script)
      --remote-url URL   OpenAI-compatible STT endpoint (or $VOICE_INPUT_REMOTE_URL)
      --no-remote        force local transcription even if an API key is present
      --no-tray          run headless, no system-tray icon
      --config PATH      pipeline+context TOML  [~/.config/voice-input/config.toml]
      --agent-url URL    override [agent].url (or $VOICE_INPUT_AGENT_URL)
      --no-pipeline      run every key raw, even with a pipeline configured
      --no-context-listener   skip the AT-SPI focus listener (context rules inert)
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
