#!/usr/bin/env bash
# Install labern (push-to-talk voice input) as a user service with autostart +
# launcher. Uses uv for the Python environment. Idempotent — safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
APP_SCRIPT="$SCRIPT_DIR/voice_input.py"

# 1. uv (Python + venv manager)
if ! command -v uv >/dev/null; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. System deps
#    AT-SPI bits (python3-gi, gir1.2-atspi-2.0) are used by voice_input_context.py
#    — labern's focus listener helper that powers context-aware pipeline routing.
echo "==> installing system packages (requires sudo)"
sudo apt install -y \
    libportaudio2 \
    xdotool \
    gir1.2-ayatanaappindicator3-0.1 \
    gnome-shell-extension-appindicator \
    python3-gi \
    gir1.2-atspi-2.0

# 3. Python environment — uv reads pyproject.toml/uv.lock and builds .venv
echo "==> syncing Python deps with uv"
uv sync --project "$SCRIPT_DIR"

# 4. Pre-download default whisper model so first run is fast (matches --model default)
echo "==> pre-downloading whisper small model"
"$PYTHON" -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# 5. Desktop entry (applications menu)
APP_DIR="$HOME/.local/share/applications"
mkdir -p "$APP_DIR"
cat > "$APP_DIR/voice-input.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Voice Input
Comment=Push-to-talk voice transcription with Whisper
Exec=$PYTHON $APP_SCRIPT
Icon=audio-input-microphone
Terminal=false
Categories=Utility;Accessibility;
StartupNotify=false
X-GNOME-UsesNotifications=true
EOF

# 6. Autostart entry
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cp "$APP_DIR/voice-input.desktop" "$AUTOSTART_DIR/voice-input.desktop"

# 7. Enable AppIndicator extension so the tray icon is visible in GNOME
if command -v gnome-extensions >/dev/null; then
    gnome-extensions enable ubuntu-appindicators@ubuntu.com 2>/dev/null || \
    gnome-extensions enable appindicatorsupport@rgcjonas.gmail.com 2>/dev/null || \
    echo "   (install/enable an appindicator extension manually if no icon appears)"
fi

# 8. Enable AT-SPI globally so apps publish focus events to the a11y bus —
#    labern's context listener needs this. GNOME ships with it off by default.
if command -v gsettings >/dev/null; then
    gsettings set org.gnome.desktop.interface toolkit-accessibility true 2>/dev/null || true
fi

echo
echo "==> done"
echo "  launch now:      uv run --project $SCRIPT_DIR voice_input.py"
echo "  or from GNOME:   Activities → Voice Input"
echo "  autostart:       enabled (runs on login)"
echo
echo "  if the tray icon doesn't appear, log out and back in so the"
echo "  AppIndicator extension picks up."
