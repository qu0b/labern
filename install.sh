#!/usr/bin/env bash
# Install voice-input as a user service with autostart + launcher.
# Idempotent — safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
APP_SCRIPT="$SCRIPT_DIR/voice_input.py"

# 1. System deps
echo "==> installing system packages (requires sudo)"
sudo apt install -y \
    libportaudio2 \
    xdotool \
    gir1.2-ayatanaappindicator3-0.1 \
    gnome-shell-extension-appindicator \
    python3-venv

# 2. Python venv
if [[ ! -d "$VENV" ]]; then
    echo "==> creating venv"
    python3 -m venv "$VENV"
fi
echo "==> installing Python deps"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$SCRIPT_DIR/requirements.txt"

# 3. Pre-download default whisper model so first run is fast
echo "==> pre-downloading whisper base model"
"$PYTHON" -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"

# 4. Desktop entry (applications menu)
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

# 5. Autostart entry
AUTOSTART_DIR="$HOME/.config/autostart"
mkdir -p "$AUTOSTART_DIR"
cp "$APP_DIR/voice-input.desktop" "$AUTOSTART_DIR/voice-input.desktop"

# 6. Enable AppIndicator extension so the tray icon is visible in GNOME
if command -v gnome-extensions >/dev/null; then
    gnome-extensions enable ubuntu-appindicators@ubuntu.com 2>/dev/null || \
    gnome-extensions enable appindicatorsupport@rgcjonas.gmail.com 2>/dev/null || \
    echo "   (install/enable an appindicator extension manually if no icon appears)"
fi

echo
echo "==> done"
echo "  launch now:      $PYTHON $APP_SCRIPT"
echo "  or from GNOME:   Activities → Voice Input"
echo "  autostart:       enabled (runs on login)"
echo
echo "  if the tray icon doesn't appear, log out and back in so the"
echo "  AppIndicator extension picks up."
