# Smart Scribe

Push-to-talk voice transcription for macOS using [ElevenLabs Scribe](https://elevenlabs.io/speech-to-text) with automatic Voice Activity Detection (VAD).

## Features

- **Push-to-talk**: Trigger via hotkey, auto-stops when you stop speaking
- **VAD**: Automatically detects speech start/end - no manual stop needed
- **Fast**: Transcribes and pastes directly into your active app
- **Calibration**: Auto-tune silence threshold for your environment
- **Configurable**: JSON config, environment variables, or both

## Requirements

- macOS (uses system sounds, AppleScript, pbcopy)
- Python 3.8+
- [PortAudio](https://www.portaudio.com/) (for audio capture)
- [ElevenLabs API key](https://elevenlabs.io/)

## Installation

### 1. Install system dependencies

```bash
brew install portaudio
```

### 2. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

Or manually:
```bash
pip3 install sounddevice numpy requests
```

### 3. Set up your API key

Choose one method:

```bash
# Option A: Environment variable (add to ~/.zshrc or ~/.bashrc)
export ELEVENLABS_API_KEY="your-key-here"

# Option B: Key file
echo "your-key-here" > ~/.smart_scribe_key

# Option C: In config file (~/.smart_scribe.json)
# { "api_key": "your-key-here" }
```

### 4. Calibrate for your environment

```bash
python3 smart_scribe.py --calibrate
```

Stay quiet for 3 seconds. This measures ambient noise and saves an appropriate silence threshold to `~/.smart_scribe.json`.

### 5. Test it

```bash
python3 smart_scribe.py
```

Speak after the chirp. It will auto-stop after ~1.2s of silence, transcribe, and paste the text.

## Usage

```bash
# Record → Transcribe → Paste into active app
python3 smart_scribe.py

# Record → Transcribe → Clipboard only (no paste)
python3 smart_scribe.py --clipboard

# Calibrate silence threshold
python3 smart_scribe.py --calibrate
```

## Hotkey Setup with Hammerspoon

[Hammerspoon](https://www.hammerspoon.org/) lets you bind global hotkeys.

### 1. Install Hammerspoon

```bash
brew install --cask hammerspoon
```

### 2. Add to `~/.hammerspoon/init.lua`

See [examples/hammerspoon.lua](examples/hammerspoon.lua) for a complete example.

### 3. Reload Hammerspoon config

Click the Hammerspoon menu bar icon → Reload Config

### Default hotkeys

| Hotkey | Action |
|--------|--------|
| `Ctrl+Alt+Space` | Record → Transcribe → Paste |
| `Ctrl+Alt+Shift+Space` | Record → Transcribe → Clipboard only |

Press the hotkey again while recording to cancel.

## Configuration

Config is stored in `~/.smart_scribe.json` (auto-created on calibrate):

```json
{
  "silence_threshold": 12.0,
  "silence_duration": 1.2,
  "min_recording_duration": 0.8,
  "max_duration": 45.0,
  "language_code": "en"
}
```

### Environment variable overrides

| Variable | Description | Default |
|----------|-------------|---------|
| `ELEVENLABS_API_KEY` | API key | - |
| `SCRIBE_SILENCE_DURATION` | Seconds of silence to stop | 1.2 |
| `SCRIBE_MIN_DURATION` | Minimum recording length | 0.8 |
| `SCRIBE_MAX_DURATION` | Maximum recording length | 45.0 |
| `SCRIBE_LANG` | Language code | en |
| `SCRIBE_APPEND_SPACE` | Add trailing space | true |

## macOS Permissions

Grant these permissions in **System Settings → Privacy & Security**:

| Permission | App |
|------------|-----|
| **Microphone** | Terminal (or iTerm, Hammerspoon) |
| **Accessibility** | Hammerspoon (for Cmd+V paste simulation) |

## Audio feedback

| Sound | Meaning |
|-------|---------|
| Tink | Listening started |
| Pop | Recording captured, processing |
| Glass | Success, text pasted |
| Basso | Error occurred |

## Troubleshooting

### "No audio from mic"
- Check microphone permissions in System Settings
- Try a different input device: `python3 -c "import sounddevice; print(sounddevice.query_devices())"`

### "Missing ElevenLabs API key"
- Ensure one of the three API key methods is configured
- Check the key has no extra whitespace

### Transcription too sensitive / not sensitive enough
- Run `--calibrate` in a quiet environment
- Or manually adjust `silence_threshold` in `~/.smart_scribe.json`

### Hotkey doesn't work
- Check Hammerspoon has Accessibility permission
- Verify config is loaded: Hammerspoon Console → check for errors

## License

MIT
