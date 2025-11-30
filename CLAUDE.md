# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Smart Scribe is a macOS push-to-talk voice transcription tool that uses ElevenLabs Scribe API with automatic Voice Activity Detection (VAD). It records audio, detects when speech ends via silence detection, transcribes via API, and pastes the result into the active app.

## Commands

```bash
# Run transcription (record → transcribe → paste)
python3 smart_scribe.py

# Clipboard only (no auto-paste)
python3 smart_scribe.py --clipboard

# Calibrate silence threshold for your mic/environment
python3 smart_scribe.py --calibrate

# List available audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Install dependencies
brew install portaudio
pip3 install -r requirements.txt
```

## Architecture

Single-file Python script (`smart_scribe.py`) with these major sections:

- **Config loading** (lines 76-159): Reads `~/.smart_scribe.json` and `~/.smart_scribe_key`, with env var overrides
- **Audio feedback** (lines 165-191): `chirp()` plays system sounds, `notify()` shows macOS notifications
- **Audio helpers** (lines 194-254): Queue-based audio capture, RMS calculation for VAD, silence trimming, WAV writing
- **Calibration** (lines 257-305): Samples ambient noise to auto-set `silence_threshold`
- **Recording** (lines 308-399): Main recording loop with VAD - stops after `silence_duration` seconds of silence
- **Transcription** (lines 402-475): POST to ElevenLabs `/v1/speech-to-text` endpoint
- **Output** (lines 478-513): Copy to clipboard via `pbcopy`, paste via AppleScript keystroke simulation

## Key Configuration

Config file: `~/.smart_scribe.json`
```json
{
  "input_device": 0,
  "silence_threshold": 0.05,
  "silence_duration": 1.5,
  "min_recording_duration": 1.0,
  "max_duration": 45.0,
  "language_code": "en"
}
```

API key sources (checked in order):
1. `ELEVENLABS_API_KEY` env var
2. `~/.smart_scribe_key` file
3. `api_key` in config JSON

## Hammerspoon Integration

Global hotkeys are configured via `~/.hammerspoon/init.lua`. The Lua script spawns the Python script via `/bin/bash -c`. See `examples/hammerspoon.lua` for reference.

## Platform Notes

- macOS only (uses `afplay`, `pbcopy`, `osascript`, System Events)
- Requires Microphone permission for Terminal/Hammerspoon
- Requires Accessibility permission for Hammerspoon (paste simulation)
- Default input device may need explicit `input_device` config if system default is wrong (e.g., virtual audio devices)
