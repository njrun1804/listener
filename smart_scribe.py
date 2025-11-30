#!/usr/bin/env python3
"""
Smart Push-to-Talk for ElevenLabs Scribe (macOS).

Usage:
    python3 smart_scribe.py                # Record -> Transcribe -> Paste
    python3 smart_scribe.py --clipboard    # Record -> Transcribe -> Clipboard only (no paste)
    python3 smart_scribe.py --calibrate    # Measure ambient noise, save threshold

Behavior:
    1. Hotkey triggers script (via Hammerspoon or directly)
    2. Chirp confirms listening
    3. VAD detects speech start/end
    4. Auto-stops after N seconds of silence
    5. Transcribes via ElevenLabs Scribe
    6. Pastes into active app (or clipboard-only mode)

Config & persistence:
    - API key:
        * env: ELEVENLABS_API_KEY
        * or file: ~/.smart_scribe_key (contains key only)
        * optional: "api_key" in ~/.smart_scribe.json

    - Behavior config in ~/.smart_scribe.json (auto-created on calibrate):
        {
          "input_device": 0,
          "silence_threshold": 0.05,
          "silence_duration": 1.5,
          "min_recording_duration": 1.0,
          "max_duration": 45.0,
          "language_code": "en"
        }

    - Env overrides:
        SCRIBE_SILENCE_DURATION  (float, seconds)
        SCRIBE_MIN_DURATION      (float, seconds)
        SCRIBE_MAX_DURATION      (float, seconds)
        SCRIBE_LANG              (language code, e.g. "en", "es")
        SCRIBE_APPEND_SPACE      ("0"/"false" to disable trailing space)
"""

import os
import sys
import tempfile
import subprocess
import queue
import json
import time
import wave
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

import numpy as np
import sounddevice as sd
import requests

# -------------------------------------------------------------------
# CONFIG & CONSTANTS
# -------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".smart_scribe.json"
API_KEY_PATH = Path.home() / ".smart_scribe_key"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_BLOCK_SIZE = 800              # 50ms chunks at 16kHz

DEFAULT_SILENCE_THRESHOLD = 0.05      # RMS level, use --calibrate to tune
DEFAULT_SILENCE_DURATION = 1.2        # Seconds of silence to trigger stop
DEFAULT_MIN_RECORDING_DURATION = 0.8  # Ignore very short bursts
DEFAULT_MAX_DURATION = 45.0           # Hard failsafe
DEFAULT_LANGUAGE_CODE = "en"


def load_config():
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return {}


CONFIG = load_config()


def save_config(updates: dict):
    cfg = dict(CONFIG)
    cfg.update(updates)
    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"💾 Config updated: {CONFIG_PATH}")
    except OSError as e:
        print(f"⚠️  Could not save config: {e}")


def env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.lower() not in ("0", "false", "no")


def load_api_key():
    # 1) Environment variable
    key = os.environ.get("ELEVENLABS_API_KEY")
    if key:
        return key.strip()

    # 2) Key file
    if API_KEY_PATH.exists():
        try:
            return API_KEY_PATH.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    # 3) Config JSON
    if CONFIG.get("api_key"):
        return str(CONFIG["api_key"]).strip()

    return None


SAMPLE_RATE = int(CONFIG.get("sample_rate", DEFAULT_SAMPLE_RATE))
BLOCK_SIZE = int(CONFIG.get("block_size", DEFAULT_BLOCK_SIZE))
INPUT_DEVICE = CONFIG.get("input_device", None)  # None = system default, or device index/name

SILENCE_THRESHOLD = float(CONFIG.get("silence_threshold", DEFAULT_SILENCE_THRESHOLD))
SILENCE_DURATION = env_float(
    "SCRIBE_SILENCE_DURATION",
    float(CONFIG.get("silence_duration", DEFAULT_SILENCE_DURATION)),
)
MIN_RECORDING_DURATION = env_float(
    "SCRIBE_MIN_DURATION",
    float(CONFIG.get("min_recording_duration", DEFAULT_MIN_RECORDING_DURATION)),
)
MAX_DURATION = env_float(
    "SCRIBE_MAX_DURATION",
    float(CONFIG.get("max_duration", DEFAULT_MAX_DURATION)),
)
LANGUAGE_CODE = os.environ.get(
    "SCRIBE_LANG",
    str(CONFIG.get("language_code", DEFAULT_LANGUAGE_CODE)),
)

APPEND_SPACE = env_bool("SCRIBE_APPEND_SPACE", True)

ELEVEN_KEY = load_api_key()

# Audio queue for callback
q: "queue.Queue[np.ndarray]" = queue.Queue()


# -------------------------------------------------------------------
# AUDIO FEEDBACK (macOS)
# -------------------------------------------------------------------

def chirp(sound="Tink"):
    """Play a system sound without blocking."""
    sound_path = f"/System/Library/Sounds/{sound}.aiff"
    if os.path.exists(sound_path):
        subprocess.Popen(
            ["/usr/bin/afplay", sound_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _escape_applescript(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def notify(msg, title="Scribe", sound=None):
    """macOS notification via AppleScript, with basic escaping."""
    msg_esc = _escape_applescript(str(msg))
    title_esc = _escape_applescript(str(title))
    script = f'display notification "{msg_esc}" with title "{title_esc}"'
    if sound:
        script += f' sound name "{sound}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)


# -------------------------------------------------------------------
# AUDIO HELPERS
# -------------------------------------------------------------------

def drain_queue():
    """Clear any stale audio data from previous runs."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice in a separate thread."""
    if status:
        print(f"Audio status: {status}", file=sys.stderr)
    q.put(indata.copy())


def get_rms(chunk) -> float:
    """
    Calculate RMS loudness (scaled) used for VAD.
    Rough scale: speech ~ 8–20, ambient ~ 0–5, depending on mic.
    """
    data = np.asarray(chunk, dtype=np.float32).flatten()
    if data.size == 0:
        return 0.0
    rms = np.sqrt(np.mean(data ** 2))
    return rms * 100.0  # scaling for nicer thresholds


def trim_leading_silence(audio_chunks, threshold):
    """Remove silent chunks from the beginning, keep small buffer."""
    for i, chunk in enumerate(audio_chunks):
        if get_rms(chunk) >= threshold:
            # Found first speech - keep ~150ms buffer before it
            start = max(0, i - 3)
            return audio_chunks[start:]
    return audio_chunks


def trim_trailing_silence(audio_chunks, threshold, keep_chunks=5):
    """Remove excess trailing silence, keep ~250ms for natural end."""
    for i in range(len(audio_chunks) - 1, -1, -1):
        if get_rms(audio_chunks[i]) >= threshold:
            end = min(len(audio_chunks), i + keep_chunks)
            return audio_chunks[:end]
    return audio_chunks


def write_wav_mono(path: str, sample_rate: int, data: np.ndarray):
    """Write mono float32 audio [-1,1] to 16-bit PCM WAV using stdlib."""
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    int16 = np.int16(arr * 32767)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())


# -------------------------------------------------------------------
# CALIBRATION
# -------------------------------------------------------------------

def calibrate_threshold(seconds=3):
    """Sample ambient noise to auto-set threshold and save to config."""
    print(f"🔇 Calibrating... stay quiet for {seconds}s")
    chirp("Tink")
    # Give the chirp time to end so it doesn't pollute the sample
    time.sleep(0.5)

    samples = []
    num_blocks = int(seconds * SAMPLE_RATE / BLOCK_SIZE)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=BLOCK_SIZE,
            device=INPUT_DEVICE,
        ) as stream:
            for _ in range(num_blocks):
                data, _ = stream.read(BLOCK_SIZE)
                samples.append(get_rms(data))
    except sd.PortAudioError as e:
        print(f"❌ Audio error during calibration: {e}")
        notify("Mic error during calibration – check permissions", sound="Basso")
        chirp("Basso")
        return None

    if not samples:
        print("❌ No audio samples captured during calibration")
        notify("No audio captured during calibration", sound="Basso")
        chirp("Basso")
        return None

    ambient = float(np.percentile(samples, 90))  # 90th percentile of "silence"
    suggested = ambient * 2.5                    # Speech ~2.5x ambient
    suggested = max(suggested, 0.03)             # Floor to avoid hypersensitivity

    save_config({"silence_threshold": suggested})

    chirp("Pop")
    print(f"\n📊 Results:")
    print(f"   Ambient noise (90th %ile): {ambient:.1f}")
    print(f"   Suggested threshold:       {suggested:.1f}")
    print(f"   Saved to config:           {CONFIG_PATH}\n")

    return suggested


# -------------------------------------------------------------------
# CORE RECORDING
# -------------------------------------------------------------------

def record_until_silence():
    """
    Record audio until silence is detected.
    Returns path to temp WAV file, or None on error / no speech.
    """
    drain_queue()
    chirp("Tink")  # "I'm listening"
    print("🎙️  Listening...")

    audio_chunks = []
    silent_chunks = 0
    chunk_duration = BLOCK_SIZE / SAMPLE_RATE
    silence_limit = int(SILENCE_DURATION / chunk_duration)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=audio_callback,
            blocksize=BLOCK_SIZE,
            device=INPUT_DEVICE,
        ):
            last_audio_time = time.time()

            while True:
                try:
                    indata = q.get(timeout=0.5)
                except queue.Empty:
                    # If we haven't seen any data for a while, bail out
                    if time.time() - last_audio_time > 5.0:
                        print("❌ No audio data received from device")
                        notify("No audio from mic (device issue?)", sound="Basso")
                        chirp("Basso")
                        return None
                    continue

                last_audio_time = time.time()
                audio_chunks.append(indata)
                volume = get_rms(indata)

                # Track silence
                if volume < SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                total_duration = len(audio_chunks) * chunk_duration

                # Check stop conditions
                if total_duration > MIN_RECORDING_DURATION and silent_chunks > silence_limit:
                    print("🛑 Silence detected")
                    break

                if total_duration > MAX_DURATION:
                    print("🛑 Max duration reached")
                    break

    except sd.PortAudioError as e:
        print(f"❌ Audio error: {e}")
        notify("Microphone error – check permissions", sound="Basso")
        chirp("Basso")
        return None

    chirp("Pop")  # "Got it, processing"

    # Trim silence from both ends
    audio_chunks = trim_leading_silence(audio_chunks, SILENCE_THRESHOLD)
    audio_chunks = trim_trailing_silence(audio_chunks, SILENCE_THRESHOLD)

    if not audio_chunks:
        print("⚠️  No non-silent audio captured")
        return None

    combined = np.concatenate(audio_chunks, axis=0)

    # Check if we actually captured speech
    peak_volume = max(get_rms(c) for c in audio_chunks)
    if peak_volume < SILENCE_THRESHOLD:
        print("⚠️  No speech detected (all below threshold)")
        return None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write_wav_mono(tmp.name, SAMPLE_RATE, combined)

    duration = len(combined) / SAMPLE_RATE
    print(f"📁 Recorded {duration:.1f}s -> {tmp.name}")

    return tmp.name


# -------------------------------------------------------------------
# TRANSCRIPTION
# -------------------------------------------------------------------

def transcribe(wav_path, language_code=LANGUAGE_CODE):
    """Send audio to ElevenLabs Scribe API."""
    if not ELEVEN_KEY:
        print("❌ Missing ElevenLabs API key")
        return None

    print("🧠 Transcribing...")

    url = "https://api.elevenlabs.io/v1/speech-to-text"

    try:
        with open(wav_path, "rb") as f:
            resp = requests.post(
                url,
                headers={"xi-api-key": ELEVEN_KEY},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={
                    "model_id": "scribe_v1",
                    "language_code": language_code,
                    "tag_audio_events": "false",
                },
                timeout=60,
            )

        if resp.status_code == 401:
            print("❌ Invalid API key (401)")
            chirp("Basso")
            return None

        if resp.status_code == 429:
            print("❌ Rate limited (429)")
            chirp("Basso")
            return None

        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"❌ HTTP error: {e} (status {resp.status_code})")
            print(f"Response body (truncated): {resp.text[:200]!r}")
            chirp("Basso")
            return None

        try:
            payload = resp.json()
        except ValueError:
            print("❌ Non-JSON response from API")
            print(f"Body (truncated): {resp.text[:200]!r}")
            chirp("Basso")
            return None

        text = payload.get("text", "")
        if not isinstance(text, str):
            print("⚠️  'text' field missing or not a string in response")
            return None

        text = text.strip()
        if not text:
            print("⚠️  Empty transcript from API")
            return None

        return text

    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        chirp("Basso")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        chirp("Basso")
        return None


# -------------------------------------------------------------------
# OUTPUT
# -------------------------------------------------------------------

def copy_to_clipboard(text: str):
    """Copy text to macOS clipboard."""
    p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    p.communicate(text.encode("utf-8"))


def paste_keystroke():
    """Simulate Cmd+V via AppleScript."""
    script = 'tell application "System Events" to keystroke "v" using command down'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def output_text(text, paste=True):
    """Copy to clipboard, optionally paste."""
    if not text:
        return

    if APPEND_SPACE and not text.endswith(" "):
        text = text + " "

    copy_to_clipboard(text)

    if paste:
        # Small delay to ensure clipboard is ready
        time.sleep(0.05)
        paste_keystroke()

    # Feedback
    word_count = len(text.split())
    preview = text[:30] + "..." if len(text) > 30 else text
    print(f"✅ {word_count} words: {preview}")
    chirp("Glass")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    # Parse args
    clipboard_only = "--clipboard" in sys.argv or "-c" in sys.argv
    calibrate_mode = "--calibrate" in sys.argv

    if calibrate_mode:
        result = calibrate_threshold()
        return 0 if result is not None else 1

    if not ELEVEN_KEY:
        print("❌ Set ELEVENLABS_API_KEY, ~/.smart_scribe_key, or api_key in config")
        notify("Missing ElevenLabs API key", sound="Basso")
        return 1

    try:
        # Record
        wav_path = record_until_silence()

        if not wav_path:
            notify("No speech detected or audio error", sound="Basso")
            return 1

        # Transcribe
        text = transcribe(wav_path, language_code=LANGUAGE_CODE)

        # Cleanup temp file
        try:
            os.unlink(wav_path)
        except OSError:
            pass

        if not text:
            notify("Transcription failed (see console)", sound="Basso")
            return 1

        # Output
        output_text(text, paste=not clipboard_only)
        return 0

    except KeyboardInterrupt:
        print("\n⚠️  Cancelled")
        chirp("Basso")
        return 130


if __name__ == "__main__":
    sys.exit(main())
