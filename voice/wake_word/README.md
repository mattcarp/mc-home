# Claudette Wake Word Detection

Local, offline wake word detection for the Claudette Home voice system.

## Overview

This module provides **privacy-first** wake word detection using [Porcupine](https://picovoice.ai/platform/porcupine/) by Picovoice. It runs entirely offline — no audio leaves your network until after the wake word is detected.

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Microphone     │────▶│  Wake Word       │────▶│  Record Command │
│  (MOES Panel)   │     │  (Porcupine)     │     │  (5 seconds)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              │
│  TTS Response   │◀────│  Intent/Action   │◀─────────────┘
│  (Speaker)      │     │  (Claudette AI)  │     (STT/Whisper)
└─────────────────┘     └──────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd /home/sysop/projects/mc-home/voice/wake_word
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get Picovoice Access Key

1. Sign up at https://console.picovoice.ai/ (free tier available)
2. Copy your Access Key
3. Set environment variable:
   ```bash
   export PICOVOICE_ACCESS_KEY="your-key-here"
   ```

### 3. Test with Built-in Keywords

```bash
python3 wake_word_detector.py --mode test
```

Say "porcupine" — you should see detection output.

## Training Custom "Claudette" Wake Word

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.

Quick version:
1. Go to https://console.picovoice.ai/ppn
2. Train wake word "claudette"
3. Download the `.ppn` model file
4. Save to `models/claudette.ppn`
5. Run: `python3 wake_word_detector.py --mode claudette`

## Files

| File | Purpose |
|------|---------|
| `wake_word_detector.py` | Core detection engine — importable class |
| `claudette_voice_loop.py` | Full pipeline integration (wake → STT → response) |
| `test_wake_word.py` | Unit tests |
| `TRAINING_GUIDE.md` | How to train custom wake word |
| `requirements.txt` | Python dependencies |

## API Usage

```python
from wake_word_detector import WakeWordDetector, create_claudette_config

# Create config
config = create_claudette_config(
    access_key="your-key",
    model_path="models/claudette.ppn"
)

# Initialize detector
detector = WakeWordDetector(config)
detector.initialize()

# Start listening
def on_wake(keyword_index):
    print("Wake word detected! Start STT...")
    # Trigger STT pipeline here

detector.start_microphone_stream(callback=on_wake)
```

## Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| Linux x86_64 | ✅ Ready | Workshop/development |
| Android | ✅ Ready | MOES panel target |
| Raspberry Pi | ✅ Ready | Future satellite terminals |
| macOS | ✅ Ready | Development on MacBook |

## Integration with STT Pipeline

When wake word is detected, the system:

1. Records 5 seconds of audio
2. POSTs to `localhost:8765/transcribe` (Whisper STT)
3. Sends transcription to Claudette (OpenClaw)
4. Receives text response
5. Plays TTS audio through panel speaker

See `../stt_pipeline/` for the Whisper STT service.

## Performance

- **Latency**: < 100ms from speech to detection
- **CPU**: < 5% on modern x86_64, runs on Raspberry Pi 4
- **Memory**: ~10MB footprint
- **Accuracy**: Configurable sensitivity (0.0-1.0)

## Testing

```bash
# Run tests (requires PICOVOICE_ACCESS_KEY)
python3 -m pytest test_wake_word.py -v

# Test with audio file
python3 wake_word_detector.py --mode file --input test_audio.wav

# List available built-in keywords
python3 claudette_voice_loop.py --list-keywords
```

## Next Steps

- [ ] Train "claudette" model for Linux (Workshop testing)
- [ ] Train "claudette" model for Android (MOES panel)
- [ ] Integrate with MOES panel hardware (April 1)
- [ ] Field test false positive rate
- [ ] Optimize sensitivity for home environment

## License

Porcupine is free for personal use. Commercial use requires a license from Picovoice.
