# Wake Word Detection — "Claudette"

Local, offline wake word detection. No cloud. No Amazon. No Google.

## Strategy

**Prototype:** Porcupine (Picovoice) — train via console, deploy in seconds, lowest false-positive rate  
**Production:** openWakeWord — MIT license, fully open source, HA-native

Both run on-device. Both work on Workshop (x86) and target Android panel (ARM).

## Quick Start

### Porcupine (recommended for prototype)

1. Sign up at console.picovoice.ai (free, non-commercial)
2. Train custom wake word: type "claudette" → Train → download .ppn
3. Get your AccessKey from the console
4. Store in /etc/environment:
   ```
   PORCUPINE_ACCESS_KEY=your_key_here
   ```
5. Drop .ppn into voice/wake_word/models/claudette_linux.ppn
6. Run:
   ```bash
   cd voice/wake_word
   pip install -r requirements.txt
   python3 porcupine_listener.py
   ```

### openWakeWord (production / HA-native)

1. Install requirements: pip install -r requirements.txt
2. Train model (see training/ dir or HA add-on UI)
3. Drop .tflite into voice/wake_word/models/claudette.tflite
4. Run:
   ```bash
   python3 oww_listener.py --model models/claudette.tflite
   ```

## Architecture

```
Mic input (16kHz mono PCM)
    |
Wake word engine (Porcupine or openWakeWord)
    | (on detection)
STT pipeline trigger -> Whisper (issue #10)
    |
Intent parser -> HA action (issue #7)
    |
Home Assistant API bridge (issue #6)
```

## Tuning Notes

- Porcupine sensitivity: 0.5 default. Higher = fewer misses, more false positives.
- openWakeWord threshold: 0.5 default. Tune after real-world testing.
- "Claudette" — 3 syllables, French origin. Works well with both engines.
- Test in the actual room (echoes, AC noise) — not just quiet office.
