# Training "Claudette" Wake Word Models

## Option A: Porcupine (Recommended for Prototype)

**Time:** ~5 minutes  
**Requires:** Browser, Picovoice account (free)

1. Go to https://console.picovoice.ai
2. Sign up (free, non-commercial OK)
3. Porcupine → Custom Wake Word → New Wake Word
4. Type: `claudette`
5. Preview pronunciation — if wrong, try `claw-det` or `clo-det` phonetically
6. Click Train (takes ~30 seconds)
7. Download for Linux (x86_64) → `claudette_linux.ppn`
8. Also download for Android (arm64-v8a) → for future panel deployment
9. Copy `.ppn` file to `voice/wake_word/models/`

**Sensitivity tuning:**
- Start at 0.5
- If missing real detections: increase to 0.6-0.7
- If triggering on background speech: decrease to 0.3-0.4

---

## Option B: openWakeWord via HA Add-On (Production)

**Time:** 10-60 minutes depending on training examples  
**Requires:** Home Assistant running (issue #12)

1. HA → Settings → Add-ons → Add-on Store → openWakeWord → Install
2. Start add-on → Open Web UI
3. Enter wake word: `claudette`
4. Click play to preview TTS pronunciation
5. Set training examples: 30,000 (more = better accuracy, slower training)
6. Click Train
7. Download `.tflite` file
8. Copy to `voice/wake_word/models/claudette.tflite`

**Note:** HA isn't installed yet (issue #12). Use Porcupine for now.

---

## Option C: openWakeWord Standalone (No HA Required)

**Time:** 1-2 hours  
**Requires:** Python 3.9+, ~4GB disk for training data

```bash
git clone https://github.com/dscripka/openWakeWord
cd openWakeWord

# Install training deps
pip install -r requirements_training.txt

# Open training notebook
jupyter notebook notebooks/training_models.ipynb
```

Follow notebook instructions:
- Wake word: `claudette`
- Examples: 30,000 synthetic (auto-generated via Piper TTS)
- Negatives: use included background noise dataset
- Export to `.tflite`

---

## Testing After Training

```bash
# Quick test — should detect when you say "Claudette"
python3 porcupine_listener.py  # Porcupine
python3 oww_listener.py        # openWakeWord

# Target false positive rate: < 1 per hour
# Target detection rate: > 95% of clear utterances
```

## Notes on "Claudette" Pronunciation

- 3 syllables: clo-DET (stress on second)
- French feminine name — Piper TTS handles it well
- If training on Porcupine console, the preview audio is accurate
- Avoid training on "claudette" vs "clawed-it" — they sound similar
