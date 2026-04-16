# Claudette Wake Word Training Guide

This guide explains how to train a custom "Claudette" wake word model using Picovoice Porcupine.

## Overview

**Porcupine** is a lightweight, local wake word engine that runs entirely offline. It's free for personal use and runs on everything from Raspberry Pi to Android devices.

## Step 1: Get a Picovoice Account

1. Go to https://console.picovoice.ai/
2. Sign up (free tier available)
3. Copy your `Access Key` from the dashboard

## Step 2: Train the Custom Wake Word

1. In the Picovoice Console, navigate to **Porcupine** → **Train Wake Word**
2. Enter "claudette" as the wake word
3. Select language: **English (US)**
4. Choose platform:
   - For MOES panel (Android): Select **Android (.ppn)**
   - For Workshop (Linux x86_64): Select **Linux (.ppn)**
   - For Raspberry Pi: Select **Raspberry Pi (.ppn)**
5. Click **Train** and download the `.ppn` file

## Step 3: Download the Model

The downloaded file will be named something like `claudette_android.ppn` or `claudette_linux.ppn`.

Save it to:
```
/home/sysop/projects/mc-home/voice/wake_word/models/claudette.ppn
```

## Step 4: Set Environment Variable

```bash
export PICOVOICE_ACCESS_KEY="your-access-key-here"
```

Add to `~/.bashrc` or `/etc/environment` for persistence.

## Step 5: Test the Wake Word

```bash
cd /home/sysop/projects/mc-home/voice/wake_word
source venv/bin/activate
python3 wake_word_detector.py --mode claudette
```

Say "Claudette" — you should see:
```
[WAKE WORD DETECTED] 'claudette' at 14:32:15
  -> Wake word triggered! Starting STT pipeline...
```

## Hardware Targets

### Workshop (Linux x86_64)
- Platform: Linux x86_64
- Use case: Development, testing, server-side processing
- Model suffix: `_linux.ppn`

### MOES Panel (Android)
- Platform: Android
- Use case: In-wall voice terminal
- Model suffix: `_android.ppn`
- Note: Porcupine has an Android SDK

### Future: Raspberry Pi
- Platform: Raspberry Pi (ARM)
- Use case: Low-cost satellite voice terminals
- Model suffix: `_raspberry-pi.ppn`

## Tips for Best Accuracy

1. **Say it naturally** — don't over-enunciate during training
2. **Multiple accents** — if multiple people will use it, record samples with different accents
3. **Background noise** — train with some ambient noise samples if possible
4. **Sensitivity tuning** — adjust the `sensitivity` parameter (0.0-1.0):
   - Lower (0.3-0.5): Fewer false positives, might miss some detections
   - Higher (0.7-0.9): More detections, might have false triggers

## Testing Without Hardware

Use the test mode with built-in keywords:

```bash
python3 wake_word_detector.py --mode test
```

Try saying "porcupine" or "hey google" to verify the pipeline works.

## Integration with STT Pipeline

When wake word is detected, the detector should:

1. Start recording audio buffer
2. Send to STT service at `localhost:8765/transcribe`
3. Wait for transcription
4. Send transcript to Claudette (OpenClaw)
5. Play TTS response through panel speaker

See `../stt_pipeline/` for the STT service implementation.

## Files Generated

| File | Purpose |
|------|---------|
| `wake_word_detector.py` | Main detection engine |
| `models/claudette.ppn` | Custom wake word model (download from Picovoice) |
| `test_samples/` | Audio samples for testing |
| `requirements.txt` | Python dependencies |

## Next Steps

1. [ ] Create Picovoice account and get access key
2. [ ] Train "claudette" wake word for Linux (Workshop testing)
3. [ ] Train "claudette" wake word for Android (MOES panel)
4. [ ] Test detection accuracy
5. [ ] Integrate with STT pipeline
6. [ ] Deploy to MOES panel on April 1

## Resources

- Porcupine docs: https://picovoice.ai/docs/api/porcupine-python/
- GitHub: https://github.com/Picovoice/porcupine
- Console: https://console.picovoice.ai/
