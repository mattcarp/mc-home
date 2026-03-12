#!/usr/bin/env python3
"""
Claudette Home — openWakeWord Listener
Uses openWakeWord for fully open-source, HA-native wake word detection.

Status (2026-03-12): Model trained and ready at models/claudette.onnx
  - 314 positive + 209 negative synthetic samples
  - 97.1% val_recall, 0% false positives on synthetic data
  - Needs real-world testing with a microphone

Setup:
  pip install openwakeword==0.6.0 pyaudio numpy soundfile

Usage:
  python3 oww_listener.py [--model models/claudette.onnx] [--threshold 0.5]

Re-train model:
  python3 generate_training_data.py --count 200   # generate synthetic data
  python3 train_claudette.py                       # train ONNX model
  python3 train_claudette.py --quick               # quick 500-step test run

Improve accuracy with real voice:
  Record 20-30 WAVs of yourself saying 'Claudette' (16kHz mono PCM WAV)
  Save to training_data/positive/real_*.wav
  Re-run: python3 train_claudette.py
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime

import numpy as np
import pyaudio
from openwakeword.model import Model


def on_wake_word_detected(word: str, score: float):
    """
    Called when 'Claudette' is detected above threshold.
    Integration point for STT pipeline (issue #10).
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] ✅ Wake word detected! word={word} score={score:.3f}")
    print("  → [TODO] Trigger STT pipeline (issue #10)")
    print("  → [TODO] Start recording until silence")
    print("  → [TODO] Send to intent parser (issue #7)")


def listen(model_path: str, threshold: float = 0.5):
    """
    Main listening loop. Runs forever until Ctrl+C.
    Audio: 16kHz mono PCM in 1620-sample chunks (openWakeWord requirement).
    """
    print(f"Loading openWakeWord model: {model_path}")
    print(f"Detection threshold: {threshold}")

    oww_model = Model(wakeword_models=[model_path])

    CHUNK = 1620   # ~0.1s at 16kHz; openWakeWord recommended
    RATE = 16000
    CHANNELS = 1
    FORMAT = pyaudio.paInt16

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=RATE,
        channels=CHANNELS,
        format=FORMAT,
        input=True,
        frames_per_buffer=CHUNK,
    )

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n🎙️  Claudette Home — listening for wake word (openWakeWord)...")
    print(f"    Model: {model_name}")
    print(f"    Sample rate: {RATE} Hz | Chunk: {CHUNK} samples")
    print("    Press Ctrl+C to stop\n")

    def shutdown(sig, frame):
        print("\nShutting down...")
        stream.close()
        pa.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    detection_count = 0
    start_time = time.time()

    while True:
        audio = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        predictions = oww_model.predict(audio_np)

        for word, score in predictions.items():
            if score >= threshold:
                detection_count += 1
                on_wake_word_detected(word, score)

        # Log stats every 60s
        elapsed = time.time() - start_time
        if int(elapsed) % 60 == 0 and int(elapsed) > 0:
            rate = detection_count / (elapsed / 3600)
            print(f"  [stats] {detection_count} detections in {elapsed/60:.0f}m ({rate:.1f}/hr)")


def main():
    parser = argparse.ArgumentParser(description="Claudette Home — openWakeWord listener")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "models", "claudette.onnx"),
        help="Path to .onnx model file (default: models/claudette.onnx)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection threshold 0.0-1.0 (default: 0.5). Higher = fewer false positives.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("  Options:")
        print("  A) Train via HA add-on: Settings > Add-ons > openWakeWord > Open Web UI")
        print("  B) Train via notebook: github.com/dscripka/openWakeWord notebooks/training_models.ipynb")
        print("  C) Use pre-trained model: python3 -c \"import openwakeword; openwakeword.utils.download_models()\"")
        sys.exit(1)

    listen(args.model, args.threshold)


if __name__ == "__main__":
    main()
