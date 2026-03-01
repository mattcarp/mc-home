#!/usr/bin/env python3
"""
Claudette Home — Porcupine Wake Word Listener
Uses Picovoice Porcupine for offline, low-latency wake word detection.

Setup:
  1. Train "claudette" model at console.picovoice.ai
  2. Download .ppn file, place in models/claudette_linux.ppn
  3. Set PORCUPINE_ACCESS_KEY in /etc/environment
  4. pip install pvporcupine pyaudio

Usage:
  python3 porcupine_listener.py [--model models/claudette_linux.ppn] [--sensitivity 0.5]
"""

import argparse
import os
import signal
import struct
import sys
import time
from datetime import datetime

import pvporcupine
import pyaudio


def on_wake_word_detected(backend: str = "porcupine"):
    """
    Called when 'Claudette' is detected.
    This is the integration point for the STT pipeline (issue #10).
    For now: prints timestamp and plays a tone/notification.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{ts}] ✅ Wake word detected! (backend={backend})")
    print("  → [TODO] Trigger STT pipeline (issue #10)")
    print("  → [TODO] Start recording until silence")
    print("  → [TODO] Send to intent parser (issue #7)")
    # Future: subprocess.run(['aplay', 'sounds/wake.wav'])
    # Future: publish to local MQTT or Unix socket for STT to pick up


def listen(model_path: str, access_key: str, sensitivity: float = 0.5):
    """
    Main listening loop. Runs forever until Ctrl+C.
    """
    print(f"Loading Porcupine model: {model_path}")
    print(f"Sensitivity: {sensitivity}")

    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[model_path],
        sensitivities=[sensitivity],
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    print(f"\n🎙️  Claudette Home — listening for wake word...")
    print(f"    Sample rate: {porcupine.sample_rate} Hz")
    print(f"    Frame length: {porcupine.frame_length} samples")
    print("    Press Ctrl+C to stop\n")

    def shutdown(sig, frame):
        print("\nShutting down...")
        stream.close()
        pa.terminate()
        porcupine.delete()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    false_positive_count = 0
    detection_count = 0
    start_time = time.time()

    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        result = porcupine.process(pcm)
        if result >= 0:
            detection_count += 1
            on_wake_word_detected(backend="porcupine")

        # Log stats every 60s (useful for tuning)
        elapsed = time.time() - start_time
        if int(elapsed) % 60 == 0 and int(elapsed) > 0:
            rate = detection_count / (elapsed / 3600)
            print(f"  [stats] {detection_count} detections in {elapsed/60:.0f}m ({rate:.1f}/hr)")


def main():
    parser = argparse.ArgumentParser(description="Claudette Home — Porcupine wake word listener")
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "models", "claudette_linux.ppn"),
        help="Path to .ppn model file (default: models/claudette_linux.ppn)",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.5,
        help="Detection sensitivity 0.0-1.0 (default: 0.5). Higher = fewer misses, more false positives.",
    )
    parser.add_argument(
        "--access-key",
        default=os.environ.get("PORCUPINE_ACCESS_KEY"),
        help="Picovoice access key (or set PORCUPINE_ACCESS_KEY env var)",
    )
    args = parser.parse_args()

    if not args.access_key:
        print("ERROR: No access key. Set PORCUPINE_ACCESS_KEY or pass --access-key")
        print("  Get yours free at console.picovoice.ai")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("  Train 'claudette' at console.picovoice.ai → download .ppn → place here")
        sys.exit(1)

    listen(args.model, args.access_key, args.sensitivity)


if __name__ == "__main__":
    main()
