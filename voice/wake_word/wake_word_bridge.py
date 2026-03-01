#!/usr/bin/env python3
"""
Claudette Home — Wake Word Bridge
Unified interface that wraps either Porcupine or openWakeWord.
Swap the backend without touching the rest of the pipeline.

Usage:
  python3 wake_word_bridge.py --backend porcupine --model models/claudette_linux.ppn
  python3 wake_word_bridge.py --backend oww --model models/claudette.tflite

Environment:
  WAKE_WORD_BACKEND=porcupine|oww   (default: porcupine)
  PORCUPINE_ACCESS_KEY=...          (required for porcupine backend)
  WAKE_WORD_MODEL=path/to/model     (optional, overrides --model default)
  WAKE_WORD_THRESHOLD=0.5           (optional, detection threshold)

This script is the entry point for the systemd service (future).
On detection: writes event to stdout (JSON) and triggers callback.
Other services (STT, intent parser) can listen to that stream.
"""

import argparse
import json
import os
import signal
import struct
import sys
import time
from datetime import datetime, timezone


def emit_event(event_type: str, data: dict):
    """Write a JSON event line to stdout. Downstream services can tail this."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        **data,
    }
    print(json.dumps(event), flush=True)


def on_detection(backend: str, word: str, score: float = None):
    """Called on wake word detection. Emits event for downstream pipeline."""
    emit_event("wake_word_detected", {
        "backend": backend,
        "word": word,
        "score": score,
    })
    # TODO (issue #10): Signal STT pipeline to start recording
    # TODO (issue #7): After STT, pass transcript to intent parser


def run_porcupine(model_path: str, access_key: str, sensitivity: float):
    """Run Porcupine backend."""
    import struct
    import pvporcupine
    import pyaudio

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

    emit_event("listener_started", {
        "backend": "porcupine",
        "model": model_path,
        "sensitivity": sensitivity,
        "sample_rate": porcupine.sample_rate,
    })

    def shutdown(sig, frame):
        stream.close()
        pa.terminate()
        porcupine.delete()
        emit_event("listener_stopped", {"backend": "porcupine"})
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        if porcupine.process(pcm) >= 0:
            on_detection("porcupine", "claudette")


def run_oww(model_path: str, threshold: float):
    """Run openWakeWord backend."""
    import numpy as np
    import pyaudio
    from openwakeword.model import Model

    oww_model = Model(wakeword_models=[model_path])
    CHUNK = 1620
    pa = pyaudio.PyAudio()
    stream = pa.open(rate=16000, channels=1, format=pyaudio.paInt16,
                     input=True, frames_per_buffer=CHUNK)

    emit_event("listener_started", {
        "backend": "oww",
        "model": model_path,
        "threshold": threshold,
    })

    def shutdown(sig, frame):
        stream.close()
        pa.terminate()
        emit_event("listener_stopped", {"backend": "oww"})
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        audio = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        for word, score in oww_model.predict(audio_np).items():
            if score >= threshold:
                on_detection("oww", word, score)


def main():
    parser = argparse.ArgumentParser(description="Claudette Home — Wake Word Bridge")
    parser.add_argument(
        "--backend",
        default=os.environ.get("WAKE_WORD_BACKEND", "porcupine"),
        choices=["porcupine", "oww"],
        help="Wake word backend (default: porcupine)"
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("WAKE_WORD_MODEL"),
        help="Path to model file (.ppn for porcupine, .tflite for oww)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("WAKE_WORD_THRESHOLD", "0.5")),
        help="Detection threshold (default: 0.5)"
    )
    parser.add_argument(
        "--access-key",
        default=os.environ.get("PORCUPINE_ACCESS_KEY"),
        help="Picovoice access key (porcupine backend only)"
    )
    args = parser.parse_args()

    # Default model paths by backend
    if not args.model:
        base = os.path.dirname(__file__)
        if args.backend == "porcupine":
            args.model = os.path.join(base, "models", "claudette_linux.ppn")
        else:
            args.model = os.path.join(base, "models", "claudette.tflite")

    if args.backend == "porcupine":
        if not args.access_key:
            print('{"error": "PORCUPINE_ACCESS_KEY not set"}', flush=True)
            sys.exit(1)
        run_porcupine(args.model, args.access_key, args.threshold)
    else:
        run_oww(args.model, args.threshold)


if __name__ == "__main__":
    main()
