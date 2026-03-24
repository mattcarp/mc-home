#!/usr/bin/env python3
"""
Claudette Home — STT Backend Benchmark
Compare faster-whisper vs openai-whisper latency on Workshop hardware.

Usage:
  python3 voice/stt_pipeline/benchmark_backends.py

Reports model load time, inference time on silence and on synthetic speech,
across available backends and model sizes.
"""

import io
import struct
import time
import wave
import json
import sys
from typing import Optional


def make_wav(duration_s: float = 2.0, sample_rate: int = 16000) -> io.BytesIO:
    """Create a silent WAV in memory."""
    n_samples = int(duration_s * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    buf.seek(0)
    return buf


def benchmark_faster_whisper(model_size: str, wav: io.BytesIO) -> Optional[dict]:
    """Benchmark faster-whisper backend."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    load_time = time.time() - t0

    # Warm-up
    wav.seek(0)
    model.transcribe(wav, language="en")

    # Timed run
    wav.seek(0)
    t0 = time.time()
    segments, info = model.transcribe(wav, language="en")
    text = " ".join(s.text.strip() for s in segments)
    infer_time = time.time() - t0

    return {
        "backend": "faster-whisper",
        "model": model_size,
        "load_s": round(load_time, 2),
        "infer_s": round(infer_time, 3),
        "text": text,
    }


def benchmark_openai_whisper(model_size: str, wav: io.BytesIO) -> Optional[dict]:
    """Benchmark openai-whisper backend."""
    try:
        import whisper
    except ImportError:
        return None

    import tempfile
    import os

    # openai-whisper needs a file path
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.seek(0)
    tmp.write(wav.read())
    tmp.close()

    # Map model name
    oai_map = {"base.en": "base.en", "small.en": "small.en", "tiny.en": "tiny.en"}
    oai_name = oai_map.get(model_size, model_size)

    try:
        t0 = time.time()
        model = whisper.load_model(oai_name)
        load_time = time.time() - t0

        # Warm-up
        model.transcribe(tmp.name, language="en")

        # Timed run
        t0 = time.time()
        result = model.transcribe(tmp.name, language="en")
        infer_time = time.time() - t0

        return {
            "backend": "openai-whisper",
            "model": oai_name,
            "load_s": round(load_time, 2),
            "infer_s": round(infer_time, 3),
            "text": result.get("text", "").strip(),
        }
    finally:
        os.unlink(tmp.name)


def main():
    print("=" * 60)
    print("Claudette Home — STT Backend Benchmark")
    print("=" * 60)

    wav = make_wav(duration_s=2.0)
    model_size = "base.en"
    results = []

    print(f"\nModel size: {model_size} | Audio: 2s silence | CPU only")
    print("-" * 60)

    # faster-whisper
    print("Testing faster-whisper...", flush=True)
    r = benchmark_faster_whisper(model_size, wav)
    if r:
        results.append(r)
        print(f"  Load: {r['load_s']}s | Infer: {r['infer_s']}s | Text: \"{r['text']}\"")
    else:
        print("  NOT INSTALLED")

    # openai-whisper
    print("Testing openai-whisper...", flush=True)
    r = benchmark_openai_whisper(model_size, wav)
    if r:
        results.append(r)
        print(f"  Load: {r['load_s']}s | Infer: {r['infer_s']}s | Text: \"{r['text']}\"")
    else:
        print("  NOT INSTALLED")

    print("-" * 60)
    if len(results) == 2:
        fw = next(r for r in results if r["backend"] == "faster-whisper")
        ow = next(r for r in results if r["backend"] == "openai-whisper")
        speedup = ow["infer_s"] / fw["infer_s"] if fw["infer_s"] > 0 else 0
        print(f"faster-whisper is {speedup:.1f}x faster at inference")
        print(f"Load: {fw['load_s']}s vs {ow['load_s']}s")

    print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
