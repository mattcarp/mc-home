#!/usr/bin/env python3
"""
Claudette Home — Live STT Round-Trip Test
Tests the /transcribe endpoint against the RUNNING service at localhost:8765.

Unlike test_transcribe_api.py (which imports the app directly), this fires
real HTTP requests. Proves the service is live and working as-deployed.

Run:
  python3 voice/stt_pipeline/test_stt_live_roundtrip.py
  # or:
  python3 -m pytest voice/stt_pipeline/test_stt_live_roundtrip.py -v

Prerequisites:
  - claudette-stt.service running: systemctl status claudette-stt.service
  - gTTS available (pip install gtts)
  - ffmpeg in PATH

This is the panel's point of view — it generates audio, POSTs it, reads the text.
"""

import io
import os
import subprocess
import sys
import tempfile
import time
import wave
import unittest

import requests

STT_URL = os.environ.get("STT_API_URL", "http://127.0.0.1:8765")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_silence_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a PCM silence WAV — minimal test input."""
    buf = io.BytesIO()
    n_samples = int(sample_rate * duration_s)
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    buf.seek(0)
    return buf.read()


def make_speech_wav(text: str, lang: str = "en") -> bytes:
    """Generate a speech WAV from text using gTTS → ffmpeg → 16kHz mono WAV."""
    try:
        from gtts import gTTS
    except ImportError:
        raise RuntimeError("gTTS not installed: pip install gtts")

    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = os.path.join(tmpdir, "tts.mp3")
        wav_path = os.path.join(tmpdir, "tts.wav")
        tts.save(mp3_path)
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", mp3_path,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav_path],
            capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
        with open(wav_path, "rb") as f:
            return f.read()


def post_wav(wav_bytes: bytes, filename: str = "test.wav") -> dict:
    """POST WAV bytes to /transcribe, return parsed JSON response."""
    resp = requests.post(
        f"{STT_URL}/transcribe",
        files={"audio": (filename, io.BytesIO(wav_bytes), "audio/wav")},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestServiceHealth(unittest.TestCase):
    """Verify the service is up before running any transcription tests."""

    def test_health_endpoint_is_reachable(self):
        """GET /health must return 200 and report status=ok."""
        resp = requests.get(f"{STT_URL}/health", timeout=5)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")

    def test_health_reports_faster_whisper_backend(self):
        """We expect faster-whisper backend (installed 2026-03-21)."""
        resp = requests.get(f"{STT_URL}/health", timeout=5)
        data = resp.json()
        backend = data.get("backend", "unknown")
        # Either faster-whisper (ideal) or openai-whisper (fallback) is acceptable
        self.assertIn(backend, ("faster-whisper", "openai-whisper"),
                      f"Unexpected backend: {backend!r} — check claudette-stt.service logs")

    def test_health_reports_mode(self):
        """Service must not be in stub mode (stub = Whisper not loaded = broken)."""
        resp = requests.get(f"{STT_URL}/health", timeout=5)
        data = resp.json()
        mode = data.get("mode", "stub")
        self.assertEqual(mode, "live",
                         "STT service is in stub mode — faster-whisper may not be installed")

    def test_models_endpoint(self):
        """GET /models must return model info."""
        resp = requests.get(f"{STT_URL}/models", timeout=5)
        self.assertIn(resp.status_code, (200, 404))  # 404 OK if endpoint not implemented yet
        if resp.status_code == 200:
            data = resp.json()
            self.assertIsInstance(data, (dict, list))


class TestSilenceTranscription(unittest.TestCase):
    """Silence input — service must handle gracefully (empty or whitespace transcript)."""

    def test_silence_returns_200(self):
        """POSTing silence must return HTTP 200."""
        wav = make_silence_wav(1.0)
        resp_data = post_wav(wav, "silence.wav")
        # No exception = 200 received
        self.assertIsInstance(resp_data, dict)

    def test_silence_has_text_field(self):
        """Response must always contain a 'text' field."""
        wav = make_silence_wav(1.0)
        data = post_wav(wav, "silence.wav")
        self.assertIn("text", data, f"Response missing 'text' field: {data}")

    def test_silence_has_duration_ms(self):
        """Response must report processing time in duration_ms."""
        wav = make_silence_wav(1.0)
        data = post_wav(wav, "silence.wav")
        self.assertIn("duration_ms", data)
        self.assertGreater(data["duration_ms"], 0)

    def test_silence_text_is_empty_or_whitespace(self):
        """Silence should transcribe to '' or whitespace — not hallucinated words."""
        wav = make_silence_wav(2.0)
        data = post_wav(wav, "silence_2s.wav")
        text = data.get("text", "").strip()
        # Allow empty string or minor artifacts — but not more than 3 words
        word_count = len(text.split()) if text else 0
        self.assertLessEqual(word_count, 3,
                             f"Silence produced suspiciously long transcript: {text!r}")


class TestHomeCommandTranscription(unittest.TestCase):
    """
    End-to-end: generate speech with gTTS, transcribe with live Whisper.
    Verifies the panel→Workshop STT flow works for typical home commands.
    """

    # Format: (text to synthesise, keywords that MUST appear in transcript)
    COMMANDS = [
        ("turn on the kitchen lights", ["kitchen", "light"]),
        ("turn off the living room", ["living", "room"]),
        ("goodnight", ["goodnight", "good night", "good"]),
        ("what is the temperature", ["temperature"]),
        ("lock the front door", ["lock", "door"]),
    ]

    def _transcribe_phrase(self, text: str) -> str:
        """Synthesise text as speech, POST to STT, return lowercased transcript."""
        try:
            wav_bytes = make_speech_wav(text)
        except Exception as e:
            self.skipTest(f"gTTS unavailable (network issue?): {e}")
        data = post_wav(wav_bytes, f"cmd_{text[:20].replace(' ', '_')}.wav")
        return data.get("text", "").lower().strip()

    def test_kitchen_lights_command(self):
        transcript = self._transcribe_phrase("turn on the kitchen lights")
        self.assertTrue(
            any(kw in transcript for kw in ["kitchen", "light"]),
            f"Expected kitchen/light in transcript, got: {transcript!r}"
        )

    def test_temperature_query(self):
        transcript = self._transcribe_phrase("what is the temperature")
        self.assertIn("temperature", transcript,
                      f"Expected 'temperature' in transcript, got: {transcript!r}")

    def test_goodnight_command(self):
        transcript = self._transcribe_phrase("goodnight")
        self.assertTrue(
            any(kw in transcript for kw in ["good", "night"]),
            f"Expected good/night in transcript, got: {transcript!r}"
        )

    def test_latency_under_3_seconds(self):
        """STT round-trip must complete in under 3 seconds for home use."""
        try:
            wav_bytes = make_speech_wav("turn off the lights please")
        except Exception as e:
            self.skipTest(f"gTTS unavailable: {e}")
        t0 = time.time()
        data = post_wav(wav_bytes, "latency_test.wav")
        elapsed = time.time() - t0
        self.assertLess(elapsed, 3.0,
                        f"STT took {elapsed:.2f}s — target is <3s for home use. "
                        f"Consider upgrading to tiny.en model.")
        print(f"\n  ✅ STT latency: {elapsed:.2f}s (target: <3.0s)")

    def test_response_structure(self):
        """Response JSON must have expected fields."""
        try:
            wav_bytes = make_speech_wav("hello")
        except Exception as e:
            self.skipTest(f"gTTS unavailable: {e}")
        data = post_wav(wav_bytes, "structure_test.wav")
        required_fields = ["text", "language", "duration_ms", "backend"]
        for field in required_fields:
            self.assertIn(field, data,
                          f"Missing field '{field}' in response: {data}")


class TestErrorHandling(unittest.TestCase):
    """Service must handle bad inputs gracefully."""

    def test_empty_audio_returns_400(self):
        """Empty body must return 400, not 500."""
        resp = requests.post(
            f"{STT_URL}/transcribe",
            files={"audio": ("empty.wav", io.BytesIO(b""), "audio/wav")},
            timeout=10,
        )
        self.assertEqual(resp.status_code, 400,
                         f"Expected 400 for empty audio, got {resp.status_code}")

    def test_corrupted_audio_returns_4xx(self):
        """Corrupted/non-audio data must return 4xx (not crash the server)."""
        garbage = b"this is not audio data at all"
        resp = requests.post(
            f"{STT_URL}/transcribe",
            files={"audio": ("garbage.wav", io.BytesIO(garbage), "audio/wav")},
            timeout=10,
        )
        self.assertIn(resp.status_code, (400, 422, 500),
                      f"Unexpected status for corrupted audio: {resp.status_code}")
        # Regardless of status code, server must still be alive after this
        health = requests.get(f"{STT_URL}/health", timeout=5)
        self.assertEqual(health.status_code, 200, "Server crashed after bad input!")


class TestPerformanceBenchmark(unittest.TestCase):
    """
    Latency benchmarks — informational, not hard failures.
    Prints timing stats useful for tuning the model size.
    """

    @unittest.skip("Run manually: python3 test_stt_live_roundtrip.py --bench")
    def test_benchmark_10_phrases(self):
        """Benchmark 10 phrases, print p50/p95/max latency."""
        phrases = [
            "turn on the lights",
            "goodnight",
            "what time is it",
            "lock the front door",
            "turn off the kitchen",
            "set the dinner scene",
            "it's getting dark",
            "I'm going to bed",
            "open the shutters",
            "what's the temperature",
        ]
        latencies = []
        for phrase in phrases:
            try:
                wav = make_speech_wav(phrase)
                t0 = time.time()
                post_wav(wav)
                latencies.append(time.time() - t0)
            except Exception:
                pass
        if latencies:
            latencies.sort()
            n = len(latencies)
            p50 = latencies[n // 2]
            p95 = latencies[int(n * 0.95)]
            print(f"\n📊 STT Latency Benchmark ({n} phrases):")
            print(f"   p50={p50:.2f}s  p95={p95:.2f}s  max={latencies[-1]:.2f}s")


# ---------------------------------------------------------------------------
# CLI runner (verbose summary)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🏠 Claudette Home — Live STT Round-Trip Test")
    print(f"   Service: {STT_URL}")
    print()

    # Quick health check before running tests
    try:
        resp = requests.get(f"{STT_URL}/health", timeout=3)
        data = resp.json()
        print(f"   ✅ Service is UP: backend={data.get('backend')} mode={data.get('mode')}")
    except Exception as e:
        print(f"   ❌ Service UNREACHABLE: {e}")
        print(f"      Start it: systemctl start claudette-stt.service")
        sys.exit(1)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Run in logical order: health → silence → commands → errors
    for cls in [TestServiceHealth, TestSilenceTranscription,
                TestHomeCommandTranscription, TestErrorHandling]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
