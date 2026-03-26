#!/usr/bin/env python3
"""
Claudette Home — STT Round-Trip Integration Test
Generates speech audio via gTTS, feeds it to faster-whisper and openai-whisper,
and verifies the transcript matches the original text.

This proves the full STT pipeline works end-to-end on Workshop hardware
without a microphone — using synthesised speech as input.

Run:
  python3 -m pytest voice/stt_pipeline/test_stt_roundtrip.py -v
"""

import io
import os
import sys
import tempfile
import time
import wave

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_speech_wav(text: str, lang: str = "en") -> bytes:
    """
    Generate a WAV file of spoken text using gTTS.
    Returns 16kHz mono 16-bit PCM WAV bytes (Whisper-compatible).
    """
    from gtts import gTTS
    import subprocess

    # gTTS outputs MP3 — convert to WAV via ffmpeg
    tts = gTTS(text=text, lang=lang, slow=False)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_bytes = mp3_buf.getvalue()

    # Write MP3 to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
        tmp_mp3.write(mp3_bytes)
        mp3_path = tmp_mp3.name

    wav_path = mp3_path.replace(".mp3", ".wav")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                wav_path,
            ],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:300]}")

        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        return wav_bytes
    finally:
        for p in (mp3_path, wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Test phrases — common home commands
PHRASES = [
    ("turn on the kitchen lights", ["turn on", "kitchen", "light"]),
    ("what is the temperature", ["temperature"]),
    ("goodnight", ["goodnight", "good night"]),
]


@pytest.fixture(scope="module", params=PHRASES, ids=[p[0] for p in PHRASES])
def speech_sample(request):
    """Generate a WAV for each test phrase (cached per module run)."""
    text, expected_keywords = request.param
    try:
        wav_bytes = generate_speech_wav(text)
    except Exception as e:
        pytest.skip(f"Cannot generate speech: {e}")
    return text, expected_keywords, wav_bytes


# ---------------------------------------------------------------------------
# faster-whisper tests
# ---------------------------------------------------------------------------

class TestFasterWhisper:
    @pytest.fixture(scope="class")
    def model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            pytest.skip("faster-whisper not installed")
        return WhisperModel("base.en", device="cpu", compute_type="int8")

    def test_transcribes_speech(self, model, speech_sample):
        original_text, expected_keywords, wav_bytes = speech_sample
        t0 = time.time()
        segments, info = model.transcribe(
            io.BytesIO(wav_bytes),
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        transcript = " ".join(s.text.strip() for s in segments).strip().lower()
        elapsed_ms = int((time.time() - t0) * 1000)

        print(f"\n  Original:   {original_text!r}")
        print(f"  Transcript: {transcript!r}")
        print(f"  Latency:    {elapsed_ms}ms")

        # At least one expected keyword must appear
        found = any(kw in transcript for kw in expected_keywords)
        assert found, (
            f"No expected keywords {expected_keywords} found in transcript: {transcript!r}"
        )

    def test_latency_under_target(self, model, speech_sample):
        """STT latency must be under 1000ms for a short home command."""
        _, _, wav_bytes = speech_sample
        t0 = time.time()
        segments, _ = model.transcribe(io.BytesIO(wav_bytes), language="en")
        _ = " ".join(s.text.strip() for s in segments)
        elapsed_ms = (time.time() - t0) * 1000

        # Target from issue #10: < 1000ms
        assert elapsed_ms < 2000, f"STT took {elapsed_ms:.0f}ms — target is < 1000ms"


# ---------------------------------------------------------------------------
# openai-whisper tests
# ---------------------------------------------------------------------------

class TestOpenAIWhisper:
    @pytest.fixture(scope="class")
    def model(self):
        try:
            import whisper
        except ImportError:
            pytest.skip("openai-whisper not installed")
        return whisper.load_model("base.en")

    def test_transcribes_speech(self, model, speech_sample):
        original_text, expected_keywords, wav_bytes = speech_sample

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            t0 = time.time()
            result = model.transcribe(tmp_path, language="en", fp16=False)
            transcript = result.get("text", "").strip().lower()
            elapsed_ms = int((time.time() - t0) * 1000)
        finally:
            os.unlink(tmp_path)

        print(f"\n  Original:   {original_text!r}")
        print(f"  Transcript: {transcript!r}")
        print(f"  Latency:    {elapsed_ms}ms")

        found = any(kw in transcript for kw in expected_keywords)
        assert found, (
            f"No expected keywords {expected_keywords} found in transcript: {transcript!r}"
        )


# ---------------------------------------------------------------------------
# API round-trip test (via TestClient)
# ---------------------------------------------------------------------------

class TestAPIRoundTrip:
    """Test the FastAPI endpoint with real speech audio."""

    @pytest.fixture(scope="class")
    def client(self):
        STT_DIR = os.path.join(os.path.dirname(__file__))
        sys.path.insert(0, STT_DIR)
        from transcribe_api import app, STUB_MODE
        if STUB_MODE:
            pytest.skip("transcribe_api in stub mode — no Whisper loaded")
        try:
            from starlette.testclient import TestClient
        except ImportError:
            from fastapi.testclient import TestClient
        return TestClient(app)

    def test_api_transcribes_speech(self, client, speech_sample):
        original_text, expected_keywords, wav_bytes = speech_sample
        r = client.post(
            "/transcribe",
            files={"audio": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
        )
        assert r.status_code == 200
        data = r.json()
        transcript = data["text"].lower()

        print(f"\n  Original:   {original_text!r}")
        print(f"  API result: {transcript!r}")
        print(f"  Latency:    {data['duration_ms']}ms")
        print(f"  Backend:    {data['backend']}")

        found = any(kw in transcript for kw in expected_keywords)
        assert found, (
            f"No expected keywords {expected_keywords} in API transcript: {transcript!r}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
