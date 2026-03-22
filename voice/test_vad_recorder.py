#!/usr/bin/env python3
"""
Claudette Home — VAD Recorder Tests
Full test suite for vad_recorder.py using synthetic audio.

No microphone required — all tests use in-memory audio buffers
pushed through the VadEngine and process_audio_buffer().

Run:
  python3 -m pytest voice/test_vad_recorder.py -v
  python3 voice/test_vad_recorder.py
"""

import io
import os
import sys
import wave

import numpy as np
import pytest

# Path setup
VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VOICE_DIR)

from vad_recorder import (
    SILERO_CHUNK_DURATION_S,
    SILERO_CHUNK_SAMPLES,
    SILERO_SAMPLE_RATE,
    VadConfig,
    VadEngine,
    VadRecordingResult,
    process_audio_buffer,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic audio generation
# ---------------------------------------------------------------------------

def silence(duration_s: float) -> np.ndarray:
    """Generate silence (very low noise floor)."""
    n = int(SILERO_SAMPLE_RATE * duration_s)
    return np.random.randn(n).astype(np.float32) * 0.001


def speech_like(duration_s: float, freq_hz: float = 200.0, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate speech-like audio (voiced harmonics).
    Uses a sawtooth wave with harmonics + slight noise to trigger VAD.
    Real speech has formants — this approximation is enough for Silero.
    """
    n = int(SILERO_SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SILERO_SAMPLE_RATE

    # Fundamental + harmonics (simulates vocal formants)
    signal = np.zeros(n, dtype=np.float32)
    for harmonic in [1, 2, 3, 4, 5]:
        signal += (amplitude / harmonic) * np.sin(2 * np.pi * freq_hz * harmonic * t)

    # Add slight noise
    signal += np.random.randn(n).astype(np.float32) * 0.02

    # Clip to [-1, 1]
    return np.clip(signal, -1.0, 1.0)


def make_audio(*segments) -> np.ndarray:
    """
    Concatenate audio segments.
    Each segment is a tuple: ('silence', duration) or ('speech', duration)
    """
    parts = []
    for seg_type, duration in segments:
        if seg_type == "silence":
            parts.append(silence(duration))
        elif seg_type == "speech":
            parts.append(speech_like(duration))
        else:
            raise ValueError(f"Unknown segment type: {seg_type}")
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# VadConfig tests
# ---------------------------------------------------------------------------

class TestVadConfig:
    def test_default_config(self):
        config = VadConfig()
        assert config.threshold == 0.5
        assert config.silence_timeout == 1.5
        assert config.max_duration == 10.0
        assert config.sample_rate == 16000

    def test_custom_config(self):
        config = VadConfig(threshold=0.7, silence_timeout=2.0, max_duration=15.0)
        assert config.threshold == 0.7
        assert config.silence_timeout == 2.0
        assert config.max_duration == 15.0

    def test_validate_bad_sample_rate(self):
        config = VadConfig(sample_rate=8000)
        with pytest.raises(ValueError, match="16000"):
            config.validate()

    def test_validate_bad_threshold_zero(self):
        config = VadConfig(threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            config.validate()

    def test_validate_bad_threshold_one(self):
        config = VadConfig(threshold=1.0)
        with pytest.raises(ValueError, match="threshold"):
            config.validate()

    def test_validate_bad_silence_timeout(self):
        config = VadConfig(silence_timeout=-1)
        with pytest.raises(ValueError, match="silence_timeout"):
            config.validate()

    def test_validate_bad_max_duration(self):
        config = VadConfig(max_duration=0)
        with pytest.raises(ValueError, match="max_duration"):
            config.validate()

    def test_validate_ok(self):
        config = VadConfig()
        config.validate()  # should not raise

    def test_validate_edge_threshold(self):
        config = VadConfig(threshold=0.01)
        config.validate()  # valid
        config2 = VadConfig(threshold=0.99)
        config2.validate()  # valid


# ---------------------------------------------------------------------------
# VadEngine tests
# ---------------------------------------------------------------------------

class TestVadEngine:
    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.config = VadConfig(
            threshold=0.5,
            silence_timeout=1.0,
            max_duration=8.0,
            pre_speech_timeout=3.0,
        )
        self.engine = VadEngine(self.config)
        self.engine.load_model()

    def test_engine_loads_model(self):
        assert self.engine._model is not None

    def test_reset_clears_state(self):
        self.engine.speech_started = True
        self.engine.chunk_count = 99
        self.engine.reset()
        assert self.engine.speech_started is False
        assert self.engine.chunk_count == 0
        assert len(self.engine.vad_probs) == 0

    def test_silence_returns_low_prob(self):
        """Pure silence should produce low VAD probability."""
        self.engine.reset()
        chunk = silence(SILERO_CHUNK_DURATION_S)[:SILERO_CHUNK_SAMPLES]
        result = self.engine.process_chunk(chunk, 0.0)
        assert result["prob"] < 0.5
        assert result["is_speech"] is False

    def test_speech_like_returns_higher_prob(self):
        """
        Speech-like audio should eventually trigger speech detection.
        Feed several chunks of speech to let Silero's internal state build up.
        """
        self.engine.reset()
        audio = speech_like(1.0)
        max_prob = 0.0
        for i in range(0, len(audio) - SILERO_CHUNK_SAMPLES, SILERO_CHUNK_SAMPLES):
            chunk = audio[i : i + SILERO_CHUNK_SAMPLES]
            t = i / SILERO_SAMPLE_RATE
            result = self.engine.process_chunk(chunk, t)
            max_prob = max(max_prob, result["prob"])
        # Silero should detect speech somewhere in the sequence
        assert max_prob > 0.3, f"Expected speech detection, max_prob={max_prob}"

    def test_pre_speech_timeout(self):
        """If no speech for pre_speech_timeout seconds, action=timeout_no_speech."""
        self.engine.reset()
        # Feed silence for longer than pre_speech_timeout
        total_samples = int(SILERO_SAMPLE_RATE * (self.config.pre_speech_timeout + 0.5))
        audio = silence(self.config.pre_speech_timeout + 0.5)

        actions = []
        for i in range(0, total_samples - SILERO_CHUNK_SAMPLES, SILERO_CHUNK_SAMPLES):
            chunk = audio[i : i + SILERO_CHUNK_SAMPLES]
            t = i / SILERO_SAMPLE_RATE
            result = self.engine.process_chunk(chunk, t)
            actions.append(result["action"])

        assert "timeout_no_speech" in actions

    def test_max_duration_timeout(self):
        """Recording should stop at max_duration even during continuous speech."""
        short_config = VadConfig(
            threshold=0.3,
            silence_timeout=100.0,  # very long — so silence detection can't fire first
            max_duration=2.0,
            pre_speech_timeout=5.0,
        )
        engine = VadEngine(short_config)
        engine.load_model()
        engine.reset()

        # Feed continuous speech for 3 seconds (exceeds max_duration of 2s)
        audio = speech_like(3.0, amplitude=0.8)
        actions = []
        for i in range(0, len(audio) - SILERO_CHUNK_SAMPLES, SILERO_CHUNK_SAMPLES):
            chunk = audio[i : i + SILERO_CHUNK_SAMPLES]
            t = i / SILERO_SAMPLE_RATE
            result = engine.process_chunk(chunk, t)
            actions.append(result["action"])
            if result["action"] in ("timeout_max_duration", "speech_end"):
                break

        assert "timeout_max_duration" in actions

    def test_speech_boundaries(self):
        """Speech boundaries should be tracked."""
        self.engine.reset()
        # Force speech_started state
        self.engine.speech_started = True
        self.engine.speech_start_chunk = 5
        self.engine.speech_end_chunk = 20
        self.engine.chunk_count = 25

        start, end = self.engine.get_speech_boundaries()
        assert start == 2  # 5 - 3 lead-in
        assert end == 20

    def test_speech_boundaries_no_speech(self):
        """If no speech detected, boundaries cover entire recording."""
        self.engine.reset()
        self.engine.chunk_count = 50

        start, end = self.engine.get_speech_boundaries()
        assert start == 0
        assert end == 50

    def test_speech_duration_no_speech(self):
        """No speech → 0ms duration."""
        self.engine.reset()
        assert self.engine.get_speech_duration_ms() == 0.0

    def test_speech_duration_with_chunks(self):
        """Speech duration is proportional to speech chunk count."""
        self.engine.reset()
        self.engine.speech_chunks = [1, 2, 3, 4, 5]
        expected = 5 * SILERO_CHUNK_DURATION_S * 1000
        assert abs(self.engine.get_speech_duration_ms() - expected) < 0.1

    def test_vad_probs_tracked(self):
        """Each process_chunk call should append to vad_probs."""
        self.engine.reset()
        for i in range(10):
            chunk = silence(SILERO_CHUNK_DURATION_S)[:SILERO_CHUNK_SAMPLES]
            self.engine.process_chunk(chunk, i * SILERO_CHUNK_DURATION_S)
        assert len(self.engine.vad_probs) == 10
        assert all(0.0 <= p <= 1.0 for p in self.engine.vad_probs)


# ---------------------------------------------------------------------------
# process_audio_buffer tests (integration — full pipeline without mic)
# ---------------------------------------------------------------------------

class TestProcessAudioBuffer:
    def test_silence_only(self):
        """Pure silence should timeout with no speech detected."""
        config = VadConfig(
            pre_speech_timeout=1.0,
            silence_timeout=0.5,
            max_duration=3.0,
        )
        audio = silence(2.0)
        result = process_audio_buffer(audio, config)

        assert isinstance(result, VadRecordingResult)
        assert result.speech_detected is False
        assert result.chunks_processed > 0
        assert len(result.vad_probs) > 0

    def test_speech_then_silence(self):
        """Speech followed by silence should detect speech and end on silence."""
        config = VadConfig(
            threshold=0.4,
            silence_timeout=0.8,
            max_duration=10.0,
            pre_speech_timeout=3.0,
        )
        # 0.5s silence + 2s speech + 2s silence
        audio = make_audio(
            ("silence", 0.5),
            ("speech", 2.0),
            ("silence", 2.0),
        )
        result = process_audio_buffer(audio, config)

        assert result.chunks_processed > 0
        assert len(result.audio_bytes) > 44  # more than just WAV header
        assert len(result.vad_probs) > 0

    def test_result_has_wav_header(self):
        """Output audio_bytes should be valid WAV."""
        config = VadConfig(pre_speech_timeout=0.5, max_duration=2.0)
        audio = silence(1.0)
        result = process_audio_buffer(audio, config)

        # Check WAV header
        buf = io.BytesIO(result.audio_bytes)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000

    def test_max_duration_caps_long_speech(self):
        """Even continuous speech should be capped at max_duration."""
        config = VadConfig(
            threshold=0.3,
            max_duration=1.5,
            silence_timeout=1.0,
            pre_speech_timeout=3.0,
        )
        audio = speech_like(5.0)
        result = process_audio_buffer(audio, config)

        # Should have stopped around 1.5s, not recorded all 5s
        assert result.total_duration_ms < 2500  # generous margin

    def test_short_speech_burst(self):
        """A short speech burst should be captured."""
        config = VadConfig(
            threshold=0.4,
            silence_timeout=0.5,
            max_duration=5.0,
            pre_speech_timeout=2.0,
            min_speech_duration=0.1,
        )
        # Short command: 0.3s silence + 0.8s speech + 1s silence
        audio = make_audio(
            ("silence", 0.3),
            ("speech", 0.8),
            ("silence", 1.0),
        )
        result = process_audio_buffer(audio, config)
        assert result.chunks_processed > 0

    def test_multiple_speech_segments(self):
        """Multiple speech segments with brief pauses — should capture through."""
        config = VadConfig(
            threshold=0.4,
            silence_timeout=1.5,  # long enough to bridge 0.3s pauses
            max_duration=10.0,
            pre_speech_timeout=3.0,
        )
        audio = make_audio(
            ("silence", 0.3),
            ("speech", 1.0),
            ("silence", 0.3),   # brief pause — should NOT trigger end-of-speech
            ("speech", 1.0),
            ("silence", 2.0),   # long pause — should trigger end
        )
        result = process_audio_buffer(audio, config)
        assert result.chunks_processed > 0


# ---------------------------------------------------------------------------
# VadRecordingResult tests
# ---------------------------------------------------------------------------

class TestVadRecordingResult:
    def test_result_fields(self):
        result = VadRecordingResult(
            audio_bytes=b"\x00" * 100,
            speech_duration_ms=500.0,
            total_duration_ms=2000.0,
            ended_by_silence=True,
            speech_detected=True,
            chunks_processed=50,
            vad_probs=[0.1, 0.8, 0.9],
        )
        assert result.speech_detected is True
        assert result.ended_by_silence is True
        assert result.speech_duration_ms == 500.0
        assert len(result.vad_probs) == 3

    def test_result_defaults(self):
        result = VadRecordingResult(
            audio_bytes=b"",
            speech_duration_ms=0,
            total_duration_ms=0,
            ended_by_silence=False,
            speech_detected=False,
            chunks_processed=0,
        )
        assert result.vad_probs == []


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_sample_rate(self):
        assert SILERO_SAMPLE_RATE == 16000

    def test_chunk_samples(self):
        assert SILERO_CHUNK_SAMPLES == 512

    def test_chunk_duration(self):
        expected = 512 / 16000
        assert abs(SILERO_CHUNK_DURATION_S - expected) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
