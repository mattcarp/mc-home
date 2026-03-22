#!/usr/bin/env python3
"""
Claudette Home — VAD-based Smart Audio Recorder
Uses Silero VAD to detect speech boundaries after wake word detection.

Instead of recording a fixed 5 seconds, this module:
  1. Starts recording immediately after wake word
  2. Detects when the user starts speaking (speech onset)
  3. Detects when the user stops speaking (end-of-speech via silence timeout)
  4. Returns only the speech audio (trimmed, WAV format)

This replaces the fixed-time record_audio() in pipeline.py for the live
voice pipeline. Fixed recording remains as a fallback.

Architecture:
  wake_word_detected → vad_recorder.record_until_silence()
    → streams mic chunks through Silero VAD
    → detects speech_start / speech_end
    → returns WAV bytes of just the spoken command

Performance:
  Silero VAD processes 512-sample chunks (32ms @ 16kHz) on CPU in ~1ms.
  Total overhead: negligible. The bottleneck is always STT, not VAD.

Usage:
  from vad_recorder import create_vad_recorder, VadConfig

  config = VadConfig(silence_timeout=1.5, max_duration=10.0)
  recorder = create_vad_recorder(config)
  result = recorder.record()
  # result.audio_bytes = WAV, result.speech_duration_ms, result.total_duration_ms

Environment:
  VAD_THRESHOLD         — speech probability threshold (default: 0.5)
  VAD_SILENCE_TIMEOUT   — seconds of silence to end recording (default: 1.5)
  VAD_MAX_DURATION      — max recording seconds (default: 10.0)
  VAD_PRE_SPEECH_TIMEOUT — max seconds to wait for speech start (default: 5.0)
  VAD_SAMPLE_RATE       — sample rate (default: 16000, Silero requires 16000)
"""

import io
import logging
import os
import struct
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Silero VAD requires 16000Hz sample rate and specific chunk sizes
SILERO_SAMPLE_RATE = 16000
# Silero accepts 512 samples (32ms) at 16kHz
SILERO_CHUNK_SAMPLES = 512
SILERO_CHUNK_DURATION_S = SILERO_CHUNK_SAMPLES / SILERO_SAMPLE_RATE  # 0.032s


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VadConfig:
    """Configuration for VAD-based recording."""

    # Speech detection threshold (0.0–1.0). Higher = stricter.
    threshold: float = float(os.environ.get("VAD_THRESHOLD", "0.5"))

    # Seconds of continuous silence after speech to trigger end-of-speech.
    silence_timeout: float = float(os.environ.get("VAD_SILENCE_TIMEOUT", "1.5"))

    # Maximum recording duration (safety cap).
    max_duration: float = float(os.environ.get("VAD_MAX_DURATION", "10.0"))

    # Max seconds to wait for speech to begin after wake word.
    pre_speech_timeout: float = float(os.environ.get("VAD_PRE_SPEECH_TIMEOUT", "5.0"))

    # Sample rate — must be 16000 for Silero.
    sample_rate: int = SILERO_SAMPLE_RATE

    # Minimum speech duration to accept (filters out clicks/pops).
    min_speech_duration: float = 0.3

    def validate(self):
        if self.sample_rate != 16000:
            raise ValueError("Silero VAD requires sample_rate=16000")
        if not (0.0 < self.threshold < 1.0):
            raise ValueError(f"threshold must be 0.0–1.0, got {self.threshold}")
        if self.silence_timeout <= 0:
            raise ValueError("silence_timeout must be > 0")
        if self.max_duration <= 0:
            raise ValueError("max_duration must be > 0")
        if self.pre_speech_timeout <= 0:
            raise ValueError("pre_speech_timeout must be > 0")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class VadRecordingResult:
    """Result from a VAD-based recording session."""

    # WAV audio bytes (16kHz, mono, 16-bit PCM)
    audio_bytes: bytes

    # Duration of detected speech in milliseconds
    speech_duration_ms: float

    # Total recording duration in milliseconds (includes silence)
    total_duration_ms: float

    # Whether recording ended due to silence (vs max_duration)
    ended_by_silence: bool

    # Whether speech was detected at all
    speech_detected: bool

    # Number of audio chunks processed
    chunks_processed: int

    # VAD probabilities per chunk (for debugging)
    vad_probs: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core VAD engine (no hardware dependency — works on raw chunks)
# ---------------------------------------------------------------------------

class VadEngine:
    """
    Wraps Silero VAD for streaming speech boundary detection.

    This is the core logic, separated from audio I/O so it can be
    tested without a microphone.
    """

    def __init__(self, config: Optional[VadConfig] = None):
        self.config = config or VadConfig()
        self.config.validate()
        self._model = None
        self._reset_state()

    def _reset_state(self):
        """Reset all tracking state for a new recording session."""
        self.speech_started = False
        self.speech_ended = False
        self.speech_start_chunk = -1
        self.speech_end_chunk = -1
        self.silence_start_time: Optional[float] = None
        self.first_chunk_time: Optional[float] = None
        self.chunk_count = 0
        self.vad_probs: List[float] = []
        self.speech_chunks: List[int] = []  # indices of chunks with speech

    def load_model(self):
        """Load Silero VAD model (lazy — called once, cached)."""
        if self._model is not None:
            return
        try:
            import torch
            self._model, _ = torch.hub.load(
                "snakers4/silero-vad", "silero_vad", trust_repo=True
            )
            logger.info("Silero VAD model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD: {e}")

    def reset(self):
        """Reset for a new recording session."""
        self._reset_state()
        # Reset Silero internal state
        if self._model is not None:
            self._model.reset_states()

    def process_chunk(self, audio_chunk: np.ndarray, current_time: float) -> dict:
        """
        Process a single audio chunk through VAD.

        Args:
            audio_chunk: numpy float32 array, length=SILERO_CHUNK_SAMPLES
            current_time: monotonic time of this chunk (for timeout tracking)

        Returns:
            dict with:
              - prob: float (speech probability 0.0–1.0)
              - is_speech: bool
              - action: str ('continue' | 'speech_start' | 'speech_end' |
                             'timeout_no_speech' | 'timeout_max_duration')
        """
        import torch

        if self._model is None:
            raise RuntimeError("Call load_model() before process_chunk()")

        if self.first_chunk_time is None:
            self.first_chunk_time = current_time

        self.chunk_count += 1
        elapsed = current_time - self.first_chunk_time

        # Run Silero VAD
        tensor = torch.from_numpy(audio_chunk).float()
        prob = float(self._model(tensor, SILERO_SAMPLE_RATE))
        self.vad_probs.append(prob)

        is_speech = prob >= self.config.threshold

        if is_speech:
            self.speech_chunks.append(self.chunk_count)

        # State machine
        action = "continue"

        if not self.speech_started:
            # Waiting for speech to begin
            if is_speech:
                self.speech_started = True
                self.speech_start_chunk = self.chunk_count
                self.silence_start_time = None
                action = "speech_start"
                logger.debug(f"Speech started at chunk {self.chunk_count} (prob={prob:.3f})")
            elif elapsed >= self.config.pre_speech_timeout:
                action = "timeout_no_speech"
                logger.info(f"No speech detected after {elapsed:.1f}s — timeout")
        else:
            # Speech has started — watching for end-of-speech
            if is_speech:
                # Still speaking — reset silence timer
                self.silence_start_time = None
            else:
                # Silence chunk
                if self.silence_start_time is None:
                    self.silence_start_time = current_time

                silence_duration = current_time - self.silence_start_time
                if silence_duration >= self.config.silence_timeout:
                    self.speech_ended = True
                    self.speech_end_chunk = self.chunk_count
                    action = "speech_end"
                    logger.debug(
                        f"Speech ended at chunk {self.chunk_count} "
                        f"(silence={silence_duration:.2f}s)"
                    )

        # Max duration safety cap (always checked)
        if elapsed >= self.config.max_duration and action == "continue":
            action = "timeout_max_duration"
            logger.info(f"Max duration {self.config.max_duration}s reached")

        return {
            "prob": prob,
            "is_speech": is_speech,
            "action": action,
            "elapsed": elapsed,
            "chunk": self.chunk_count,
        }

    def get_speech_boundaries(self) -> Tuple[int, int]:
        """
        Return (start_chunk, end_chunk) of detected speech.
        If no speech, returns (0, chunk_count).
        """
        start = max(0, self.speech_start_chunk - 3)  # include 3 chunks before speech (~96ms lead-in)
        end = self.speech_end_chunk if self.speech_end_chunk > 0 else self.chunk_count
        return start, end

    def get_speech_duration_ms(self) -> float:
        """Return estimated speech duration in ms based on speech chunks."""
        if not self.speech_chunks:
            return 0.0
        return len(self.speech_chunks) * SILERO_CHUNK_DURATION_S * 1000


# ---------------------------------------------------------------------------
# Full recorder (VadEngine + audio I/O)
# ---------------------------------------------------------------------------

class VadRecorder:
    """
    Records audio from microphone with VAD-based end-of-speech detection.
    For live use — requires PyAudio and a microphone.
    """

    def __init__(self, config: Optional[VadConfig] = None):
        self.config = config or VadConfig()
        self.engine = VadEngine(self.config)

    def record(self) -> VadRecordingResult:
        """
        Record from microphone until speech ends or timeout.
        Returns VadRecordingResult with WAV audio bytes.
        """
        try:
            import pyaudio
        except ImportError:
            raise ImportError("pyaudio required — run: pip install pyaudio")

        self.engine.load_model()
        self.engine.reset()

        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=SILERO_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=SILERO_CHUNK_SAMPLES,
        )

        all_chunks: List[bytes] = []
        t0 = time.monotonic()

        logger.info("VAD recorder: listening for speech...")

        try:
            while True:
                raw = stream.read(SILERO_CHUNK_SAMPLES, exception_on_overflow=False)
                all_chunks.append(raw)

                # Convert int16 → float32 for Silero
                audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                result = self.engine.process_chunk(audio_f32, time.monotonic())

                if result["action"] in ("speech_end", "timeout_no_speech", "timeout_max_duration"):
                    break
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

        total_ms = (time.monotonic() - t0) * 1000

        # Extract just the speech portion
        start_chunk, end_chunk = self.engine.get_speech_boundaries()
        speech_chunks_raw = all_chunks[start_chunk:end_chunk]

        # Build WAV
        audio_bytes = self._chunks_to_wav(speech_chunks_raw)
        speech_ms = self.engine.get_speech_duration_ms()

        speech_detected = self.engine.speech_started

        logger.info(
            f"VAD recording done: speech={speech_detected}, "
            f"speech_ms={speech_ms:.0f}, total_ms={total_ms:.0f}, "
            f"chunks={self.engine.chunk_count}"
        )

        return VadRecordingResult(
            audio_bytes=audio_bytes,
            speech_duration_ms=speech_ms,
            total_duration_ms=total_ms,
            ended_by_silence=self.engine.speech_ended,
            speech_detected=speech_detected,
            chunks_processed=self.engine.chunk_count,
            vad_probs=self.engine.vad_probs,
        )

    @staticmethod
    def _chunks_to_wav(chunks: List[bytes]) -> bytes:
        """Convert raw int16 PCM chunks to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SILERO_SAMPLE_RATE)
            wf.writeframes(b"".join(chunks))
        return buf.getvalue()


# ---------------------------------------------------------------------------
# Process-from-buffer variant (for testing / non-live use)
# ---------------------------------------------------------------------------

def process_audio_buffer(
    audio_data: np.ndarray,
    config: Optional[VadConfig] = None,
) -> VadRecordingResult:
    """
    Run VAD on an in-memory audio buffer (float32, 16kHz).
    Useful for testing without a microphone.

    Args:
        audio_data: numpy float32 array (16kHz mono)
        config: VadConfig (optional)

    Returns:
        VadRecordingResult
    """
    config = config or VadConfig()
    engine = VadEngine(config)
    engine.load_model()
    engine.reset()

    # Pad to multiple of SILERO_CHUNK_SAMPLES
    pad_len = SILERO_CHUNK_SAMPLES - (len(audio_data) % SILERO_CHUNK_SAMPLES)
    if pad_len < SILERO_CHUNK_SAMPLES:
        audio_data = np.concatenate([audio_data, np.zeros(pad_len, dtype=np.float32)])

    all_chunks_raw: List[bytes] = []
    t_sim = 0.0

    for i in range(0, len(audio_data), SILERO_CHUNK_SAMPLES):
        chunk = audio_data[i : i + SILERO_CHUNK_SAMPLES]
        # Also store as int16 for WAV output
        int16_chunk = (chunk * 32768).astype(np.int16).tobytes()
        all_chunks_raw.append(int16_chunk)

        result = engine.process_chunk(chunk, t_sim)
        t_sim += SILERO_CHUNK_DURATION_S

        if result["action"] in ("speech_end", "timeout_no_speech", "timeout_max_duration"):
            break

    start_chunk, end_chunk = engine.get_speech_boundaries()
    speech_chunks = all_chunks_raw[start_chunk:end_chunk]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SILERO_SAMPLE_RATE)
        wf.writeframes(b"".join(speech_chunks))

    speech_ms = engine.get_speech_duration_ms()
    total_ms = t_sim * 1000

    return VadRecordingResult(
        audio_bytes=buf.getvalue(),
        speech_duration_ms=speech_ms,
        total_duration_ms=total_ms,
        ended_by_silence=engine.speech_ended,
        speech_detected=engine.speech_started,
        chunks_processed=engine.chunk_count,
        vad_probs=engine.vad_probs,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vad_recorder(config: Optional[VadConfig] = None) -> VadRecorder:
    """Create a VadRecorder with the given config."""
    return VadRecorder(config)


# ---------------------------------------------------------------------------
# CLI test mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Claudette Home — VAD Recorder test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--silence-timeout", type=float, default=1.5)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--wav", help="Process a WAV file instead of live mic")
    args = parser.parse_args()

    config = VadConfig(
        threshold=args.threshold,
        silence_timeout=args.silence_timeout,
        max_duration=args.max_duration,
    )

    if args.wav:
        import soundfile as sf

        audio, sr = sf.read(args.wav, dtype="float32")
        if sr != 16000:
            print(f"Warning: resampling from {sr}Hz to 16000Hz not implemented. Use 16kHz WAV.")
            sys.exit(1)
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # mono

        result = process_audio_buffer(audio, config)
    else:
        recorder = create_vad_recorder(config)
        result = recorder.record()

    print(f"\n{'='*50}")
    print(f"Speech detected:  {result.speech_detected}")
    print(f"Speech duration:  {result.speech_duration_ms:.0f} ms")
    print(f"Total duration:   {result.total_duration_ms:.0f} ms")
    print(f"Ended by silence: {result.ended_by_silence}")
    print(f"Chunks processed: {result.chunks_processed}")
    print(f"Audio size:       {len(result.audio_bytes)} bytes")
