#!/usr/bin/env python3
"""
Tests for the WakeWordDetector class.

Run with: pytest test_wake_word.py -v
"""

import os
import sys
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

# Skip all tests if no access key
SKIP_INTEGRATION = os.environ.get("PICOVOICE_ACCESS_KEY") is None

import numpy as np
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from wake_word_detector import (
    WakeWordConfig,
    WakeWordDetector,
    create_builtin_config,
    get_builtin_keywords,
    PORCUPINE_AVAILABLE,
)


class TestWakeWordConfig:
    """Test configuration dataclass."""
    
    def test_default_config(self):
        config = WakeWordConfig(
            keyword_paths=["/path/to/model.ppn"],
            sensitivities=[0.5],
        )
        assert config.keyword_paths == ["/path/to/model.ppn"]
        assert config.sensitivities == [0.5]
        assert config.access_key is None
        assert config.sample_rate == 16000
        assert config.frame_length == 512


class TestBuiltInKeywords:
    """Test built-in keyword functionality."""
    
    def test_get_builtin_keywords(self):
        if not PORCUPINE_AVAILABLE:
            pytest.skip("Porcupine not installed")
        keywords = get_builtin_keywords()
        # Porcupine returns a list now
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Common keywords should be available
        assert "porcupine" in keywords
        assert "hey google" in keywords

    def test_create_builtin_config(self):
        if not PORCUPINE_AVAILABLE:
            pytest.skip("Porcupine not installed")
        config = create_builtin_config(["porcupine", "hey google"], access_key="test_key")
        assert len(config.keyword_paths) == 2
        assert len(config.sensitivities) == 2
        assert all(s == 0.5 for s in config.sensitivities)
        assert config.access_key == "test_key"


def create_test_wav_file(path: str, duration_sec: float = 1.0, 
                          frequency: float = 440.0, sample_rate: int = 16000):
    """Create a test WAV file with a sine wave."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
    # Generate sine wave
    audio = np.sin(2 * np.pi * frequency * t) * 32767
    audio = audio.astype(np.int16)
    
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())


class TestWakeWordDetector:
    """Test the WakeWordDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector with built-in keywords for testing."""
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            pytest.skip("PICOVOICE_ACCESS_KEY not set")
        config = create_builtin_config(["porcupine"], access_key=access_key)
        detector = WakeWordDetector(config)
        yield detector
        detector.release()
    
    def test_initialization(self, detector):
        """Test that detector initializes successfully."""
        success = detector.initialize()
        assert success is True
        assert detector.porcupine is not None
        
    def test_frame_length_validation(self, detector):
        """Test that wrong frame size raises error."""
        detector.initialize()
        
        # Wrong frame length should raise ValueError
        wrong_frame = np.zeros(256, dtype=np.int16)
        with pytest.raises(ValueError):
            detector.process_frame(wrong_frame)
    
    def test_silence_detection(self, detector):
        """Test that silence doesn't trigger detection."""
        detector.initialize()
        
        # Process silence (zeros)
        silence = np.zeros(detector.porcupine.frame_length, dtype=np.int16)
        result = detector.process_frame(silence)
        
        # Should return -1 (no detection)
        assert result == -1
    
    def test_audio_file_processing(self, detector, tmp_path):
        """Test processing an audio file."""
        detector.initialize()
        
        # Create test audio file (won't contain wake word, just testing pipeline)
        test_file = tmp_path / "test.wav"
        create_test_wav_file(str(test_file), duration_sec=2.0)
        
        # Process file
        detections = detector.process_audio_file(str(test_file))
        
        # No wake words in synthetic sine wave
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_callback_invocation(self, detector, tmp_path):
        """Test that callback is invoked on detection."""
        detector.initialize()
        
        # Create test file
        test_file = tmp_path / "test.wav"
        create_test_wav_file(str(test_file), duration_sec=0.5)
        
        # Mock callback
        callback_mock = MagicMock()
        
        # Process
        detector.process_audio_file(str(test_file), callback_mock)
        
        # Callback may or may not be called (depends on audio content)
        # We're just verifying the code path works


class TestIntegration:
    """Integration tests that require actual Porcupine engine."""

    def test_full_pipeline_builtin(self):
        """Test full pipeline with built-in keyword."""
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            pytest.skip("PICOVOICE_ACCESS_KEY not set")
        config = create_builtin_config(["porcupine"], access_key=access_key)
        detector = WakeWordDetector(config)
        
        try:
            assert detector.initialize()
            assert detector.porcupine is not None
            
            # Process some silence frames
            for _ in range(10):
                silence = np.zeros(detector.porcupine.frame_length, dtype=np.int16)
                result = detector.process_frame(silence)
                assert result == -1  # No detection on silence
                
        finally:
            detector.release()


def test_cli_help():
    """Test CLI help output."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "wake_word_detector.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--mode" in result.stdout
    assert "--input" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
