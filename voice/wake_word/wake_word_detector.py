#!/usr/bin/env python3
"""
Claudette Wake Word Detection System

Uses Porcupine (Picovoice) for local, offline wake word detection.
Designed to run on the MOES panel (Android) or Workshop (Linux).

Usage:
    python3 wake_word_detector.py --mode test          # Test with built-in keywords
    python3 wake_word_detector.py --mode claudette     # Use custom "Claudette" model
    python3 wake_word_detector.py --mode file --input audio.wav  # Test on audio file

Environment:
    PICOVOICE_ACCESS_KEY - Required for custom wake words (get from picovoice.ai)
"""

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import soundfile as sf

# Optional imports - only needed for microphone access
try:
    from pvrecorder import PvRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False


@dataclass
class WakeWordConfig:
    """Configuration for wake word detection."""
    keyword_paths: List[str]
    sensitivities: List[float]
    access_key: Optional[str] = None
    model_path: Optional[str] = None
    library_path: Optional[str] = None
    frame_length: int = 512  # Porcupine requires 512 samples per frame
    sample_rate: int = 16000  # Porcupine requires 16kHz


class WakeWordDetector:
    """
    Local, offline wake word detector using Porcupine.
    
    Features:
    - Runs entirely offline (no cloud)
    - Very low false positive rate
    - Custom wake word support via Picovoice Console
    - Cross-platform (Linux, macOS, Windows, Android, iOS)
    """
    
    def __init__(self, config: WakeWordConfig):
        if not PORCUPINE_AVAILABLE:
            raise RuntimeError("pvporcupine not installed. Run: pip install pvporcupine")
        
        self.config = config
        self.porcupine: Optional[pvporcupine.Porcupine] = None
        self._callback: Optional[Callable[[int], None]] = None
        
    def initialize(self) -> bool:
        """Initialize the Porcupine engine."""
        try:
            kwargs = {
                'keyword_paths': self.config.keyword_paths,
                'sensitivities': self.config.sensitivities,
            }
            
            if self.config.access_key:
                kwargs['access_key'] = self.config.access_key
            if self.config.model_path:
                kwargs['model_path'] = self.config.model_path
            if self.config.library_path:
                kwargs['library_path'] = self.config.library_path
                
            self.porcupine = pvporcupine.create(**kwargs)
            
            # Validate sample rate
            if self.porcupine.sample_rate != self.config.sample_rate:
                print(f"Warning: Porcupine sample rate is {self.porcupine.sample_rate}, "
                      f"expected {self.config.sample_rate}")
                
            return True
            
        except Exception as e:
            print(f"Failed to initialize Porcupine: {e}")
            return False
    
    def process_frame(self, pcm: np.ndarray) -> int:
        """
        Process a single audio frame.
        
        Args:
            pcm: Audio samples as int16 array, must be frame_length samples
            
        Returns:
            Index of detected keyword, or -1 if no keyword detected
        """
        if self.porcupine is None:
            raise RuntimeError("Detector not initialized. Call initialize() first.")
            
        if len(pcm) != self.porcupine.frame_length:
            raise ValueError(f"Frame must be {self.porcupine.frame_length} samples, got {len(pcm)}")
            
        return self.porcupine.process(pcm)
    
    def process_audio_file(self, audio_path: str, callback: Optional[Callable[[int, float], None]] = None) -> List[int]:
        """
        Process an audio file and detect wake words.
        
        Args:
            audio_path: Path to audio file (any format soundfile can read)
            callback: Optional callback(index, timestamp_seconds) called on detection
            
        Returns:
            List of detection indices
        """
        # Read audio file
        audio, sample_rate = sf.read(audio_path, dtype='int16')
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1).astype(np.int16)
        
        # Resample to 16kHz if needed (simple downsampling)
        if sample_rate != self.config.sample_rate:
            ratio = sample_rate / self.config.sample_rate
            indices = np.round(np.arange(0, len(audio), ratio)).astype(int)
            indices = indices[indices < len(audio)]
            audio = audio[indices]
        
        # Process frame by frame
        detections = []
        frame_length = self.porcupine.frame_length
        
        for i in range(0, len(audio) - frame_length + 1, frame_length):
            frame = audio[i:i + frame_length]
            result = self.process_frame(frame)
            
            if result >= 0:
                timestamp = i / self.config.sample_rate
                detections.append(result)
                if callback:
                    callback(result, timestamp)
                    
        return detections
    
    def start_microphone_stream(self, device_index: int = -1, 
                                 callback: Optional[Callable[[int], None]] = None) -> None:
        """
        Start listening on microphone.
        
        Args:
            device_index: Audio device index (-1 for default)
            callback: Called with keyword index when wake word detected
        """
        if not RECORDER_AVAILABLE:
            raise RuntimeError("pvrecorder not installed. Run: pip install pvrecorder")
            
        self._callback = callback
        
        recorder = PvRecorder(
            frame_length=self.porcupine.frame_length,
            device_index=device_index
        )
        
        print(f"Listening for wake words... (Press Ctrl+C to stop)")
        print(f"Sample rate: {self.config.sample_rate}Hz, Frame length: {self.porcupine.frame_length}")
        
        try:
            recorder.start()
            
            while True:
                pcm = recorder.read()
                pcm_array = np.array(pcm, dtype=np.int16)
                result = self.process_frame(pcm_array)
                
                if result >= 0:
                    keyword_name = Path(self.config.keyword_paths[result]).stem
                    print(f"[WAKE WORD DETECTED] '{keyword_name}' at {time.strftime('%H:%M:%S')}")
                    
                    if self._callback:
                        self._callback(result)
                        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            recorder.stop()
            recorder.delete()
    
    def release(self):
        """Release resources."""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None


def get_builtin_keywords():
    """Get list of available built-in keywords."""
    if not PORCUPINE_AVAILABLE:
        return []
    # Porcupine 4.x returns a set, convert to sorted list
    return sorted(list(pvporcupine.KEYWORDS))


def create_claudette_config(access_key: str, model_path: Optional[str] = None) -> WakeWordConfig:
    """
    Create config for custom 'Claudette' wake word.
    
    Requires:
    - Picovoice Console account (free for personal use)
    - Trained 'Claudette' model (.ppn file)
    
    Get your access key at: https://console.picovoice.ai/
    Train custom wake word at: https://console.picovoice.ai/ppn
    """
    if model_path is None:
        # Look for model in standard locations
        possible_paths = [
            Path(__file__).parent / "claudette.ppn",
            Path(__file__).parent / "models" / "claudette.ppn",
            Path("/opt/claudette/models/claudette.ppn"),
        ]
        for path in possible_paths:
            if path.exists():
                model_path = str(path)
                break
        else:
            raise FileNotFoundError(
                "Custom 'Claudette' model not found. Train one at "
                "https://console.picovoice.ai/ppn or use --mode test "
                "to try built-in keywords."
            )
    
    return WakeWordConfig(
        keyword_paths=[model_path],
        sensitivities=[0.5],  # 0.0 to 1.0, higher = more sensitive
        access_key=access_key,
    )


def create_builtin_config(keywords: List[str], access_key: Optional[str] = None) -> WakeWordConfig:
    """Create config using built-in Porcupine keywords."""
    keyword_paths = [
        pvporcupine.KEYWORD_PATHS[kw] for kw in keywords
    ]
    return WakeWordConfig(
        keyword_paths=keyword_paths,
        sensitivities=[0.5] * len(keywords),
        access_key=access_key,
    )


def main():
    parser = argparse.ArgumentParser(description="Claudette Wake Word Detection")
    parser.add_argument("--mode", choices=["test", "claudette", "file"], default="test",
                       help="Detection mode: test=built-in keywords, claudette=custom model")
    parser.add_argument("--input", type=str, help="Audio file path (for file mode)")
    parser.add_argument("--device", type=int, default=-1, help="Audio device index")
    parser.add_argument("--model", type=str, help="Path to custom .ppn model file")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="Detection sensitivity (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Create configuration based on mode
    if args.mode == "claudette":
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            print("Error: PICOVOICE_ACCESS_KEY environment variable required for custom wake words")
            print("Get your free key at: https://console.picovoice.ai/")
            sys.exit(1)
        config = create_claudette_config(access_key, args.model)
        
    elif args.mode == "test":
        # Use built-in keywords for testing - still needs access key in v4.x
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            print("Error: PICOVOICE_ACCESS_KEY environment variable required")
            print("Get your free key at: https://console.picovoice.ai/")
            print("\nAlternatively, use --mode file to test on an audio file without a key.")
            sys.exit(1)

        available = get_builtin_keywords()
        print(f"Available built-in keywords: {', '.join(available[:10])}...")
        print("Using 'porcupine' and 'hey google' for testing\n")

        config = create_builtin_config(["porcupine", "hey google"], access_key=access_key)
        
    else:  # file mode
        if not args.input:
            print("Error: --input required for file mode")
            sys.exit(1)
        # File mode still needs access key for now
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
        if not access_key:
            print("Error: PICOVOICE_ACCESS_KEY environment variable required")
            print("Get your free key at: https://console.picovoice.ai/")
            sys.exit(1)
        config = create_builtin_config(["porcupine"], access_key=access_key)
    
    # Initialize detector
    detector = WakeWordDetector(config)
    if not detector.initialize():
        sys.exit(1)
    
    try:
        if args.mode == "file":
            print(f"Processing: {args.input}")
            
            def on_detect(index: int, timestamp: float):
                print(f"  Detection at {timestamp:.2f}s (keyword {index})")
                
            detections = detector.process_audio_file(args.input, on_detect)
            print(f"\nTotal detections: {len(detections)}")
            
        else:
            # Microphone mode
            def on_wake(index: int):
                print(f"  -> Wake word triggered! Starting STT pipeline...")
                # Here we would trigger the STT pipeline
                # In production: POST to localhost:8765/transcribe
                
            detector.start_microphone_stream(device_index=args.device, callback=on_wake)
            
    finally:
        detector.release()


if __name__ == "__main__":
    main()
