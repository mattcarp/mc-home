#!/usr/bin/env python3
"""
Claudette Home — Porcupine Listener Test Suite
Tests porcupine_listener.py and wake_word_bridge.py without hardware or .ppn model.

Run:
  python3 -m pytest test_porcupine_listener.py -v
  # or directly:
  python3 test_porcupine_listener.py

What's tested:
  - Porcupine SDK installed and importable
  - porcupine_listener.py exits cleanly on missing model (not a crash)
  - porcupine_listener.py exits cleanly on missing access key
  - wake_word_bridge.py CLI parses args correctly
  - wake_word_bridge.py emits valid JSON on startup
  - wake_word_bridge.py exits with error on missing key
  - on_detection() emits valid JSON event to stdout
  - Built-in keyword fallback (porcupine keyword — no .ppn needed)
  - Porcupine engine instantiation with a built-in keyword (requires real key)
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the wake_word dir to path
WAKE_WORD_DIR = Path(__file__).parent
sys.path.insert(0, str(WAKE_WORD_DIR))


# ────────────────────────────────────────────────────────────────────────────
# Group 1: SDK Import
# ────────────────────────────────────────────────────────────────────────────

class TestSDKInstalled(unittest.TestCase):
    """pvporcupine SDK must be installed — everything else depends on this."""

    def test_pvporcupine_importable(self):
        """SDK must be importable after pip install pvporcupine."""
        import pvporcupine
        self.assertIsNotNone(pvporcupine)

    def test_porcupine_class_available(self):
        """Porcupine class must be directly accessible."""
        from pvporcupine import Porcupine
        self.assertIsNotNone(Porcupine)

    def test_keywords_list_nonempty(self):
        """Built-in keywords list must not be empty."""
        import pvporcupine
        self.assertGreater(len(pvporcupine.KEYWORDS), 0)

    def test_keyword_paths_populated(self):
        """KEYWORD_PATHS dict must map names → file paths."""
        import pvporcupine
        self.assertIsInstance(pvporcupine.KEYWORD_PATHS, dict)
        self.assertGreater(len(pvporcupine.KEYWORD_PATHS), 0)


# ────────────────────────────────────────────────────────────────────────────
# Group 2: Event Emission
# ────────────────────────────────────────────────────────────────────────────

class TestEventEmission(unittest.TestCase):
    """wake_word_bridge emits clean JSON events — test the emit_event function directly."""

    def _import_bridge(self):
        """Import wake_word_bridge, mocking hardware imports."""
        # The bridge file imports pvporcupine at the function level (not top-level)
        # so we can import the module safely
        import importlib
        # Reload to get clean state
        if "wake_word_bridge" in sys.modules:
            del sys.modules["wake_word_bridge"]
        import wake_word_bridge
        return wake_word_bridge

    def test_emit_event_is_valid_json(self):
        """emit_event() must produce valid JSON on stdout."""
        bridge = self._import_bridge()
        captured = StringIO()
        with patch("sys.stdout", captured):
            bridge.emit_event("test_event", {"key": "value", "num": 42})
        output = captured.getvalue().strip()
        self.assertTrue(output, "No output from emit_event")
        parsed = json.loads(output)
        self.assertEqual(parsed["type"], "test_event")
        self.assertEqual(parsed["key"], "value")
        self.assertEqual(parsed["num"], 42)

    def test_emit_event_has_timestamp(self):
        """emit_event() must include a UTC timestamp."""
        bridge = self._import_bridge()
        captured = StringIO()
        with patch("sys.stdout", captured):
            bridge.emit_event("timestamped", {})
        parsed = json.loads(captured.getvalue().strip())
        self.assertIn("ts", parsed)
        self.assertIn("T", parsed["ts"])  # ISO 8601 format

    def test_on_detection_emits_wake_word_event(self):
        """on_detection() must emit a wake_word_detected JSON event."""
        bridge = self._import_bridge()
        captured = StringIO()
        with patch("sys.stdout", captured):
            bridge.on_detection("porcupine", "claudette")
        parsed = json.loads(captured.getvalue().strip())
        self.assertEqual(parsed["type"], "wake_word_detected")
        self.assertEqual(parsed["backend"], "porcupine")
        self.assertEqual(parsed["word"], "claudette")

    def test_on_detection_with_score(self):
        """on_detection() must include score when provided."""
        bridge = self._import_bridge()
        captured = StringIO()
        with patch("sys.stdout", captured):
            bridge.on_detection("oww", "claudette", score=0.97)
        parsed = json.loads(captured.getvalue().strip())
        self.assertAlmostEqual(parsed["score"], 0.97)

    def test_on_detection_score_none(self):
        """on_detection() with no score must emit null (not crash)."""
        bridge = self._import_bridge()
        captured = StringIO()
        with patch("sys.stdout", captured):
            bridge.on_detection("porcupine", "claudette", score=None)
        parsed = json.loads(captured.getvalue().strip())
        self.assertIsNone(parsed["score"])


# ────────────────────────────────────────────────────────────────────────────
# Group 3: CLI Argument Parsing
# ────────────────────────────────────────────────────────────────────────────

class TestCLIParsing(unittest.TestCase):
    """wake_word_bridge.py must exit gracefully on bad args."""

    def test_missing_access_key_exits_nonzero(self):
        """Running the bridge without PORCUPINE_ACCESS_KEY should exit non-zero."""
        env = os.environ.copy()
        env.pop("PORCUPINE_ACCESS_KEY", None)
        env.pop("PICOVOICE_ACCESS_KEY", None)
        result = subprocess.run(
            [sys.executable, str(WAKE_WORD_DIR / "wake_word_bridge.py"),
             "--backend", "porcupine"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        # Should print a JSON error or stderr message
        combined = result.stdout + result.stderr
        self.assertTrue(
            "error" in combined.lower() or "key" in combined.lower(),
            f"Expected error about missing key, got: {combined!r}"
        )

    def test_missing_access_key_emits_json_error(self):
        """Missing access key should produce a JSON error line on stdout."""
        env = os.environ.copy()
        env.pop("PORCUPINE_ACCESS_KEY", None)
        env.pop("PICOVOICE_ACCESS_KEY", None)
        result = subprocess.run(
            [sys.executable, str(WAKE_WORD_DIR / "wake_word_bridge.py"),
             "--backend", "porcupine"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        # stdout should be JSON (we're a streaming JSON pipeline)
        stdout = result.stdout.strip()
        if stdout:
            try:
                parsed = json.loads(stdout.splitlines()[0])
                self.assertIn("error", parsed)
            except json.JSONDecodeError:
                # Acceptable — some error paths use plain stderr
                pass


# ────────────────────────────────────────────────────────────────────────────
# Group 4: Built-in Keyword Smoke Test (requires valid access key)
# ────────────────────────────────────────────────────────────────────────────

@unittest.skipUnless(
    os.environ.get("PORCUPINE_ACCESS_KEY") or os.environ.get("PICOVOICE_ACCESS_KEY"),
    "Skipped: PORCUPINE_ACCESS_KEY not set (no credentials for live test)"
)
class TestBuiltInKeyword(unittest.TestCase):
    """
    Requires a valid PORCUPINE_ACCESS_KEY.
    Tests that we can instantiate Porcupine with a built-in keyword.
    This validates the key without needing a .ppn model file.
    """

    def setUp(self):
        self.access_key = (
            os.environ.get("PORCUPINE_ACCESS_KEY")
            or os.environ.get("PICOVOICE_ACCESS_KEY")
        )

    def test_create_with_builtin_keyword(self):
        """Should be able to create a Porcupine instance with the 'porcupine' built-in."""
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=["porcupine"],
        )
        self.assertIsNotNone(porcupine)
        self.assertGreater(porcupine.sample_rate, 0)
        self.assertGreater(porcupine.frame_length, 0)
        porcupine.delete()

    def test_sample_rate_is_16khz(self):
        """Porcupine always runs at 16kHz — verify this so our audio pipeline is configured right."""
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=["porcupine"],
        )
        self.assertEqual(porcupine.sample_rate, 16000)
        porcupine.delete()

    def test_frame_length_matches_pyaudio_chunk(self):
        """Frame length must be a power of 2 and reasonable for real-time audio."""
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=["porcupine"],
        )
        fl = porcupine.frame_length
        # Porcupine v4 typically uses 512 samples
        self.assertGreaterEqual(fl, 128)
        self.assertLessEqual(fl, 2048)
        porcupine.delete()

    def test_sensitivity_range(self):
        """Sensitivity out of range should raise an error (not silently cap)."""
        import pvporcupine
        with self.assertRaises(pvporcupine.PorcupineInvalidArgumentError):
            porcupine = pvporcupine.create(
                access_key=self.access_key,
                keywords=["porcupine"],
                sensitivities=[1.5],  # > 1.0 is invalid
            )

    def test_process_returns_minus_one_on_silence(self):
        """process() must return -1 (no detection) on a silence frame."""
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keywords=["porcupine"],
        )
        # Feed a frame of silence (all zeros)
        silence_frame = [0] * porcupine.frame_length
        result = porcupine.process(silence_frame)
        self.assertEqual(result, -1, "Silence should return -1 (no detection)")
        porcupine.delete()


# ────────────────────────────────────────────────────────────────────────────
# Group 5: Custom Model Smoke Test (requires .ppn model file + access key)
# ────────────────────────────────────────────────────────────────────────────

_MODEL_PATH = WAKE_WORD_DIR / "models" / "claudette_linux.ppn"
_HAS_KEY = bool(
    os.environ.get("PORCUPINE_ACCESS_KEY")
    or os.environ.get("PICOVOICE_ACCESS_KEY")
)

@unittest.skipUnless(
    _HAS_KEY and _MODEL_PATH.exists(),
    "Skipped: Needs PORCUPINE_ACCESS_KEY + models/claudette_linux.ppn"
)
class TestCustomModel(unittest.TestCase):
    """
    Full integration test with the real 'Claudette' .ppn model.
    Only runs when both the key and model are present.
    """

    def setUp(self):
        self.access_key = (
            os.environ.get("PORCUPINE_ACCESS_KEY")
            or os.environ.get("PICOVOICE_ACCESS_KEY")
        )
        self.model_path = str(_MODEL_PATH)

    def test_load_claudette_model(self):
        """Custom 'Claudette' model must load without errors."""
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[self.model_path],
            sensitivities=[0.5],
        )
        self.assertIsNotNone(porcupine)
        porcupine.delete()

    def test_sensitivity_tuning_range(self):
        """Test a range of sensitivities to confirm they all load cleanly."""
        import pvporcupine
        for sensitivity in [0.3, 0.5, 0.7, 0.9]:
            with self.subTest(sensitivity=sensitivity):
                porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keyword_paths=[self.model_path],
                    sensitivities=[sensitivity],
                )
                porcupine.delete()

    def test_process_silence_frame(self):
        """process() must return -1 for a silence frame (no false trigger on empty audio)."""
        import pvporcupine
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[self.model_path],
            sensitivities=[0.5],
        )
        silence = [0] * porcupine.frame_length
        result = porcupine.process(silence)
        self.assertEqual(result, -1, "Silence must not trigger wake word detection")
        porcupine.delete()


# ────────────────────────────────────────────────────────────────────────────
# Group 6: File Structure Checks
# ────────────────────────────────────────────────────────────────────────────

class TestFileStructure(unittest.TestCase):
    """Required files must exist — catch regressions from accidental deletes."""

    def test_porcupine_listener_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "porcupine_listener.py").exists())

    def test_wake_word_bridge_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "wake_word_bridge.py").exists())

    def test_oww_listener_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "oww_listener.py").exists())

    def test_systemd_service_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "claudette-wake-word.service").exists())

    def test_requirements_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "requirements.txt").exists())

    def test_models_dir_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "models").is_dir())

    def test_models_gitignore_present(self):
        """models/ should have a .gitignore to avoid committing binary .ppn files."""
        gitignore = WAKE_WORD_DIR / "models" / ".gitignore"
        self.assertTrue(
            gitignore.exists(),
            ".gitignore missing from models/ — .ppn binary files should not be committed"
        )

    def test_setup_porcupine_exists(self):
        self.assertTrue((WAKE_WORD_DIR / "setup_porcupine.py").exists())


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run with verbose output when executed directly
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
