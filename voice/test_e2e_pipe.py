#!/usr/bin/env python3
"""
End-to-end pipe integration test for the Claudette Home voice pipeline.

Tests the full chain:
  wake_word_bridge.py (stub) | pipeline.py (stub) | tts_responder.py (dry-run)

All components run in stub / dry-run mode — no hardware required.
This test validates the pipe contracts and JSON event flow between all stages.

Run from the mc-home root:
  python3 -m pytest voice/test_e2e_pipe.py -v
  python3 voice/test_e2e_pipe.py            # direct
"""

import json
import subprocess
import sys
import os
import time
from pathlib import Path

VOICE_DIR = Path(__file__).parent
MC_HOME = VOICE_DIR.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_pipeline_text(utterance: str, timeout: int = 15) -> dict:
    """
    Run pipeline.py in stub + text mode.
    Returns the parsed JSON output.
    """
    result = subprocess.run(
        [sys.executable, str(VOICE_DIR / "pipeline.py"), "--stub", "--text", utterance],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(MC_HOME),
    )
    assert result.returncode == 0, (
        f"pipeline.py exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    # pipeline.py prints [STUB] lines to stdout before the JSON block.
    # Extract the JSON object by finding the first '{' that starts a line.
    stdout = result.stdout.strip()
    json_start = None
    for i, line in enumerate(stdout.splitlines()):
        if line.strip().startswith("{"):
            json_start = i
            break
    assert json_start is not None, f"No JSON found in pipeline output:\n{stdout}"
    json_text = "\n".join(stdout.splitlines()[json_start:])
    return json.loads(json_text)


def run_tts_responder_dry(pipeline_event: dict, timeout: int = 10) -> str:
    """
    Pipe a pipeline_response JSON event through tts_responder.py in dry-run mode.
    Returns the captured stdout (the event text).
    """
    event_json = json.dumps({"type": "pipeline_response", "text": pipeline_event.get("response", "")})
    result = subprocess.run(
        [sys.executable, str(VOICE_DIR / "tts_responder.py"), "--dry-run"],
        input=event_json + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(MC_HOME),
    )
    assert result.returncode == 0, (
        f"tts_responder.py exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    return result.stdout.strip()


def run_wake_word_bridge_stub(timeout: int = 5) -> list:
    """
    Run wake_word_bridge.py in stub mode and collect the emitted JSON events.
    """
    result = subprocess.run(
        [sys.executable, str(VOICE_DIR / "wake_word" / "wake_word_bridge.py"),
         "--backend", "stub", "--max-events", "3"],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(MC_HOME),
    )
    events = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Tests: Pipeline --text mode (stage 2 + 3)
# ---------------------------------------------------------------------------

class TestPipelineTextMode:
    """Stage 2 (intent parse + HA stub) + Stage 3 (response builder)."""

    def test_turn_on_light(self):
        out = run_pipeline_text("turn on the living room lights")
        assert out["action"]["action"] == "call_service"
        assert out["action"]["domain"] == "light"
        assert out["action"]["service"] == "turn_on"
        assert "living" in out["action"]["entity_id"]
        assert "on" in out["response"].lower() or "done" in out["response"].lower()

    def test_turn_off_light(self):
        out = run_pipeline_text("turn off the kitchen light")
        assert out["action"]["action"] == "call_service"
        assert out["action"]["service"] == "turn_off"
        assert "off" in out["response"].lower() or "done" in out["response"].lower()

    def test_lock_front_door(self):
        out = run_pipeline_text("lock the front door")
        assert out["action"]["action"] == "call_service"
        assert out["action"]["domain"] == "lock"
        assert out["action"]["service"] == "lock"
        assert "lock" in out["response"].lower()

    def test_unlock_front_door(self):
        out = run_pipeline_text("unlock the front door")
        assert out["action"]["action"] == "call_service"
        assert out["action"]["service"] == "unlock"

    def test_query_temperature(self):
        out = run_pipeline_text("what's the temperature?")
        assert out["action"]["action"] == "query"
        assert "temperature" in out["action"]["entity_id"]
        # In stub mode, response should be graceful (not crash)
        assert isinstance(out["response"], str)
        assert len(out["response"]) > 0

    def test_activate_scene(self):
        out = run_pipeline_text("activate the movie night scene")
        # Should map to scene or clarify
        assert out["action"]["action"] in ("call_service", "clarify")

    def test_multi_action(self):
        out = run_pipeline_text("turn off all the lights and lock the front door")
        # Multi-action returns a list
        action = out["action"]
        assert isinstance(action, list) or action.get("action") in ("call_service", "clarify")

    def test_close_shutters(self):
        out = run_pipeline_text("close the bedroom shutters")
        assert out["action"]["action"] in ("call_service", "clarify")

    def test_italian_utterance(self):
        """Multilingual support — Italian."""
        out = run_pipeline_text("spegni le luci del soggiorno")
        # Should parse to turn_off light
        assert out["action"]["action"] in ("call_service", "clarify")
        assert isinstance(out["response"], str)

    def test_maltese_utterance(self):
        """Multilingual support — Maltese."""
        out = run_pipeline_text("agħlaq id-dawl")
        assert out["action"]["action"] in ("call_service", "clarify")
        assert isinstance(out["response"], str)

    def test_results_structure(self):
        """Pipeline always returns required fields."""
        out = run_pipeline_text("turn on the entrance light")
        assert "transcript" in out
        assert "action" in out
        assert "results" in out
        assert "response" in out

    def test_transcript_preserved(self):
        """Transcript field matches what was sent."""
        utterance = "dim the bedroom light to fifty percent"
        out = run_pipeline_text(utterance)
        assert out["transcript"] == utterance


# ---------------------------------------------------------------------------
# Tests: TTS Responder (stage 4) dry-run
# ---------------------------------------------------------------------------

class TestTTSResponderDryRun:
    """Stage 4: tts_responder.py receives pipeline_response and handles it."""

    def test_speaks_response(self):
        """tts_responder processes a pipeline_response event without error."""
        pipeline_out = {"response": "Done, living room is on."}
        tts_out = run_tts_responder_dry(pipeline_out)
        # In dry-run mode, tts_responder should not crash
        # Output may be empty (no audio) or log line — either is fine
        assert isinstance(tts_out, str)

    def test_handles_locked_response(self):
        pipeline_out = {"response": "Locked."}
        tts_out = run_tts_responder_dry(pipeline_out)
        assert isinstance(tts_out, str)

    def test_handles_empty_response(self):
        pipeline_out = {"response": ""}
        # Should not crash on empty
        event_json = json.dumps({"type": "pipeline_response", "text": ""})
        result = subprocess.run(
            [sys.executable, str(VOICE_DIR / "tts_responder.py"), "--dry-run"],
            input=event_json + "\n",
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(MC_HOME),
        )
        assert result.returncode == 0

    def test_passthrough_non_response_event(self):
        """Non-pipeline_response events should pass through without crashing."""
        other_event = json.dumps({"type": "wake_word_detected", "word": "claudette"})
        result = subprocess.run(
            [sys.executable, str(VOICE_DIR / "tts_responder.py"), "--dry-run"],
            input=other_event + "\n",
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(MC_HOME),
        )
        assert result.returncode == 0


# ---------------------------------------------------------------------------
# Tests: Full pipe chain (stages 2+3+4)
# ---------------------------------------------------------------------------

class TestFullPipeChain:
    """
    End-to-end: pipeline.py (text mode) → tts_responder.py (dry-run).
    Validates that the JSON event produced by pipeline.py is correctly
    consumed by tts_responder.py.
    """

    def test_light_on_full_chain(self):
        pipeline_out = run_pipeline_text("turn on the kitchen light")
        tts_out = run_tts_responder_dry(pipeline_out)
        # Chain completed without error
        assert isinstance(tts_out, str)

    def test_lock_full_chain(self):
        pipeline_out = run_pipeline_text("lock the front door")
        tts_out = run_tts_responder_dry(pipeline_out)
        assert isinstance(tts_out, str)

    def test_query_full_chain(self):
        pipeline_out = run_pipeline_text("what's the humidity?")
        tts_out = run_tts_responder_dry(pipeline_out)
        assert isinstance(tts_out, str)
        # Response should be non-trivial
        assert len(pipeline_out["response"]) > 5

    def test_scene_full_chain(self):
        pipeline_out = run_pipeline_text("set the living room to dim")
        tts_out = run_tts_responder_dry(pipeline_out)
        assert isinstance(tts_out, str)

    def test_multi_room_full_chain(self):
        pipeline_out = run_pipeline_text("turn off all lights in the bedroom")
        tts_out = run_tts_responder_dry(pipeline_out)
        assert isinstance(tts_out, str)


# ---------------------------------------------------------------------------
# Tests: Response quality (build_response function)
# ---------------------------------------------------------------------------

class TestResponseQuality:
    """Check that build_response produces human-sounding output."""

    def test_light_on_response_natural(self):
        out = run_pipeline_text("turn on the living room lights")
        response = out["response"].lower()
        # Should not contain raw entity_id format
        assert "light." not in response
        assert "turn_on" not in response

    def test_lock_response_natural(self):
        out = run_pipeline_text("lock the front door")
        response = out["response"].lower()
        assert "locked" in response
        assert "lock." not in response

    def test_off_response_natural(self):
        out = run_pipeline_text("turn off the kitchen light")
        response = out["response"].lower()
        assert "lock." not in response
        assert "kitchen" in response or "off" in response or "done" in response

    def test_stub_query_response_graceful(self):
        """In stub mode, sensor query should return graceful 'no live reading' message."""
        out = run_pipeline_text("what's the temperature?")
        response = out["response"].lower()
        # Should NOT say "is stub" — that was the bug we fixed
        assert "is stub" not in response
        # Should give some kind of graceful response
        assert len(response) > 5


# ---------------------------------------------------------------------------
# Entry point for direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
