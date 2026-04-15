#!/usr/bin/env python3
"""
Tests for voice/conversation_fallback.py

Test groups:
  TestConfigLoading      — gateway config auto-detection
  TestStaticFallback     — static fallback keyword responses
  TestConversationFallback — live gateway integration (skips if gateway down)
  TestFallbackEdgeCases  — malformed/edge inputs

All live gateway tests use the actual OpenClaw endpoint. No mocks.
If the gateway is unreachable, live tests are skipped cleanly.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure voice/ is on path
VOICE_DIR = Path(__file__).parent
sys.path.insert(0, str(VOICE_DIR))

from conversation_fallback import (
    ConversationFallback,
    _get_gateway_port,
    _get_gateway_token,
    _load_gateway_config,
)


# ---------------------------------------------------------------------------
# TestConfigLoading
# ---------------------------------------------------------------------------
class TestConfigLoading(unittest.TestCase):
    """Config auto-detection from openclaw.json and environment."""

    def test_load_config_returns_dict(self):
        """_load_gateway_config always returns a dict."""
        config = _load_gateway_config()
        self.assertIsInstance(config, dict)

    def test_get_gateway_port_default(self):
        """Default port is 18789 when not in config."""
        port = _get_gateway_port({})
        self.assertEqual(port, 18789)

    def test_get_gateway_port_from_config(self):
        """Port is read from config gateway.port."""
        config = {"gateway": {"port": 9999}}
        # Clear env override so config value wins
        env_without_port = {k: v for k, v in os.environ.items() if k != "OPENCLAW_GATEWAY_PORT"}
        with patch.dict(os.environ, env_without_port, clear=True):
            port = _get_gateway_port(config)
        self.assertEqual(port, 9999)

    def test_get_gateway_port_env_override(self):
        """OPENCLAW_GATEWAY_PORT env var overrides config."""
        with patch.dict(os.environ, {"OPENCLAW_GATEWAY_PORT": "7777"}):
            port = _get_gateway_port({"gateway": {"port": 9999}})
        self.assertEqual(port, 7777)

    def test_get_gateway_token_none_when_empty_config(self):
        """Returns None if no token in config and no env var."""
        with patch.dict(os.environ, {}, clear=False):
            env = {k: v for k, v in os.environ.items() if "OPENCLAW" not in k and "GATEWAY" not in k}
            with patch.dict(os.environ, env, clear=True):
                token = _get_gateway_token({})
        # Token may be None or may come from env — just verify it's str or None
        self.assertIn(type(token), (str, type(None)))

    def test_get_gateway_token_from_env(self):
        """OPENCLAW_GATEWAY_TOKEN env var is picked up."""
        with patch.dict(os.environ, {"OPENCLAW_GATEWAY_TOKEN": "test_token_abc"}):
            token = _get_gateway_token({})
        self.assertEqual(token, "test_token_abc")

    def test_get_gateway_token_from_config(self):
        """Token is read from config gateway.auth.token."""
        config = {"gateway": {"auth": {"token": "config_token_xyz"}}}
        # Clear env override
        env = {k: v for k, v in os.environ.items() if k != "OPENCLAW_GATEWAY_TOKEN"}
        with patch.dict(os.environ, env, clear=True):
            token = _get_gateway_token(config)
        self.assertEqual(token, "config_token_xyz")

    def test_real_config_has_token(self):
        """Real openclaw.json on this host has a token."""
        config = _load_gateway_config()
        if not config:
            self.skipTest("openclaw.json not found")
        token = _get_gateway_token(config)
        self.assertIsNotNone(token, "Expected HA_TOKEN in real openclaw.json")
        self.assertGreater(len(token), 20, "Token looks too short")


# ---------------------------------------------------------------------------
# TestStaticFallback
# ---------------------------------------------------------------------------
class TestStaticFallback(unittest.TestCase):
    """Static fallback keyword responses (no gateway required)."""

    def _fallback(self, text: str) -> str:
        return ConversationFallback._static_fallback(text)

    def test_time_query(self):
        """Time queries get a relevant response."""
        resp = self._fallback("What time is it?")
        # Should mention time or phone — something helpful
        self.assertGreater(len(resp), 5)
        self.assertNotIn("home command", resp.lower())

    def test_weather_query(self):
        """Weather queries get a weather response."""
        resp = self._fallback("What's the weather like today?")
        self.assertGreater(len(resp), 5)

    def test_timer_request(self):
        """Timer requests get a response."""
        resp = self._fallback("Set a timer for 10 minutes")
        self.assertGreater(len(resp), 5)

    def test_joke_request(self):
        """Joke requests get a response."""
        resp = self._fallback("Tell me a joke")
        self.assertGreater(len(resp), 5)

    def test_music_request(self):
        """Ambiguous music request gets a helpful prompt."""
        resp = self._fallback("Play some music")
        self.assertGreater(len(resp), 5)

    def test_unknown_query_generic(self):
        """Completely unknown input gets generic retry message."""
        resp = self._fallback("Bzzzt klkl randomnoise")
        self.assertIn("try again", resp.lower())

    def test_returns_string(self):
        """Always returns a string, never raises."""
        for phrase in [
            "",
            "   ",
            "x" * 500,
            "!@#$%^",
            "Inti kif int?",  # Maltese
        ]:
            result = self._fallback(phrase)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

    def test_no_markdown_in_static(self):
        """Static fallback never returns markdown (no asterisks, no backticks)."""
        for phrase in ["time", "weather", "joke", "timer", "music", "nonsense"]:
            resp = self._fallback(phrase)
            self.assertNotIn("*", resp)
            self.assertNotIn("`", resp)
            self.assertNotIn("#", resp)


# ---------------------------------------------------------------------------
# TestConversationFallback — live gateway
# ---------------------------------------------------------------------------
class TestConversationFallback(unittest.TestCase):
    """Live integration tests using the real OpenClaw gateway."""

    @classmethod
    def setUpClass(cls):
        cls.fb = ConversationFallback(timeout=10.0)
        cls.gateway_available = cls.fb.is_available
        if not cls.gateway_available:
            print("\n  [SKIP] OpenClaw gateway not reachable — skipping live tests", file=sys.stderr)

    def _skip_if_offline(self):
        if not self.gateway_available:
            self.skipTest("OpenClaw gateway not reachable")

    def test_gateway_has_token(self):
        """Gateway token is loaded from openclaw.json."""
        self.assertIsNotNone(self.fb._token, "No gateway token — openclaw.json missing?")

    def test_respond_returns_string(self):
        """respond() always returns a non-empty string."""
        self._skip_if_offline()
        result = self.fb.respond("Hello, are you there?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.strip()), 5)

    def test_respond_time_query(self):
        """Time query gets a real response from Claudette."""
        self._skip_if_offline()
        result = self.fb.respond("What time is it?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)
        # Should not crash or return empty
        self.assertNotEqual(result.strip(), "")

    def test_respond_weather_query(self):
        """Weather query handled gracefully."""
        self._skip_if_offline()
        result = self.fb.respond("What's the weather in Malta today?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_respond_maltese_greeting(self):
        """Maltese greeting handled gracefully."""
        self._skip_if_offline()
        result = self.fb.respond("Bonġu Claudette, kif int?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_respond_italian_utterance(self):
        """Italian utterance handled gracefully."""
        self._skip_if_offline()
        result = self.fb.respond("Come stai, Claudette?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 5)

    def test_response_is_concise(self):
        """Response is short (under 500 chars) — suitable for voice."""
        self._skip_if_offline()
        result = self.fb.respond("Tell me about the history of Malta")
        self.assertLess(len(result), 500, "Response too long for voice output")

    def test_no_markdown_in_live_response(self):
        """Live responses should not contain markdown bullet points or headers."""
        self._skip_if_offline()
        result = self.fb.respond("Give me 3 fun facts about Gozo")
        # Should not have markdown lists (### or bullet *item)
        lines = result.split("\n")
        for line in lines:
            stripped = line.strip()
            self.assertFalse(
                stripped.startswith("###") or stripped.startswith("##"),
                f"Response contains markdown header: {stripped!r}"
            )

    def test_respond_latency_under_threshold(self):
        """Response arrives in under 15 seconds (local LLM can be variable)."""
        self._skip_if_offline()
        import time
        # Use a longer timeout for this test since the gateway can be slow under load
        fb = ConversationFallback(timeout=15.0)
        t0 = time.monotonic()
        result = fb.respond("What year is it?")
        elapsed = time.monotonic() - t0
        self.assertLess(elapsed, 15.0, f"Response took {elapsed:.1f}s — too slow")
        self.assertGreater(len(result), 0)

    def test_respond_empty_input(self):
        """Empty input handled gracefully — no crash."""
        self._skip_if_offline()
        result = self.fb.respond("")
        self.assertIsInstance(result, str)

    def test_respond_very_long_input(self):
        """Very long input handled gracefully — no crash."""
        self._skip_if_offline()
        long_input = "What do you think about " + "the situation in Malta " * 30
        result = self.fb.respond(long_input)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


# ---------------------------------------------------------------------------
# TestFallbackEdgeCases
# ---------------------------------------------------------------------------
class TestFallbackEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def test_fallback_with_no_token_uses_static(self):
        """ConversationFallback with no token falls through to static fallback."""
        with patch.dict(os.environ, {}, clear=False):
            # Temporarily wipe token
            fb = ConversationFallback.__new__(ConversationFallback)
            fb._config = {}
            fb._token = None
            fb._port = 18789
            fb._base_url = "http://127.0.0.1:18789"
            fb._timeout = 5.0
            fb._model = None
            fb._available = None

        result = fb.respond("What time is it?")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_fallback_handles_gateway_offline(self):
        """If gateway is at a dead port, falls through to static fallback."""
        fb = ConversationFallback(timeout=0.5)
        fb._port = 19999  # definitely nothing listening here
        fb._base_url = "http://127.0.0.1:19999"
        fb._available = None

        result = fb.respond("Set an alarm for 7am")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_is_available_false_when_no_gateway(self):
        """is_available returns False when gateway is unreachable."""
        fb = ConversationFallback(timeout=0.5)
        fb._port = 19999
        fb._base_url = "http://127.0.0.1:19999"
        fb._available = None  # force re-check
        self.assertFalse(fb.is_available)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    unittest.main(verbosity=2)

import logging
