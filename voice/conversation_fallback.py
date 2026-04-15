#!/usr/bin/env python3
"""
Claudette Home — Conversational Fallback
Routes non-device-control queries to OpenClaw's local chat completions endpoint.

When someone says something to Claudette that isn't a home control command
(e.g. "What's the weather like?", "Set a 10-minute timer", "Tell me a joke"),
the intent parser returns action_type="unknown". This module handles those.

Architecture:
    pipeline.py → process_intent() → action_type="unknown"
        → ConversationFallback.respond(transcript)
        → POST /v1/chat/completions (OpenClaw local gateway, port 18789)
        → natural language response → TTS

Endpoint:
    POST http://127.0.0.1:18789/v1/chat/completions
    Authorization: Bearer <OPENCLAW_GATEWAY_TOKEN>

The gateway token is read from ~/.openclaw/openclaw.json automatically.

Environment:
    OPENCLAW_GATEWAY_TOKEN  — override the token (optional, auto-detected)
    OPENCLAW_GATEWAY_PORT   — override the port (optional, default: 18789)
    HA_FALLBACK_MODEL       — model override (default: auto from gateway)
    HA_FALLBACK_DEBUG       — set to 1 for verbose logging

Usage (standalone):
    python3 voice/conversation_fallback.py "What time is it?"
    python3 voice/conversation_fallback.py "Set a timer for 5 minutes"
    python3 voice/conversation_fallback.py --debug "Tell me a joke"

Usage (as module):
    from conversation_fallback import ConversationFallback
    fb = ConversationFallback()
    text = fb.respond("What's the weather?")
    # text: "I don't have live weather data right now, but I can check..."
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("HA_FALLBACK_DEBUG", "0") == "1"
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenClaw gateway config (auto-detected from ~/.openclaw/openclaw.json)
# ---------------------------------------------------------------------------
_OPENCLAW_CONFIG_PATH = Path.home() / ".openclaw" / "openclaw.json"
_DEFAULT_PORT = 18789

# Claudette's home assistant persona prompt
_SYSTEM_PROMPT = """You are Claudette, a smart home AI assistant embedded in a voice interface panel in a home in Malta. 
You have a warm, slightly French personality — occasionally drop a "mon cher" or "c'est la vie". 
Be concise for voice output: 1-3 sentences max. No markdown, no lists, no asterisks. Plain speech only.
You have context about the home (Xagħra, Gozo, Malta) but NOT live data like weather or current time unless told.
If asked about home devices, say you didn't catch that as a device command and they can try again.
If asked about live data you don't have, be honest but helpful."""


def _load_gateway_config() -> dict:
    """Load OpenClaw gateway config from ~/.openclaw/openclaw.json."""
    if not _OPENCLAW_CONFIG_PATH.exists():
        return {}
    try:
        with open(_OPENCLAW_CONFIG_PATH) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not read openclaw.json: {e}")
        return {}


def _get_gateway_token(config: dict) -> Optional[str]:
    """Extract gateway token from config or environment."""
    # Env override takes priority
    env_token = os.environ.get("OPENCLAW_GATEWAY_TOKEN")
    if env_token:
        return env_token
    # Try auth.token path
    token = config.get("gateway", {}).get("auth", {}).get("token")
    if token:
        return token
    return None


def _get_gateway_port(config: dict) -> int:
    """Extract gateway port from config or environment."""
    env_port = os.environ.get("OPENCLAW_GATEWAY_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    return config.get("gateway", {}).get("port", _DEFAULT_PORT)


# ---------------------------------------------------------------------------
# ConversationFallback
# ---------------------------------------------------------------------------
class ConversationFallback:
    """
    Routes non-HA queries to OpenClaw's local chat completions endpoint.
    Produces short, voice-ready responses for the TTS pipeline.
    """

    def __init__(
        self,
        timeout: float = 8.0,
        model: Optional[str] = None,
    ):
        self._config = _load_gateway_config()
        self._token = _get_gateway_token(self._config)
        self._port = _get_gateway_port(self._config)
        self._base_url = f"http://127.0.0.1:{self._port}"
        self._timeout = timeout
        self._model = model or os.environ.get("HA_FALLBACK_MODEL")
        self._available: Optional[bool] = None  # cached health check result

    @property
    def is_available(self) -> bool:
        """Check if OpenClaw gateway is reachable (cached after first check)."""
        if self._available is None:
            self._available = self._check_health()
        return self._available

    def _check_health(self) -> bool:
        """Ping the gateway health endpoint."""
        try:
            req = urllib.request.Request(
                f"{self._base_url}/v1/models",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception as e:
            log.debug(f"Gateway health check failed: {e}")
            return False

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def respond(self, transcript: str) -> str:
        """
        Generate a conversational response for a non-HA query.

        Args:
            transcript: The user's spoken text (already transcribed).

        Returns:
            A short, voice-ready response string.
            Falls back to a polite error message if gateway unavailable.
        """
        if not self._token:
            log.warning("No OpenClaw gateway token — using static fallback")
            return self._static_fallback(transcript)

        payload = {
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            "max_tokens": 150,
            "stream": False,
        }
        if self._model:
            payload["model"] = self._model

        if DEBUG:
            log.debug(f"Fallback request: {transcript!r}")

        t0 = time.monotonic()
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self._base_url}/v1/chat/completions",
                data=data,
                headers=self._headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read())

            elapsed = time.monotonic() - t0
            text = body["choices"][0]["message"]["content"].strip()

            if DEBUG:
                log.debug(f"Fallback response ({elapsed:.2f}s): {text!r}")

            return text

        except urllib.error.HTTPError as e:
            log.warning(f"Gateway HTTP error {e.code}: {e.reason}")
            return self._static_fallback(transcript)
        except urllib.error.URLError as e:
            log.warning(f"Gateway unreachable: {e.reason}")
            return self._static_fallback(transcript)
        except (KeyError, json.JSONDecodeError) as e:
            log.warning(f"Unexpected gateway response: {e}")
            return self._static_fallback(transcript)
        except Exception as e:
            log.warning(f"Fallback error: {e}")
            return self._static_fallback(transcript)

    @staticmethod
    def _static_fallback(transcript: str) -> str:
        """
        Last-resort fallback when gateway is unreachable.
        Returns a polite, context-aware response without LLM.
        """
        # Simple keyword matching for common non-HA queries
        lower = transcript.lower()
        if any(w in lower for w in ("time", "clock", "hour")):
            return "I'm not connected to get the time right now. Check your phone, mon cher."
        if any(w in lower for w in ("weather", "rain", "temperature outside", "forecast")):
            return "I can't check the weather right now, but Malta's usually lovely."
        if any(w in lower for w in ("timer", "alarm", "remind")):
            return "I can't set timers just yet — that's coming soon."
        if any(w in lower for w in ("joke", "funny", "laugh")):
            return "My sense of humour is offline, apparently. Try me later."
        if any(w in lower for w in ("music", "play", "song", "radio")) and "stop" not in lower:
            return "Try saying the room name too — like 'play jazz in the living room'."
        return "I didn't quite catch that as a home command. Could you try again?"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Claudette Home — Conversational Fallback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("transcript", help="Text to respond to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--timeout", type=float, default=8.0, help="Request timeout (s)")
    args = parser.parse_args()

    if args.debug:
        os.environ["HA_FALLBACK_DEBUG"] = "1"
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    fb = ConversationFallback(timeout=args.timeout, model=args.model)

    if not fb._token:
        print("⚠️  No gateway token detected — will use static fallback only", file=sys.stderr)
    elif args.debug:
        avail = fb.is_available
        print(f"Gateway available: {avail} (port {fb._port})", file=sys.stderr)

    result = fb.respond(args.transcript)
    print(result)


if __name__ == "__main__":
    main()
