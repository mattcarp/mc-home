#!/usr/bin/env python3
"""
Claudette Home — Intent Parser
Natural language → Home Assistant service call.

Uses Claude (via Anthropic API) to parse user utterances into structured
HA actions. The entity list is injected into the system prompt dynamically.

Usage (standalone):
  python3 intent_parser.py "turn off the living room lights"
  python3 intent_parser.py "it's getting dark, sort the living room"
  python3 intent_parser.py --json-out "set up for dinner"

Usage (as module):
  from intent_parser import parse_intent
  result = parse_intent("I'm going to bed")
  # result: {"action": "call_service", "domain": "scene", ...}

Environment:
  ANTHROPIC_API_KEY   — required
  HA_INTENT_MODEL     — Claude model to use (default: claude-haiku-3-5)
  HA_INTENT_DEBUG     — set to 1 to print raw API response
"""

import argparse
import json
import os
import sys
from typing import Optional, Union

try:
    import anthropic
except ImportError:
    print("anthropic package not installed. Run: pip install anthropic", file=sys.stderr)
    sys.exit(1)

from ha_context import build_system_prompt, SAMPLE_ENTITIES


# Default to haiku for low latency / low cost — home commands are short.
DEFAULT_MODEL = os.environ.get("HA_INTENT_MODEL", "claude-haiku-3-5-20241022")
MAX_TOKENS = 512
DEBUG = os.environ.get("HA_INTENT_DEBUG", "0") == "1"


def parse_intent(
    transcript: str,
    entities: Optional[dict] = None,
    model: str = DEFAULT_MODEL,
    client: Optional[anthropic.Anthropic] = None,
) -> Union[dict, list]:
    """
    Parse a natural language home command into a structured HA action.

    Args:
        transcript: The user's spoken/typed request.
        entities: HA entity dict (lights, switches, scenes, etc.).
                  Defaults to SAMPLE_ENTITIES if None.
        model: Claude model to use.
        client: Optional pre-constructed Anthropic client.

    Returns:
        dict or list of dicts with HA action(s).

    Raises:
        ValueError: If the API response cannot be parsed as JSON.
        anthropic.APIError: On API failure.
    """
    if entities is None:
        entities = SAMPLE_ENTITIES

    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api_key)

    system_prompt = build_system_prompt(entities)

    if DEBUG:
        print(f"[DEBUG] Sending to {model}: {transcript!r}", file=sys.stderr)

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[
            {"role": "user", "content": transcript}
        ],
    )

    raw = message.content[0].text.strip()

    if DEBUG:
        print(f"[DEBUG] Raw response: {raw}", file=sys.stderr)

    # Strip markdown code fences if model added them (it shouldn't, but be safe)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Intent parser returned non-JSON: {raw!r}") from e

    return result


def format_action_summary(action: Union[dict, list]) -> str:
    """Human-readable summary of a parsed action (for logging/display)."""
    if isinstance(action, list):
        return " + ".join(format_action_summary(a) for a in action)

    a = action.get("action", "unknown")
    if a == "call_service":
        domain = action.get("domain", "?")
        service = action.get("service", "?")
        entity = action.get("entity_id", "?")
        params = action.get("params", {})
        param_str = f" ({params})" if params else ""
        return f"{domain}.{service}({entity}){param_str}"
    elif a == "clarify":
        return f"CLARIFY: {action.get('question', '?')}"
    elif a == "query":
        return f"QUERY: {action.get('entity_id', '?')} — {action.get('question', '')}"
    else:
        return f"UNKNOWN_ACTION: {action}"


def main():
    parser = argparse.ArgumentParser(
        description="Claudette Home — parse natural language into HA action"
    )
    parser.add_argument("transcript", help="The user's spoken/typed home command")
    parser.add_argument(
        "--json-out", action="store_true",
        help="Output raw JSON (default: human-readable summary)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Claude model (default: {DEFAULT_MODEL})"
    )
    args = parser.parse_args()

    try:
        result = parse_intent(args.transcript, model=args.model)
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        print(f"Input:  {args.transcript!r}")
        print(f"Action: {format_action_summary(result)}")
        print(f"JSON:   {json.dumps(result)}")


if __name__ == "__main__":
    main()
