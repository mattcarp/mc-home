#!/usr/bin/env python3
"""
Claudette Home — Intent Parser
Natural language → Home Assistant service call.

Uses a language model to parse user utterances into structured HA actions.
The entity list is injected into the system prompt dynamically.

Supports multiple backends:
  - anthropic   — direct Anthropic API (requires ANTHROPIC_API_KEY)
  - openai      — OpenAI-compatible API (requires OPENAI_API_KEY)
  - openrouter  — OpenRouter (requires OPENROUTER_API_KEY); can access any model

Usage (standalone):
  python3 intent_parser.py "turn off the living room lights"
  python3 intent_parser.py "it's getting dark, sort the living room"
  python3 intent_parser.py --json-out "set up for dinner"
  python3 intent_parser.py --backend openrouter "I'm going to bed"

Usage (as module):
  from intent_parser import parse_intent
  result = parse_intent("I'm going to bed")
  # result: {"action": "call_service", "domain": "scene", ...}

Environment:
  ANTHROPIC_API_KEY     — required for backend=anthropic
  OPENAI_API_KEY        — required for backend=openai
  OPENROUTER_API_KEY    — required for backend=openrouter
  HA_INTENT_MODEL       — model to use (default depends on backend)
  HA_INTENT_BACKEND     — backend to use (default: auto-detect from available keys)
  HA_INTENT_DEBUG       — set to 1 to print raw API response
"""

import argparse
import json
import os
import sys
from typing import Optional, Union

# ---------------------------------------------------------------------------
# Debug flag
# ---------------------------------------------------------------------------
DEBUG = os.environ.get("HA_INTENT_DEBUG", "0") == "1"

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------
# Models per backend — tuned for low-latency home control (short utterances)
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-3-5-20241022",
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
}

MAX_TOKENS = 512


def _detect_backend() -> str:
    """Auto-detect best available backend from environment keys."""
    explicit = os.environ.get("HA_INTENT_BACKEND", "").lower()
    if explicit in ("anthropic", "openai", "openrouter"):
        return explicit

    # Prefer openrouter (widest model choice, no separate billing)
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"

    raise EnvironmentError(
        "No API key found. Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
    )


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _call_anthropic(
    transcript: str,
    system_prompt: str,
    model: str,
    client=None,
) -> str:
    """Call Anthropic API and return raw response text."""
    try:
        import anthropic as anthropic_sdk
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    if client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        client = anthropic_sdk.Anthropic(api_key=api_key)

    if DEBUG:
        print(f"[DEBUG] Anthropic backend, model={model}, input={transcript!r}", file=sys.stderr)

    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": transcript}],
    )
    return message.content[0].text.strip()


def _call_openai_compatible(
    transcript: str,
    system_prompt: str,
    model: str,
    api_key: str,
    base_url: Optional[str] = None,
    extra_headers: Optional[dict] = None,
) -> str:
    """Call an OpenAI-compatible API and return raw response text."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    kwargs = {
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers

    client = openai.OpenAI(**kwargs)

    if DEBUG:
        print(f"[DEBUG] OpenAI-compat backend, base_url={base_url}, model={model}, input={transcript!r}", file=sys.stderr)

    response = client.chat.completions.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,  # Deterministic for home control
    )
    return response.choices[0].message.content.strip()


def _call_openrouter(transcript: str, system_prompt: str, model: str) -> str:
    """Call OpenRouter API (OpenAI-compatible)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not set")

    return _call_openai_compatible(
        transcript=transcript,
        system_prompt=system_prompt,
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        extra_headers={
            "HTTP-Referer": "https://github.com/mattcarp/mc-home",
            "X-Title": "Claudette Home",
        },
    )


def _call_openai(transcript: str, system_prompt: str, model: str) -> str:
    """Call OpenAI API directly."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set")

    return _call_openai_compatible(
        transcript=transcript,
        system_prompt=system_prompt,
        model=model,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Strip markdown code fences
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Strip markdown code fences if the model added them (it shouldn't, but be safe)."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_intent(
    transcript: str,
    entities: Optional[dict] = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    client=None,  # Anthropic client (legacy/testing)
) -> Union[dict, list]:
    """
    Parse a natural language home command into a structured HA action.

    Args:
        transcript: The user's spoken/typed request (any language — EN, MT, IT, AR)
        entities: HA entity dict (lights, switches, scenes, etc.).
                  Defaults to SAMPLE_ENTITIES if None.
        model: LLM model to use. Defaults to backend-appropriate default.
        backend: 'anthropic', 'openai', or 'openrouter'.
                 Auto-detects from env if not specified.
        client: Optional pre-constructed Anthropic client (for testing/legacy).
                If provided, forces anthropic backend.

    Returns:
        dict or list of dicts with HA action(s).

    Raises:
        ValueError: If the API response cannot be parsed as JSON.
        EnvironmentError: If required API key is not set.
    """
    from ha_context import build_system_prompt, SAMPLE_ENTITIES

    if entities is None:
        entities = SAMPLE_ENTITIES

    system_prompt = build_system_prompt(entities)

    # If a pre-built Anthropic client is passed (e.g. from tests), use it directly
    if client is not None:
        raw = _call_anthropic(
            transcript=transcript,
            system_prompt=system_prompt,
            model=model or DEFAULT_MODELS["anthropic"],
            client=client,
        )
    else:
        selected_backend = backend or _detect_backend()
        selected_model = model or os.environ.get("HA_INTENT_MODEL") or DEFAULT_MODELS[selected_backend]

        if DEBUG:
            print(f"[DEBUG] backend={selected_backend}, model={selected_model}", file=sys.stderr)

        if selected_backend == "anthropic":
            raw = _call_anthropic(transcript, system_prompt, selected_model)
        elif selected_backend == "openrouter":
            raw = _call_openrouter(transcript, system_prompt, selected_model)
        elif selected_backend == "openai":
            raw = _call_openai(transcript, system_prompt, selected_model)
        else:
            raise ValueError(f"Unknown backend: {selected_backend!r}")

    if DEBUG:
        print(f"[DEBUG] Raw response: {raw}", file=sys.stderr)

    raw = _strip_fences(raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Intent parser returned non-JSON: {raw!r}") from e

    return result


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--model", default=None,
        help="LLM model override (default: backend-appropriate)"
    )
    parser.add_argument(
        "--backend", default=None, choices=["anthropic", "openai", "openrouter"],
        help="Backend to use (default: auto-detect from env keys)"
    )
    args = parser.parse_args()

    try:
        result = parse_intent(args.transcript, model=args.model, backend=args.backend)
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json_out:
        print(json.dumps(result, indent=2))
    else:
        print(f"Input:   {args.transcript!r}")
        print(f"Backend: {_detect_backend()}")
        print(f"Action:  {format_action_summary(result)}")
        print(f"JSON:    {json.dumps(result)}")


if __name__ == "__main__":
    main()
