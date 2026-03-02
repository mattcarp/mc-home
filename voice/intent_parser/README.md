# Intent Parser — Natural Language → HA Service Call

Converts Claudette's spoken/typed home commands into structured Home Assistant service calls.

## Architecture

```
Wake word detected
  → STT transcript (issue #10)
  → intent_parser.parse_intent(transcript)
  → {"action": "call_service", "domain": "light", "service": "turn_on", ...}
  → ha_bridge.call_service(...) (issue #6)
  → HA executes
```

## How It Works

1. `ha_context.py` builds a system prompt from the entity list (lights, scenes, locks, etc.)
2. `intent_parser.py` sends the transcript + context to Claude (haiku-3-5 by default — fast + cheap)
3. Claude returns a JSON action object
4. The caller executes it via the HA bridge

**No hardcoded command parsing.** Claude handles all the NLU — "sort the living room", "I'm heading out", "it's getting dark" all work naturally.

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key

# Test a command
python3 intent_parser.py "turn off the lights"
# → Input:  'turn off the lights'
# → Action: light.turn_off(light.all)
# → JSON:   {"action": "call_service", "domain": "light", ...}

# JSON output only
python3 intent_parser.py --json-out "I'm going to bed"
```

## Run Tests

```bash
# Dry run (no API key needed)
python3 test_intent_parser.py

# Live API tests
ANTHROPIC_API_KEY=... python3 test_intent_parser.py --live
```

## Response Format

### Single action
```json
{
  "action": "call_service",
  "domain": "light",
  "service": "turn_off",
  "entity_id": "light.all",
  "params": {}
}
```

### With parameters
```json
{
  "action": "call_service",
  "domain": "light",
  "service": "turn_on",
  "entity_id": "light.living_room",
  "params": {
    "brightness_pct": 40,
    "color_temp": "warm"
  }
}
```

### Multi-action (list)
```json
[
  {"action": "call_service", "domain": "scene", "service": "activate", "entity_id": "scene.dinner", "params": {}},
  {"action": "call_service", "domain": "lock", "service": "lock", "entity_id": "lock.front_door", "params": {}}
]
```

### Clarification needed
```json
{
  "action": "clarify",
  "question": "Which room did you mean?"
}
```

### Status query
```json
{
  "action": "query",
  "entity_id": "sensor.living_room_temperature",
  "question": "What is the current temperature?"
}
```

## Live Entities

Currently uses `SAMPLE_ENTITIES` from `ha_context.py` (Xagħra prototype layout).

Once HA is installed (issue #6), swap with:
```python
from ha_bridge import get_entities
result = parse_intent(transcript, entities=get_entities())
```

## Model

Default: `claude-haiku-3-5-20241022` — chosen for low latency (< 500ms) and cost (~$0.0002 per call).

Override: `HA_INTENT_MODEL=claude-sonnet-4-5` for more complex/ambiguous commands.

## Integration Point

```python
# In the STT pipeline (issue #10), after getting a transcript:
from intent_parser import parse_intent, format_action_summary

result = parse_intent(transcript)

if isinstance(result, dict) and result.get("action") == "clarify":
    # TTS: ask the question back to the user
    speak(result["question"])
elif isinstance(result, dict) and result.get("action") == "query":
    # Fetch value from HA and speak it
    value = ha_bridge.get_state(result["entity_id"])
    speak(f"It's {value}")
else:
    # Execute the action(s)
    for action in (result if isinstance(result, list) else [result]):
        ha_bridge.call_service(action)
```
