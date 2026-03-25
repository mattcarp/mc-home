#!/usr/bin/env python3
"""
Claudette Home — HA WebSocket Event Emitter
Subscribes to Home Assistant state_changed events and emits JSON lines on stdout.

This is the bridge between the live HA instance and the voice pipeline.
The pipeline reads JSON events from stdin — this emitter produces them.

Architecture:
  ha_event_emitter.py → stdout JSON lines → pipeline.py stdin
      ↓ state_changed events
  pipeline.py → ProactiveAlerts engine → AlertDeliveryRouter → TTS

Composition with wake word bridge (both emit to pipeline stdin):
  # Option A: merge with process substitution
  cat <(python3 wake_word/wake_word_bridge.py) <(python3 ha_event_emitter.py) | python3 pipeline.py

  # Option B: use the --background flag in pipeline.py (planned)
  python3 pipeline.py --ha-events  # spawns emitter as background thread

  # Option C: standalone monitoring (log events to stdout)
  python3 ha_event_emitter.py

Stub mode (no HA required):
  python3 ha_event_emitter.py --stub  # emits simulated events every N seconds

Environment:
  HA_URL      — Home Assistant base URL (default: http://localhost:8123)
  HA_TOKEN    — Long-lived access token (required for live mode)
  HA_EVENT_FILTER — comma-separated entity_id prefixes to include
                    (default: all state_changed events)
  HA_EVENT_DEBOUNCE_MS — minimum ms between events for same entity (default: 1000)

Output format (JSON lines):
  {"type": "state_changed", "entity_id": "binary_sensor.front_door_contact",
   "old_state": "off", "new_state": "on", "timestamp": "2026-03-25T09:00:00Z",
   "attributes": {"friendly_name": "Front Door", "device_class": "door"}}
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_HA_URL = os.environ.get("HA_URL", "http://localhost:8123")
DEFAULT_HA_TOKEN = os.environ.get("HA_TOKEN", "")
DEFAULT_DEBOUNCE_MS = int(os.environ.get("HA_EVENT_DEBOUNCE_MS", "1000"))
DEFAULT_ENTITY_FILTER = os.environ.get("HA_EVENT_FILTER", "")

# Domains we care about for proactive alerts
ALERT_DOMAINS = {
    "binary_sensor",  # doors, windows, motion
    "sensor",         # temperature, humidity
    "light",          # lights left on
    "switch",         # switches left on
    "lock",           # lock state changes
    "cover",          # shutters/blinds
    "climate",        # HVAC state
    "media_player",   # audio zone changes
}


# ---------------------------------------------------------------------------
# Event processing
# ---------------------------------------------------------------------------

class EventProcessor:
    """
    Processes raw HA state_changed events into pipeline-compatible JSON lines.
    Handles debouncing, domain filtering, and format normalisation.
    """

    def __init__(
        self,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
        entity_filter: Optional[List[str]] = None,
        domains: Optional[Set[str]] = None,
    ):
        self.debounce_ms = debounce_ms
        self.entity_filter = entity_filter or []
        self.domains = domains or ALERT_DOMAINS
        self._last_emit: Dict[str, float] = {}  # entity_id → last emit timestamp
        self._stats = {
            "received": 0,
            "emitted": 0,
            "debounced": 0,
            "filtered": 0,
        }

    def process(self, event: dict) -> Optional[dict]:
        """
        Process a raw HA WebSocket state_changed event.

        Args:
            event: Raw event dict from HA WebSocket

        Returns:
            Pipeline-compatible event dict, or None if filtered/debounced
        """
        self._stats["received"] += 1

        data = event.get("data", {})
        entity_id = data.get("entity_id", "")

        if not entity_id:
            return None

        # Domain filter
        domain = entity_id.split(".")[0]
        if domain not in self.domains:
            self._stats["filtered"] += 1
            return None

        # Entity prefix filter (if configured)
        if self.entity_filter:
            if not any(entity_id.startswith(prefix) for prefix in self.entity_filter):
                self._stats["filtered"] += 1
                return None

        # Debounce — skip if same entity emitted too recently
        now_ms = time.time() * 1000
        last = self._last_emit.get(entity_id, 0)
        if (now_ms - last) < self.debounce_ms:
            self._stats["debounced"] += 1
            return None

        # Extract state info
        new_state_obj = data.get("new_state", {})
        old_state_obj = data.get("old_state", {})

        new_state = new_state_obj.get("state", "unknown") if new_state_obj else "unknown"
        old_state = old_state_obj.get("state", "unknown") if old_state_obj else "unknown"

        # Skip if state didn't actually change
        if new_state == old_state:
            self._stats["filtered"] += 1
            return None

        attributes = new_state_obj.get("attributes", {}) if new_state_obj else {}

        # Build pipeline event
        pipeline_event = {
            "type": "state_changed",
            "entity_id": entity_id,
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attributes": {
                "friendly_name": attributes.get("friendly_name", entity_id),
                "device_class": attributes.get("device_class", ""),
                "unit_of_measurement": attributes.get("unit_of_measurement", ""),
            },
        }

        self._last_emit[entity_id] = now_ms
        self._stats["emitted"] += 1
        return pipeline_event

    @property
    def stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# Live WebSocket subscriber
# ---------------------------------------------------------------------------

async def subscribe_and_emit(
    url: str = DEFAULT_HA_URL,
    token: str = DEFAULT_HA_TOKEN,
    processor: Optional[EventProcessor] = None,
    max_reconnect_delay: int = 60,
):
    """
    Connect to HA WebSocket, subscribe to state_changed events,
    and emit processed events as JSON lines on stdout.

    Reconnects automatically on disconnect with exponential backoff.
    """
    try:
        import websockets
    except ImportError:
        raise ImportError("websockets not installed — run: pip install websockets")

    if not token:
        raise EnvironmentError(
            "HA_TOKEN not set. Generate one at HA Profile → Long-Lived Access Tokens."
        )

    if processor is None:
        processor = EventProcessor()

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
    msg_id = 1
    reconnect_delay = 1

    while True:
        try:
            logger.info(f"Connecting to HA WebSocket: {ws_url}")
            async with websockets.connect(ws_url) as ws:
                # Auth handshake
                auth_required = json.loads(await ws.recv())
                if auth_required.get("type") != "auth_required":
                    logger.error(f"Unexpected initial message: {auth_required}")
                    break

                await ws.send(json.dumps({"type": "auth", "access_token": token}))
                auth_result = json.loads(await ws.recv())
                if auth_result.get("type") != "auth_ok":
                    logger.error(f"HA WebSocket auth failed: {auth_result}")
                    # Auth failure is not retryable
                    emit_event({"type": "error", "error": "ha_auth_failed",
                                "message": str(auth_result)})
                    break

                logger.info("HA WebSocket authenticated")
                reconnect_delay = 1  # Reset on successful connect

                # Subscribe to state_changed
                sub_msg = {
                    "id": msg_id,
                    "type": "subscribe_events",
                    "event_type": "state_changed",
                }
                msg_id += 1
                await ws.send(json.dumps(sub_msg))
                sub_ack = json.loads(await ws.recv())
                if not sub_ack.get("success"):
                    logger.error(f"Subscribe failed: {sub_ack}")
                    break

                # Emit connection event
                emit_event({
                    "type": "ha_connected",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "ha_version": auth_result.get("ha_version", "unknown"),
                })

                logger.info("Subscribed to state_changed events — emitting to stdout")

                # Event loop
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") != "event":
                        continue

                    event = msg.get("event", {})
                    processed = processor.process(event)
                    if processed:
                        emit_event(processed)

        except asyncio.CancelledError:
            logger.info("Event emitter cancelled")
            break
        except Exception as e:
            logger.warning(f"WebSocket disconnected: {e}")
            emit_event({
                "type": "ha_disconnected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "reconnect_in": reconnect_delay,
            })

            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)


def emit_event(event: dict):
    """Write a JSON event line to stdout (consumed by pipeline.py)."""
    try:
        print(json.dumps(event), flush=True)
    except BrokenPipeError:
        # Pipeline was killed — exit cleanly
        sys.exit(0)


# ---------------------------------------------------------------------------
# Stub mode — simulated events for testing
# ---------------------------------------------------------------------------

STUB_SCENARIOS = [
    {
        "entity_id": "binary_sensor.front_door_contact",
        "old_state": "off", "new_state": "on",
        "attributes": {"friendly_name": "Front Door", "device_class": "door"},
    },
    {
        "entity_id": "sensor.living_room_temperature",
        "old_state": "21.5", "new_state": "16.2",
        "attributes": {"friendly_name": "Living Room Temperature",
                        "device_class": "temperature", "unit_of_measurement": "°C"},
    },
    {
        "entity_id": "light.kitchen",
        "old_state": "off", "new_state": "on",
        "attributes": {"friendly_name": "Kitchen Light", "device_class": ""},
    },
    {
        "entity_id": "binary_sensor.hallway_motion",
        "old_state": "off", "new_state": "on",
        "attributes": {"friendly_name": "Hallway Motion", "device_class": "motion"},
    },
    {
        "entity_id": "lock.front_door",
        "old_state": "locked", "new_state": "unlocked",
        "attributes": {"friendly_name": "Front Door Lock", "device_class": ""},
    },
]


async def run_stub(interval: float = 5.0, count: Optional[int] = None):
    """
    Emit simulated state_changed events for testing.
    Cycles through STUB_SCENARIOS every `interval` seconds.
    """
    processor = EventProcessor(debounce_ms=0)  # No debounce in stub
    idx = 0
    emitted = 0

    emit_event({
        "type": "ha_connected",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ha_version": "stub",
    })

    while count is None or emitted < count:
        scenario = STUB_SCENARIOS[idx % len(STUB_SCENARIOS)]
        raw_event = {
            "data": {
                "entity_id": scenario["entity_id"],
                "new_state": {
                    "state": scenario["new_state"],
                    "attributes": scenario["attributes"],
                },
                "old_state": {
                    "state": scenario["old_state"],
                    "attributes": scenario["attributes"],
                },
            }
        }

        processed = processor.process(raw_event)
        if processed:
            emit_event(processed)
            emitted += 1

        idx += 1
        if count is None or emitted < count:
            await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Pipeline integration — background thread for embedding in pipeline.py
# ---------------------------------------------------------------------------

class HAEventEmitterThread:
    """
    Runs the HA event emitter in a background thread.
    Calls a callback for each processed event instead of writing to stdout.

    Usage in pipeline.py:
        def on_ha_event(event):
            # Inject into event queue or process directly
            alert_integration.on_ha_event(json.dumps(event))

        emitter = HAEventEmitterThread(callback=on_ha_event)
        emitter.start()
        # ... pipeline runs ...
        emitter.stop()
    """

    def __init__(
        self,
        callback,
        url: str = DEFAULT_HA_URL,
        token: str = DEFAULT_HA_TOKEN,
        processor: Optional[EventProcessor] = None,
    ):
        self.callback = callback
        self.url = url
        self.token = token
        self.processor = processor or EventProcessor()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = None
        self._running = False

    def start(self):
        """Start the emitter in a background thread."""
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ha-event-emitter")
        self._thread.start()
        logger.info("HA event emitter thread started")

    def stop(self):
        """Stop the emitter."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HA event emitter thread stopped")

    def _run(self):
        """Thread entry point — runs the async event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._subscribe())
        except Exception as e:
            logger.error(f"HA event emitter thread error: {e}")
        finally:
            self._loop.close()

    async def _subscribe(self):
        """Subscribe and route events through the callback."""
        try:
            import websockets
        except ImportError:
            logger.error("websockets not installed — HA event emitter disabled")
            return

        if not self.token:
            logger.warning("HA_TOKEN not set — HA event emitter disabled")
            return

        ws_url = self.url.replace("http://", "ws://").replace("https://", "wss://") + "/api/websocket"
        msg_id = 1
        reconnect_delay = 1

        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    # Auth
                    auth_required = json.loads(await ws.recv())
                    if auth_required.get("type") != "auth_required":
                        break

                    await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
                    auth_result = json.loads(await ws.recv())
                    if auth_result.get("type") != "auth_ok":
                        logger.error(f"HA auth failed: {auth_result}")
                        break

                    # Subscribe
                    await ws.send(json.dumps({
                        "id": msg_id,
                        "type": "subscribe_events",
                        "event_type": "state_changed",
                    }))
                    msg_id += 1
                    sub_ack = json.loads(await ws.recv())
                    if not sub_ack.get("success"):
                        break

                    reconnect_delay = 1
                    self.callback({
                        "type": "ha_connected",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                    async for raw in ws:
                        if not self._running:
                            break
                        msg = json.loads(raw)
                        if msg.get("type") != "event":
                            continue
                        processed = self.processor.process(msg.get("event", {}))
                        if processed:
                            self.callback(processed)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning(f"HA WebSocket error: {e}, reconnecting in {reconnect_delay}s")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [ha-events] %(message)s",
        stream=sys.stderr,  # Logs to stderr, events to stdout
    )

    parser = argparse.ArgumentParser(
        description="Claudette Home — HA WebSocket Event Emitter"
    )
    parser.add_argument("--stub", action="store_true",
                        help="Emit simulated events (no HA required)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Stub mode: seconds between events (default: 5)")
    parser.add_argument("--count", type=int, default=None,
                        help="Stub mode: emit N events then exit (default: infinite)")
    parser.add_argument("--debounce-ms", type=int, default=DEFAULT_DEBOUNCE_MS,
                        help=f"Debounce window in ms (default: {DEFAULT_DEBOUNCE_MS})")
    parser.add_argument("--filter", default=DEFAULT_ENTITY_FILTER,
                        help="Comma-separated entity_id prefixes to include")
    args = parser.parse_args()

    entity_filter = [f.strip() for f in args.filter.split(",") if f.strip()] if args.filter else []
    processor = EventProcessor(
        debounce_ms=args.debounce_ms,
        entity_filter=entity_filter,
    )

    if args.stub:
        asyncio.run(run_stub(interval=args.interval, count=args.count))
    else:
        asyncio.run(subscribe_and_emit(processor=processor))


if __name__ == "__main__":
    main()
