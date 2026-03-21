#!/usr/bin/env python3
"""
Claudette Home — Alert Delivery Router

Routes proactive alerts from the alert engine to their appropriate output:
- HIGH priority (security/safety) → immediate TTS via pipeline stdout
- LOW priority (comfort) → batched, delivered at next conversation turn
- ALL alerts → logged to alert_log for dashboard display

Delivery modes:
- "immediate": emit JSON event to TTS responder right now
- "batched": hold until next_conversation_batch() is called
- "silent": log only (quiet hours, rate-limited duplicates)

Integrates with:
- brain/proactive_alerts.py (source of alerts)
- voice/tts_responder.py (consumes JSON events from stdout)
- dashboard/ (reads alert_log for UI display)

Part of EPIC 1 (#1) and Issue #8.
"""

import datetime
import json
import logging
import os
import time
from typing import Optional, Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [alert-delivery] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Quiet hours: suppress low-priority TTS (Malta timezone approximation in UTC)
# Malta is UTC+1 (winter) / UTC+2 (summer). 23:00 local ≈ 21:00-22:00 UTC
# 08:00 local ≈ 06:00-07:00 UTC. We use a conservative UTC window.
DEFAULT_QUIET_START_UTC = 22  # 22:00 UTC ≈ 23:00-00:00 Malta
DEFAULT_QUIET_END_UTC = 7    # 07:00 UTC ≈ 08:00-09:00 Malta

# Rate limiting: don't re-alert the same entity within this window
RATE_LIMIT_MINUTES = 15

# Maximum alerts in a single batch delivery (don't overwhelm with 20 things)
MAX_BATCH_DELIVERY = 5

# Maximum log entries kept in memory
MAX_LOG_ENTRIES = 200


class AlertDeliveryRouter:
    """
    Routes alerts to the correct delivery channel based on priority,
    time of day, and rate limiting.
    """

    def __init__(
        self,
        quiet_start_utc: int = DEFAULT_QUIET_START_UTC,
        quiet_end_utc: int = DEFAULT_QUIET_END_UTC,
        rate_limit_min: int = RATE_LIMIT_MINUTES,
        output_fn: Optional[Callable[[str], None]] = None,
    ):
        self.quiet_start = quiet_start_utc
        self.quiet_end = quiet_end_utc
        self.rate_limit_sec = rate_limit_min * 60

        # Output function for TTS events — default: print to stdout (pipe to tts_responder)
        self._output_fn = output_fn or self._default_output

        # Batched low-priority alerts waiting for next conversation
        self._batch: list[dict] = []

        # Rate limiting: entity_id → last_alert_timestamp
        self._last_alerted: dict[str, float] = {}

        # Full alert log for dashboard (capped at MAX_LOG_ENTRIES)
        self._log: list[dict] = []

        # Stats
        self._stats = {
            "total_received": 0,
            "immediate_delivered": 0,
            "batched": 0,
            "rate_limited": 0,
            "quiet_suppressed": 0,
        }

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route_alert(self, alert: dict, now: Optional[float] = None) -> str:
        """
        Route a single alert to the appropriate delivery channel.

        Returns the delivery mode used: "immediate", "batched", "silent",
        or "rate_limited".
        """
        now = now or time.time()
        entity = alert.get("entity", "unknown")
        priority = alert.get("priority", "low")
        message = alert.get("message", "")

        self._stats["total_received"] += 1

        # Always log, regardless of delivery mode
        log_entry = {
            **alert,
            "received_at": datetime.datetime.fromtimestamp(
                now, tz=datetime.timezone.utc
            ).isoformat(),
            "delivery": "pending",
        }

        # Check rate limiting
        if self._is_rate_limited(entity, now):
            log_entry["delivery"] = "rate_limited"
            self._append_log(log_entry)
            self._stats["rate_limited"] += 1
            logger.debug(f"Rate-limited: {entity}")
            return "rate_limited"

        # Record this alert for rate limiting
        self._last_alerted[entity] = now

        is_quiet = self._is_quiet_hours(now)

        if priority == "high":
            # High priority always gets immediate delivery, even during quiet hours
            # (security alerts shouldn't wait)
            self._deliver_immediate(message, alert)
            log_entry["delivery"] = "immediate"
            self._append_log(log_entry)
            self._stats["immediate_delivered"] += 1
            return "immediate"

        else:
            # Low priority
            if is_quiet:
                # During quiet hours: log only, no TTS
                log_entry["delivery"] = "quiet_suppressed"
                self._append_log(log_entry)
                self._stats["quiet_suppressed"] += 1
                logger.info(f"Quiet hours — suppressed: {message[:60]}")
                return "silent"
            else:
                # Normal hours: batch for next conversation
                self._batch.append(alert)
                log_entry["delivery"] = "batched"
                self._append_log(log_entry)
                self._stats["batched"] += 1
                logger.info(f"Batched: {message[:60]}")
                return "batched"

    def route_alerts(self, alerts: list[dict], now: Optional[float] = None) -> list[str]:
        """Route multiple alerts. Returns list of delivery modes."""
        return [self.route_alert(a, now=now) for a in alerts]

    # ------------------------------------------------------------------
    # Batch delivery (called when Claudette is already talking)
    # ------------------------------------------------------------------

    def next_conversation_batch(self) -> list[dict]:
        """
        Get batched low-priority alerts for delivery during a conversation.
        Returns up to MAX_BATCH_DELIVERY alerts and clears the batch.
        """
        to_deliver = self._batch[:MAX_BATCH_DELIVERY]
        self._batch = self._batch[MAX_BATCH_DELIVERY:]
        return to_deliver

    def deliver_batch_now(self) -> int:
        """
        Deliver all batched alerts via TTS right now.
        Used when a conversation turn starts and there are pending items.
        Returns number of alerts delivered.
        """
        batch = self.next_conversation_batch()
        if not batch:
            return 0

        # Combine messages into a natural intro
        if len(batch) == 1:
            intro = "By the way — " + batch[0]["message"]
        else:
            intro = f"A few things while you're here. "
            intro += " ".join(a["message"] for a in batch)

        self._deliver_immediate(intro, {"type": "batch_delivery", "count": len(batch)})
        return len(batch)

    @property
    def pending_batch_count(self) -> int:
        """Number of alerts waiting in the batch queue."""
        return len(self._batch)

    # ------------------------------------------------------------------
    # Immediate delivery (TTS pipeline)
    # ------------------------------------------------------------------

    def _deliver_immediate(self, message: str, alert: dict) -> None:
        """Emit a pipeline_response event for the TTS responder to speak."""
        event = {
            "type": "pipeline_response",
            "text": message,
            "source": "proactive_alert",
            "priority": alert.get("priority", "low"),
            "entity": alert.get("entity"),
        }
        self._output_fn(json.dumps(event))
        logger.info(f"Delivered: {message[:80]}")

    @staticmethod
    def _default_output(event_json: str) -> None:
        """Default output: print JSON to stdout for tts_responder to consume."""
        print(event_json, flush=True)

    # ------------------------------------------------------------------
    # Quiet hours
    # ------------------------------------------------------------------

    def _is_quiet_hours(self, now: float) -> bool:
        """Check if current time is within quiet hours (UTC)."""
        hour = datetime.datetime.fromtimestamp(
            now, tz=datetime.timezone.utc
        ).hour
        if self.quiet_start > self.quiet_end:
            # Wraps midnight: e.g. 22:00 → 07:00
            return hour >= self.quiet_start or hour < self.quiet_end
        else:
            return self.quiet_start <= hour < self.quiet_end

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _is_rate_limited(self, entity_id: str, now: float) -> bool:
        """Check if this entity was alerted too recently."""
        last = self._last_alerted.get(entity_id)
        if last is None:
            return False
        return (now - last) < self.rate_limit_sec

    def clear_rate_limit(self, entity_id: str) -> None:
        """Manually clear rate limit for an entity (e.g. after state change)."""
        self._last_alerted.pop(entity_id, None)

    # ------------------------------------------------------------------
    # Alert log (for dashboard)
    # ------------------------------------------------------------------

    def _append_log(self, entry: dict) -> None:
        """Add to the alert log, capping at MAX_LOG_ENTRIES."""
        self._log.append(entry)
        if len(self._log) > MAX_LOG_ENTRIES:
            self._log = self._log[-MAX_LOG_ENTRIES:]

    def get_log(self, limit: int = 50) -> list[dict]:
        """Get recent alert log entries (newest first) for dashboard."""
        return list(reversed(self._log[-limit:]))

    def get_log_by_priority(self, priority: str, limit: int = 20) -> list[dict]:
        """Get log entries filtered by priority."""
        filtered = [e for e in reversed(self._log) if e.get("priority") == priority]
        return filtered[:limit]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """Return delivery router status for dashboard/monitoring."""
        return {
            "pending_batch": len(self._batch),
            "log_size": len(self._log),
            "rate_limited_entities": len(self._last_alerted),
            "stats": dict(self._stats),
            "quiet_hours": f"{self.quiet_start:02d}:00-{self.quiet_end:02d}:00 UTC",
        }

    # ------------------------------------------------------------------
    # Integration: ProactiveAlerts → AlertDeliveryRouter
    # ------------------------------------------------------------------

    def process_from_engine(self, engine, now: Optional[float] = None) -> list[str]:
        """
        Pull pending alerts from a ProactiveAlerts engine and route them.
        This is the main integration point.

        Usage:
            engine = ProactiveAlerts()
            router = AlertDeliveryRouter()
            # ... engine processes HA events ...
            router.process_from_engine(engine)
        """
        alerts = engine.get_pending_alerts()
        if not alerts:
            return []
        return self.route_alerts(alerts, now=now)


# ---------------------------------------------------------------------------
# Integration: Pipeline event loop
# ---------------------------------------------------------------------------

class AlertPipelineIntegration:
    """
    Connects the ProactiveAlerts engine + AlertDeliveryRouter into the
    voice pipeline event loop.

    Reads HA WebSocket state_changed events, feeds them to the alert engine,
    and routes resulting alerts through the delivery router.

    Also hooks into conversation turns to deliver batched low-priority alerts.
    """

    def __init__(
        self,
        engine=None,
        router=None,
        output_fn: Optional[Callable[[str], None]] = None,
    ):
        from brain.proactive_alerts import ProactiveAlerts

        self.engine = engine or ProactiveAlerts()
        self.router = router or AlertDeliveryRouter(output_fn=output_fn)

    def on_ha_event(self, event_json: str, eval_time: Optional[float] = None) -> list[str]:
        """
        Called when an HA state_changed event arrives (from WebSocket).
        Feeds it to the alert engine and routes any resulting alerts.
        """
        self.engine.process_event(event_json, eval_time=eval_time)
        return self.router.process_from_engine(self.engine, now=eval_time)

    def on_conversation_start(self) -> int:
        """
        Called when a new conversation turn begins (wake word detected).
        Delivers any batched low-priority alerts via TTS.
        Returns number of alerts delivered.
        """
        return self.router.deliver_batch_now()

    def status(self) -> dict:
        """Combined status from engine + router."""
        return {
            "engine": self.engine.status(),
            "delivery": self.router.status(),
        }


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Capture output for testing
    delivered = []

    def capture(event_json):
        delivered.append(json.loads(event_json))
        print(f"  → {event_json}")

    router = AlertDeliveryRouter(output_fn=capture)
    now = time.time()

    print("=== Alert Delivery Router — Smoke Test ===\n")

    # Test 1: High-priority alert → immediate
    print("1. High-priority (door open) → should deliver immediately:")
    mode = router.route_alert({
        "entity": "binary_sensor.door_front",
        "message": "The front door has been open for 35 minutes.",
        "priority": "high",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, now=now)
    assert mode == "immediate", f"Expected immediate, got {mode}"
    print(f"   Mode: {mode} ✅\n")

    # Test 2: Low-priority alert → batched
    print("2. Low-priority (light) → should batch:")
    mode = router.route_alert({
        "entity": "light.kitchen",
        "message": "Kitchen light has been on for an hour with no motion.",
        "priority": "low",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, now=now)
    assert mode == "batched", f"Expected batched, got {mode}"
    print(f"   Mode: {mode} ✅\n")

    # Test 3: Rate limiting
    print("3. Same entity again → should be rate-limited:")
    mode = router.route_alert({
        "entity": "binary_sensor.door_front",
        "message": "The front door is still open.",
        "priority": "high",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, now=now + 60)  # 1 min later (within 15-min window)
    assert mode == "rate_limited", f"Expected rate_limited, got {mode}"
    print(f"   Mode: {mode} ✅\n")

    # Test 4: Batch delivery
    print("4. Deliver batch → should combine into TTS:")
    count = router.deliver_batch_now()
    assert count == 1, f"Expected 1 batched, got {count}"
    print(f"   Delivered {count} batched alert ✅\n")

    # Test 5: Status
    print("5. Router status:")
    s = router.status()
    print(f"   {json.dumps(s, indent=2)}")
    assert s["stats"]["total_received"] == 3
    assert s["stats"]["immediate_delivered"] == 1
    assert s["stats"]["batched"] == 1
    assert s["stats"]["rate_limited"] == 1
    print("   ✅\n")

    # Test 6: Log
    print("6. Alert log (newest first):")
    log = router.get_log(limit=5)
    for entry in log:
        print(f"   [{entry['delivery']}] {entry.get('message', '')[:60]}")
    assert len(log) == 3
    print("   ✅\n")

    print(f"All smoke tests passed! {len(delivered)} events delivered to TTS.")
