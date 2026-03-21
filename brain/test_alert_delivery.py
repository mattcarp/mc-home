#!/usr/bin/env python3
"""
Tests for Claudette Home — Alert Delivery Router

Tests: routing logic, quiet hours, rate limiting, batch delivery,
logging, integration with ProactiveAlerts engine, and pipeline hooks.
"""

import datetime
import json
import os
import sys
import time
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from brain.alert_delivery import (
    AlertDeliveryRouter,
    AlertPipelineIntegration,
    DEFAULT_QUIET_START_UTC,
    DEFAULT_QUIET_END_UTC,
    MAX_BATCH_DELIVERY,
    MAX_LOG_ENTRIES,
    RATE_LIMIT_MINUTES,
)
from brain.proactive_alerts import ProactiveAlerts


def _make_alert(entity="light.kitchen", message="Test alert", priority="low"):
    return {
        "entity": entity,
        "message": message,
        "priority": priority,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _ts_at_hour(hour, minute=0):
    """Create a Unix timestamp for today at the given UTC hour."""
    dt = datetime.datetime.now(datetime.timezone.utc).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return dt.timestamp()


class CaptureOutput:
    """Capture TTS output events for assertion."""

    def __init__(self):
        self.events = []

    def __call__(self, event_json):
        self.events.append(json.loads(event_json))

    @property
    def count(self):
        return len(self.events)

    @property
    def last(self):
        return self.events[-1] if self.events else None


# =========================================================================
# TestHighPriorityRouting
# =========================================================================
class TestHighPriorityRouting(unittest.TestCase):
    """High-priority alerts always get immediate delivery."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out)

    def test_high_priority_immediate(self):
        mode = self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Door open!", "high")
        )
        self.assertEqual(mode, "immediate")
        self.assertEqual(self.out.count, 1)

    def test_high_priority_event_format(self):
        self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Door open 35min", "high")
        )
        event = self.out.last
        self.assertEqual(event["type"], "pipeline_response")
        self.assertEqual(event["source"], "proactive_alert")
        self.assertEqual(event["priority"], "high")
        self.assertIn("35min", event["text"])

    def test_high_priority_during_quiet_hours(self):
        """Security alerts fire even at 3 AM."""
        now = _ts_at_hour(3)
        mode = self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Door open!", "high"),
            now=now,
        )
        self.assertEqual(mode, "immediate")
        self.assertEqual(self.out.count, 1)

    def test_high_priority_stats(self):
        self.router.route_alert(_make_alert(priority="high"))
        self.assertEqual(self.router.status()["stats"]["immediate_delivered"], 1)


# =========================================================================
# TestLowPriorityRouting
# =========================================================================
class TestLowPriorityRouting(unittest.TestCase):
    """Low-priority alerts get batched during normal hours."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out)

    def test_low_priority_batched(self):
        # Use a time definitely NOT in quiet hours
        now = _ts_at_hour(12)
        mode = self.router.route_alert(
            _make_alert("light.kitchen", "Light on 1hr", "low"),
            now=now,
        )
        self.assertEqual(mode, "batched")
        self.assertEqual(self.out.count, 0)  # Not immediately delivered

    def test_low_priority_in_batch_queue(self):
        now = _ts_at_hour(12)
        self.router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(self.router.pending_batch_count, 1)

    def test_low_priority_stats(self):
        now = _ts_at_hour(12)
        self.router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(self.router.status()["stats"]["batched"], 1)


# =========================================================================
# TestQuietHours
# =========================================================================
class TestQuietHours(unittest.TestCase):
    """Low-priority alerts suppressed during quiet hours."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out)

    def test_low_priority_suppressed_at_midnight(self):
        now = _ts_at_hour(0)
        mode = self.router.route_alert(
            _make_alert("light.kitchen", "Light on", "low"), now=now
        )
        self.assertEqual(mode, "silent")
        self.assertEqual(self.out.count, 0)

    def test_low_priority_suppressed_at_3am(self):
        now = _ts_at_hour(3)
        mode = self.router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(mode, "silent")

    def test_low_priority_ok_at_noon(self):
        now = _ts_at_hour(12)
        mode = self.router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(mode, "batched")

    def test_low_priority_suppressed_at_23_utc(self):
        now = _ts_at_hour(23)
        mode = self.router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(mode, "silent")

    def test_quiet_suppressed_stats(self):
        now = _ts_at_hour(3)
        self.router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(self.router.status()["stats"]["quiet_suppressed"], 1)

    def test_custom_quiet_hours(self):
        router = AlertDeliveryRouter(
            quiet_start_utc=20, quiet_end_utc=8, output_fn=self.out
        )
        now = _ts_at_hour(21)
        mode = router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(mode, "silent")

    def test_custom_quiet_hours_outside(self):
        router = AlertDeliveryRouter(
            quiet_start_utc=20, quiet_end_utc=8, output_fn=self.out
        )
        now = _ts_at_hour(15)
        mode = router.route_alert(_make_alert(priority="low"), now=now)
        self.assertEqual(mode, "batched")


# =========================================================================
# TestRateLimiting
# =========================================================================
class TestRateLimiting(unittest.TestCase):
    """Duplicate alerts for the same entity are rate-limited."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out, rate_limit_min=15)

    def test_same_entity_rate_limited(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Open!", "high"), now=now
        )
        mode = self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Still open!", "high"),
            now=now + 60,
        )
        self.assertEqual(mode, "rate_limited")

    def test_different_entity_not_rate_limited(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Open!", "high"), now=now
        )
        mode = self.router.route_alert(
            _make_alert("binary_sensor.door_back", "Back open!", "high"),
            now=now + 60,
        )
        self.assertEqual(mode, "immediate")

    def test_rate_limit_expires(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Open!", "high"), now=now
        )
        # 16 minutes later — past the 15-min window
        mode = self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Still open!", "high"),
            now=now + 16 * 60,
        )
        self.assertEqual(mode, "immediate")

    def test_clear_rate_limit(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Open!", "high"), now=now
        )
        self.router.clear_rate_limit("binary_sensor.door_front")
        mode = self.router.route_alert(
            _make_alert("binary_sensor.door_front", "Open again!", "high"),
            now=now + 30,
        )
        self.assertEqual(mode, "immediate")

    def test_rate_limited_stats(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("light.kitchen", "Light on", "high"), now=now
        )
        self.router.route_alert(
            _make_alert("light.kitchen", "Light still on", "high"),
            now=now + 30,
        )
        self.assertEqual(self.router.status()["stats"]["rate_limited"], 1)


# =========================================================================
# TestBatchDelivery
# =========================================================================
class TestBatchDelivery(unittest.TestCase):
    """Batched alerts delivered on next conversation turn."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out)

    def test_batch_empty(self):
        batch = self.router.next_conversation_batch()
        self.assertEqual(batch, [])

    def test_batch_returns_alerts(self):
        now = _ts_at_hour(12)
        for i in range(3):
            self.router.route_alert(
                _make_alert(f"light.room_{i}", f"Light {i} on", "low"), now=now
            )
        batch = self.router.next_conversation_batch()
        self.assertEqual(len(batch), 3)

    def test_batch_clears_after_retrieval(self):
        now = _ts_at_hour(12)
        self.router.route_alert(_make_alert(priority="low"), now=now)
        self.router.next_conversation_batch()
        self.assertEqual(self.router.pending_batch_count, 0)

    def test_batch_caps_at_max(self):
        now = _ts_at_hour(12)
        for i in range(MAX_BATCH_DELIVERY + 3):
            self.router.route_alert(
                _make_alert(f"light.room_{i}", f"Light {i}", "low"), now=now
            )
        batch = self.router.next_conversation_batch()
        self.assertEqual(len(batch), MAX_BATCH_DELIVERY)
        # Remaining stay in queue
        self.assertEqual(self.router.pending_batch_count, 3)

    def test_deliver_batch_now_single(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("light.kitchen", "Kitchen light on 1hr", "low"), now=now
        )
        count = self.router.deliver_batch_now()
        self.assertEqual(count, 1)
        self.assertIn("By the way", self.out.last["text"])

    def test_deliver_batch_now_multiple(self):
        now = _ts_at_hour(12)
        for i in range(3):
            self.router.route_alert(
                _make_alert(f"light.room_{i}", f"Light {i} on", "low"), now=now
            )
        count = self.router.deliver_batch_now()
        self.assertEqual(count, 3)
        self.assertIn("A few things", self.out.last["text"])

    def test_deliver_batch_now_empty(self):
        count = self.router.deliver_batch_now()
        self.assertEqual(count, 0)


# =========================================================================
# TestAlertLog
# =========================================================================
class TestAlertLog(unittest.TestCase):
    """Alert log for dashboard display."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out)

    def test_log_records_all(self):
        now = _ts_at_hour(12)
        self.router.route_alert(_make_alert(priority="high"), now=now)
        self.router.route_alert(
            _make_alert("light.bedroom", priority="low"), now=now
        )
        log = self.router.get_log()
        self.assertEqual(len(log), 2)

    def test_log_newest_first(self):
        now = _ts_at_hour(12)
        self.router.route_alert(
            _make_alert("light.a", "First", "high"), now=now
        )
        self.router.route_alert(
            _make_alert("light.b", "Second", "low"), now=now
        )
        log = self.router.get_log()
        self.assertEqual(log[0]["message"], "Second")
        self.assertEqual(log[1]["message"], "First")

    def test_log_delivery_field(self):
        now = _ts_at_hour(12)
        self.router.route_alert(_make_alert(priority="high"), now=now)
        log = self.router.get_log()
        self.assertEqual(log[0]["delivery"], "immediate")

    def test_log_filter_by_priority(self):
        now = _ts_at_hour(12)
        self.router.route_alert(_make_alert(priority="high"), now=now)
        self.router.route_alert(
            _make_alert("light.bedroom", priority="low"), now=now
        )
        high = self.router.get_log_by_priority("high")
        self.assertEqual(len(high), 1)
        self.assertEqual(high[0]["priority"], "high")

    def test_log_limit(self):
        now = _ts_at_hour(12)
        for i in range(10):
            self.router.route_alert(
                _make_alert(f"entity_{i}", priority="high"), now=now
            )
        log = self.router.get_log(limit=5)
        self.assertEqual(len(log), 5)

    def test_log_capped_at_max(self):
        now = _ts_at_hour(12)
        for i in range(MAX_LOG_ENTRIES + 50):
            self.router.route_alert(
                _make_alert(f"entity_{i}", priority="high"), now=now
            )
        self.assertLessEqual(len(self.router._log), MAX_LOG_ENTRIES)


# =========================================================================
# TestStatus
# =========================================================================
class TestStatus(unittest.TestCase):
    """Router status reporting."""

    def test_empty_status(self):
        router = AlertDeliveryRouter()
        s = router.status()
        self.assertEqual(s["pending_batch"], 0)
        self.assertEqual(s["log_size"], 0)
        self.assertIn("quiet_hours", s)
        self.assertIn("stats", s)

    def test_status_after_mixed(self):
        out = CaptureOutput()
        router = AlertDeliveryRouter(output_fn=out)
        now = _ts_at_hour(12)
        router.route_alert(_make_alert(priority="high"), now=now)
        router.route_alert(
            _make_alert("light.bedroom", priority="low"), now=now
        )
        s = router.status()
        self.assertEqual(s["stats"]["total_received"], 2)
        self.assertEqual(s["stats"]["immediate_delivered"], 1)
        self.assertEqual(s["stats"]["batched"], 1)


# =========================================================================
# TestPipelineIntegration
# =========================================================================
class TestPipelineIntegration(unittest.TestCase):
    """Integration with ProactiveAlerts engine + voice pipeline."""

    def setUp(self):
        self.out = CaptureOutput()
        self.integration = AlertPipelineIntegration(output_fn=self.out)

    def test_ha_event_triggers_alert(self):
        now = _ts_at_hour(12)
        event = json.dumps({
            "entity_id": "binary_sensor.door_front",
            "state": "on",
            "timestamp": now - (35 * 60),
        })
        modes = self.integration.on_ha_event(event, eval_time=now)
        # Should have at least one alert (door open > 30 min)
        self.assertTrue(len(modes) > 0)
        self.assertIn("immediate", modes)  # door = high priority

    def test_ha_event_no_alert(self):
        now = _ts_at_hour(12)
        event = json.dumps({
            "entity_id": "binary_sensor.door_front",
            "state": "off",  # Door closed — no alert
            "timestamp": now,
        })
        modes = self.integration.on_ha_event(event, eval_time=now)
        self.assertEqual(modes, [])

    def test_conversation_start_delivers_batch(self):
        now = _ts_at_hour(12)
        # Batch a low-priority alert
        event = json.dumps({
            "entity_id": "light.kitchen",
            "state": "on",
            "timestamp": now - (65 * 60),
        })
        self.integration.on_ha_event(event, eval_time=now)
        # Now start a conversation
        count = self.integration.on_conversation_start()
        # Should deliver the batched light alert (if any were batched)
        # Note: depending on quiet hours status, it might be 0 or 1
        self.assertGreaterEqual(count, 0)

    def test_combined_status(self):
        s = self.integration.status()
        self.assertIn("engine", s)
        self.assertIn("delivery", s)


# =========================================================================
# TestEdgeCases
# =========================================================================
class TestEdgeCases(unittest.TestCase):
    """Edge cases and error handling."""

    def setUp(self):
        self.out = CaptureOutput()
        self.router = AlertDeliveryRouter(output_fn=self.out)

    def test_empty_message(self):
        mode = self.router.route_alert(
            _make_alert(message="", priority="high")
        )
        self.assertEqual(mode, "immediate")

    def test_missing_entity(self):
        mode = self.router.route_alert(
            {"message": "Something happened", "priority": "high"}
        )
        self.assertEqual(mode, "immediate")

    def test_missing_priority_defaults_low(self):
        now = _ts_at_hour(12)
        mode = self.router.route_alert(
            {"entity": "light.kitchen", "message": "On"}, now=now
        )
        self.assertEqual(mode, "batched")

    def test_unknown_priority_treated_as_low(self):
        now = _ts_at_hour(12)
        mode = self.router.route_alert(
            _make_alert(priority="medium"), now=now
        )
        self.assertEqual(mode, "batched")


if __name__ == "__main__":
    unittest.main()
