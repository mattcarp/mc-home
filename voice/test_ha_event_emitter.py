#!/usr/bin/env python3
"""
Tests for ha_event_emitter.py — HA WebSocket event emitter.

Tests cover:
  - EventProcessor: domain filtering, debouncing, state dedup, stats
  - Stub mode: scenario emission, count limits
  - HAEventEmitterThread: construction, start/stop lifecycle
  - Event format: pipeline-compatible JSON structure
  - Edge cases: missing data, unknown domains, rapid events
"""

import asyncio
import json
import sys
import os
import time
import unittest

# Add voice dir to path
VOICE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, VOICE_DIR)

from ha_event_emitter import (
    EventProcessor,
    HAEventEmitterThread,
    ALERT_DOMAINS,
    STUB_SCENARIOS,
    emit_event,
    run_stub,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ha_event(entity_id: str, old_state: str, new_state: str,
                  attributes: dict = None) -> dict:
    """Build a raw HA WebSocket state_changed event."""
    return {
        "data": {
            "entity_id": entity_id,
            "new_state": {
                "state": new_state,
                "attributes": attributes or {"friendly_name": entity_id},
            },
            "old_state": {
                "state": old_state,
                "attributes": attributes or {"friendly_name": entity_id},
            },
        }
    }


# ---------------------------------------------------------------------------
# EventProcessor tests
# ---------------------------------------------------------------------------

class TestEventProcessorDomainFilter(unittest.TestCase):
    """Test domain-based filtering."""

    def test_alert_domain_passes(self):
        """Events from alert-relevant domains should pass through."""
        proc = EventProcessor(debounce_ms=0)
        for domain in ["binary_sensor", "sensor", "light", "switch", "lock", "cover"]:
            event = make_ha_event(f"{domain}.test", "off", "on")
            result = proc.process(event)
            self.assertIsNotNone(result, f"{domain} should pass")

    def test_non_alert_domain_filtered(self):
        """Events from non-alert domains should be filtered."""
        proc = EventProcessor(debounce_ms=0)
        for domain in ["automation", "input_boolean", "person", "zone", "update", "weather"]:
            event = make_ha_event(f"{domain}.test", "off", "on")
            result = proc.process(event)
            self.assertIsNone(result, f"{domain} should be filtered")

    def test_custom_domains(self):
        """Custom domain set should override defaults."""
        proc = EventProcessor(debounce_ms=0, domains={"automation"})
        event = make_ha_event("automation.test", "off", "on")
        self.assertIsNotNone(proc.process(event))

        event2 = make_ha_event("light.test", "off", "on")
        self.assertIsNone(proc.process(event2))


class TestEventProcessorDebounce(unittest.TestCase):
    """Test per-entity debouncing."""

    def test_rapid_same_entity_debounced(self):
        """Same entity within debounce window should be suppressed."""
        proc = EventProcessor(debounce_ms=5000)
        event1 = make_ha_event("light.kitchen", "off", "on")
        event2 = make_ha_event("light.kitchen", "on", "off")

        result1 = proc.process(event1)
        result2 = proc.process(event2)

        self.assertIsNotNone(result1)
        self.assertIsNone(result2)  # Debounced

    def test_different_entities_not_debounced(self):
        """Different entities should not debounce each other."""
        proc = EventProcessor(debounce_ms=5000)
        event1 = make_ha_event("light.kitchen", "off", "on")
        event2 = make_ha_event("light.bedroom", "off", "on")

        self.assertIsNotNone(proc.process(event1))
        self.assertIsNotNone(proc.process(event2))

    def test_zero_debounce_allows_all(self):
        """Debounce=0 should let all events through."""
        proc = EventProcessor(debounce_ms=0)
        event = make_ha_event("light.kitchen", "off", "on")

        self.assertIsNotNone(proc.process(event))
        # Need to change state for the state-dedup filter
        event2 = make_ha_event("light.kitchen", "on", "off")
        self.assertIsNotNone(proc.process(event2))

    def test_debounce_expires(self):
        """After debounce window, same entity should emit again."""
        proc = EventProcessor(debounce_ms=50)  # 50ms debounce
        event1 = make_ha_event("light.kitchen", "off", "on")
        self.assertIsNotNone(proc.process(event1))

        time.sleep(0.06)  # Wait past debounce
        event2 = make_ha_event("light.kitchen", "on", "off")
        self.assertIsNotNone(proc.process(event2))


class TestEventProcessorEntityFilter(unittest.TestCase):
    """Test entity prefix filtering."""

    def test_matching_prefix_passes(self):
        proc = EventProcessor(debounce_ms=0, entity_filter=["binary_sensor.front_door"])
        event = make_ha_event("binary_sensor.front_door_contact", "off", "on")
        self.assertIsNotNone(proc.process(event))

    def test_non_matching_prefix_filtered(self):
        proc = EventProcessor(debounce_ms=0, entity_filter=["binary_sensor.front_door"])
        event = make_ha_event("binary_sensor.hallway_motion", "off", "on")
        self.assertIsNone(proc.process(event))

    def test_empty_filter_allows_all(self):
        proc = EventProcessor(debounce_ms=0, entity_filter=[])
        event = make_ha_event("sensor.temperature", "20", "21")
        self.assertIsNotNone(proc.process(event))


class TestEventProcessorStateDedup(unittest.TestCase):
    """Test state unchanged deduplication."""

    def test_same_state_filtered(self):
        """If old_state == new_state, event should be filtered."""
        proc = EventProcessor(debounce_ms=0)
        event = make_ha_event("light.kitchen", "on", "on")
        self.assertIsNone(proc.process(event))

    def test_different_state_passes(self):
        proc = EventProcessor(debounce_ms=0)
        event = make_ha_event("light.kitchen", "off", "on")
        self.assertIsNotNone(proc.process(event))


class TestEventProcessorFormat(unittest.TestCase):
    """Test output event format."""

    def test_pipeline_event_structure(self):
        """Output should match pipeline's expected state_changed format."""
        proc = EventProcessor(debounce_ms=0)
        event = make_ha_event(
            "binary_sensor.front_door_contact", "off", "on",
            {"friendly_name": "Front Door", "device_class": "door"}
        )
        result = proc.process(event)

        self.assertEqual(result["type"], "state_changed")
        self.assertEqual(result["entity_id"], "binary_sensor.front_door_contact")
        self.assertEqual(result["old_state"], "off")
        self.assertEqual(result["new_state"], "on")
        self.assertIn("timestamp", result)
        self.assertEqual(result["attributes"]["friendly_name"], "Front Door")
        self.assertEqual(result["attributes"]["device_class"], "door")

    def test_missing_attributes_default(self):
        """Missing attributes should get sensible defaults."""
        proc = EventProcessor(debounce_ms=0)
        event = {
            "data": {
                "entity_id": "light.test",
                "new_state": {"state": "on", "attributes": {}},
                "old_state": {"state": "off", "attributes": {}},
            }
        }
        result = proc.process(event)
        self.assertEqual(result["attributes"]["friendly_name"], "light.test")
        self.assertEqual(result["attributes"]["device_class"], "")

    def test_sensor_unit_included(self):
        """Unit of measurement should be included in output."""
        proc = EventProcessor(debounce_ms=0)
        event = make_ha_event(
            "sensor.temperature", "20.0", "21.5",
            {"friendly_name": "Temperature", "unit_of_measurement": "°C",
             "device_class": "temperature"}
        )
        result = proc.process(event)
        self.assertEqual(result["attributes"]["unit_of_measurement"], "°C")


class TestEventProcessorStats(unittest.TestCase):
    """Test statistics tracking."""

    def test_initial_stats(self):
        proc = EventProcessor()
        self.assertEqual(proc.stats, {"received": 0, "emitted": 0, "debounced": 0, "filtered": 0})

    def test_stats_after_events(self):
        proc = EventProcessor(debounce_ms=0)
        proc.process(make_ha_event("light.kitchen", "off", "on"))       # emitted
        proc.process(make_ha_event("automation.test", "off", "on"))     # filtered (domain)
        proc.process(make_ha_event("light.kitchen", "on", "on"))        # filtered (same state)

        stats = proc.stats
        self.assertEqual(stats["received"], 3)
        self.assertEqual(stats["emitted"], 1)
        self.assertEqual(stats["filtered"], 2)


class TestEventProcessorEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_empty_event(self):
        proc = EventProcessor(debounce_ms=0)
        self.assertIsNone(proc.process({}))

    def test_missing_entity_id(self):
        proc = EventProcessor(debounce_ms=0)
        event = {"data": {"new_state": {"state": "on"}, "old_state": {"state": "off"}}}
        self.assertIsNone(proc.process(event))

    def test_missing_new_state(self):
        """None new_state should default to 'unknown'."""
        proc = EventProcessor(debounce_ms=0)
        event = {"data": {"entity_id": "light.test", "new_state": None, "old_state": {"state": "off"}}}
        result = proc.process(event)
        # old_state=off, new_state=unknown → state changed, should emit
        self.assertIsNotNone(result)
        self.assertEqual(result["new_state"], "unknown")

    def test_missing_old_state(self):
        """Entity appearing for the first time (no old_state)."""
        proc = EventProcessor(debounce_ms=0)
        event = {"data": {"entity_id": "light.test", "new_state": {"state": "on", "attributes": {}}, "old_state": None}}
        result = proc.process(event)
        self.assertIsNotNone(result)
        self.assertEqual(result["old_state"], "unknown")


# ---------------------------------------------------------------------------
# Stub mode tests
# ---------------------------------------------------------------------------

class TestStubScenarios(unittest.TestCase):
    """Test stub scenario definitions."""

    def test_scenarios_not_empty(self):
        self.assertGreater(len(STUB_SCENARIOS), 0)

    def test_scenario_structure(self):
        for s in STUB_SCENARIOS:
            self.assertIn("entity_id", s)
            self.assertIn("old_state", s)
            self.assertIn("new_state", s)
            self.assertIn("attributes", s)
            self.assertIn("friendly_name", s["attributes"])

    def test_scenarios_cover_key_domains(self):
        domains = {s["entity_id"].split(".")[0] for s in STUB_SCENARIOS}
        self.assertIn("binary_sensor", domains)
        self.assertIn("sensor", domains)
        self.assertIn("light", domains)
        self.assertIn("lock", domains)


class TestStubEmission(unittest.TestCase):
    """Test stub event emission."""

    def test_stub_emits_counted_events(self):
        """Stub mode with count should emit exactly N events."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            asyncio.run(run_stub(interval=0.01, count=3))

        lines = [l for l in buf.getvalue().strip().split("\n") if l]
        # First line is ha_connected, then 3 events = 4 total
        events = [json.loads(l) for l in lines]
        ha_connected = [e for e in events if e["type"] == "ha_connected"]
        state_changed = [e for e in events if e["type"] == "state_changed"]

        self.assertEqual(len(ha_connected), 1)
        self.assertEqual(len(state_changed), 3)

    def test_stub_events_are_valid_json(self):
        """All stub events should be valid JSON with required fields."""
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            asyncio.run(run_stub(interval=0.01, count=5))

        for line in buf.getvalue().strip().split("\n"):
            if not line:
                continue
            event = json.loads(line)
            self.assertIn("type", event)
            if event["type"] == "state_changed":
                self.assertIn("entity_id", event)
                self.assertIn("old_state", event)
                self.assertIn("new_state", event)
                self.assertIn("timestamp", event)


# ---------------------------------------------------------------------------
# HAEventEmitterThread tests
# ---------------------------------------------------------------------------

class TestEmitterThread(unittest.TestCase):
    """Test the thread-based emitter for pipeline integration."""

    def test_construction(self):
        """Thread should construct without connecting."""
        received = []
        emitter = HAEventEmitterThread(callback=received.append, token="fake")
        self.assertFalse(emitter._running)
        self.assertIsNone(emitter._thread)

    def test_start_stop_no_ha(self):
        """Start/stop should not crash even without HA."""
        received = []
        emitter = HAEventEmitterThread(
            callback=received.append,
            url="http://localhost:99999",  # Nothing here
            token="fake_token",
        )
        emitter.start()
        self.assertTrue(emitter._running)
        time.sleep(0.2)
        emitter.stop()
        self.assertFalse(emitter._running)

    def test_callback_type(self):
        """Callback should be stored."""
        def my_cb(event):
            pass
        emitter = HAEventEmitterThread(callback=my_cb, token="fake")
        self.assertEqual(emitter.callback, my_cb)


# ---------------------------------------------------------------------------
# Alert domains config tests
# ---------------------------------------------------------------------------

class TestAlertDomains(unittest.TestCase):
    """Test domain configuration."""

    def test_essential_domains_included(self):
        """Critical domains for proactive alerts must be in ALERT_DOMAINS."""
        for domain in ["binary_sensor", "sensor", "light", "lock"]:
            self.assertIn(domain, ALERT_DOMAINS)

    def test_no_automation_domain(self):
        """automation domain should NOT be in alert domains (too noisy)."""
        self.assertNotIn("automation", ALERT_DOMAINS)


# ---------------------------------------------------------------------------
# JSON serialization tests
# ---------------------------------------------------------------------------

class TestEventSerialization(unittest.TestCase):
    """Test that emitted events are valid pipeline input."""

    def test_event_is_json_parseable(self):
        """Events should round-trip through JSON."""
        proc = EventProcessor(debounce_ms=0)
        event = make_ha_event("light.kitchen", "off", "on",
                              {"friendly_name": "Kitchen Light"})
        result = proc.process(event)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["entity_id"], "light.kitchen")

    def test_pipeline_can_parse_type(self):
        """Pipeline checks event['type'] == 'state_changed'."""
        proc = EventProcessor(debounce_ms=0)
        result = proc.process(make_ha_event("sensor.temp", "20", "21"))
        self.assertEqual(result["type"], "state_changed")


if __name__ == "__main__":
    unittest.main()
