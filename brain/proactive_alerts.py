import time
import json
import datetime

class ProactiveAlerts:
    def __init__(self):
        # In-memory store for device states and timers
        self.device_states = {}
        # Configuration for alert thresholds
        self.thresholds = {
            "door_open_min": 30,
            "light_empty_room_min": 60,
            "temp_low_celsius": 17.0
        }
        self.active_alerts = []

    def process_event(self, event_json):
        """Process an incoming Home Assistant event"""
        try:
            event = json.loads(event_json)
        except json.JSONDecodeError:
            return

        entity_id = event.get("entity_id")
        state = event.get("state")
        timestamp = event.get("timestamp", datetime.datetime.now().timestamp())

        # Update state cache
        if entity_id not in self.device_states:
            self.device_states[entity_id] = {}
        
        # Only update timestamp if state changed
        if self.device_states[entity_id].get("state") != state:
            self.device_states[entity_id]["last_changed"] = timestamp
            self.device_states[entity_id]["state"] = state
            self.device_states[entity_id]["alerted"] = False

        self.evaluate_alerts(timestamp)

    def evaluate_alerts(self, current_time):
        """Evaluate rules and generate alerts if thresholds are met"""
        for entity_id, data in self.device_states.items():
            state = data.get("state")
            last_changed = data.get("last_changed", current_time)
            alerted = data.get("alerted", False)
            
            elapsed_min = (current_time - last_changed) / 60

            # Rule 1: Door left open > 30 min
            if entity_id.startswith("binary_sensor.door_") and state == "on" and not alerted:
                if elapsed_min >= self.thresholds["door_open_min"]:
                    self.trigger_alert(entity_id, f"Just so you know, the {entity_id.split('.')[-1].replace('_', ' ')} has been open for {int(elapsed_min)} minutes.")
                    self.device_states[entity_id]["alerted"] = True

            # Rule 2: Lights left on in empty room > 1hr
            if entity_id.startswith("light.") and state == "on" and not alerted:
                # Naive check assuming no motion (would need motion sensor context in real app)
                if elapsed_min >= self.thresholds["light_empty_room_min"]:
                    self.trigger_alert(entity_id, f"The {entity_id.split('.')[-1].replace('_', ' ')} light has been on for over an hour. Want me to turn it off?")
                    self.device_states[entity_id]["alerted"] = True

            # Rule 3: Temperature drop
            if entity_id.startswith("sensor.temperature_") and state.replace('.','',1).isdigit() and not alerted:
                temp = float(state)
                if temp < self.thresholds["temp_low_celsius"]:
                    self.trigger_alert(entity_id, f"Temperature dropped to {temp}°C. Heating is off, want me to turn it on?")
                    self.device_states[entity_id]["alerted"] = True

    def trigger_alert(self, entity_id, message, priority="low"):
        print(f"[ALERT - {priority.upper()}] {message}")
        self.active_alerts.append({
            "entity": entity_id,
            "message": message,
            "priority": priority,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def get_pending_alerts(self):
        """Fetch and clear pending alerts for TTS delivery"""
        alerts = list(self.active_alerts)
        self.active_alerts.clear()
        return alerts

if __name__ == "__main__":
    # Test script for the engine
    alerts = ProactiveAlerts()
    now = time.time()
    
    # Simulate front door opening 35 mins ago
    alerts.process_event(json.dumps({
        "entity_id": "binary_sensor.door_front",
        "state": "on",
        "timestamp": now - (35 * 60)
    }))
    
    # Simulate kitchen light on 65 mins ago
    alerts.process_event(json.dumps({
        "entity_id": "light.kitchen",
        "state": "on",
        "timestamp": now - (65 * 60)
    }))
    
    # Simulate temperature drop
    alerts.process_event(json.dumps({
        "entity_id": "sensor.temperature_living_room",
        "state": "16.5",
        "timestamp": now
    }))

    pending = alerts.get_pending_alerts()
    assert len(pending) == 3
    print("Test passed: 3 alerts generated correctly.")
