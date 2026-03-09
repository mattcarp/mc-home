# Claudette Home — Dashboard

This is the standalone HTML dashboard for Claudette Home (EPIC 4).

## Features
- **Zero-dependency**: Single `index.html` file, no build step required.
- **Mission Control UI**: Styled to match the Mission Control aesthetic (dark theme, monospace accents, minimal).
- **Device Control**: Toggles for lights and switches, brightness sliders.
- **Sensor Dashboard**: Live readouts for temperature, humidity, doors, and motion.
- **Voice Log**: Real-time transcript of the conversation with Claudette.
- **Scenes**: One-tap activation of common lighting/device scenes.
- **Responsive**: Adapts to mobile screens and 10" wall-mounted panels.

## Usage
Just open `index.html` in a browser. It currently runs in **Stub Mode**, using hardcoded state and simulating intent parsing. Once the HA Bridge (`ha_bridge.py`) is live, it can be updated to connect to the Home Assistant WebSocket API (`ws://.../api/websocket`) for real-time state sync.

## Next Steps
1. Port this layout to a React component for the `mc-mission-control` repository (`/home` route).
2. Wire up the WebSocket connection to the live HA instance once installed.
