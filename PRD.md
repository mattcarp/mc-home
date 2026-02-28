# Claudette Home — Product Requirements Document

**Version:** 0.1 | **Date:** 2026-02-28 | **Owner:** Matt Carpenter

---

## Vision

A natural-language home control system where "Claudette" is the wake word, the brain, and eventually the brand. Built on open-source device protocols (Home Assistant, Zigbee, Matter) but with a proprietary AI layer that makes every other smart home system feel like a toy.

**Tagline:** *Talk to your home like a person.*

---

## The Problem

10 years of Alexa: rigid commands, broken context, no memory, no personality, cloud-dependent, Amazon's data. Users have learned to talk like robots to their houses. That's backwards.

---

## The Solution

- **Wake word:** "Claudette"
- **Brain:** Claude-powered AI (via OpenClaw) with full home context, memory, personality
- **Device layer:** Home Assistant (open source) handles the protocol complexity
- **Hardware:** Android POE touchscreen panels (YC-SM10P or similar) in each room
- **Voice:** Whisper STT (local) + Gemini Kore TTS (Claudette's voice)
- **No cloud dependency for core function**

---

## Pilot

**Phase 1:** Xagħra, Gozo house — single YC-SM10P panel, prototype
**Phase 2:** Full Xagħra deployment — all rooms
**Phase 3:** Valletta house
**Phase 4:** Product — Claudette as a purchasable home AI system

---

## Core User Stories

1. "Hey Claudette, it's getting dark — sort the living room"
   → Dims lights, closes shutters, maybe asks if music is wanted

2. "Claudette, I'm heading out"  
   → Turns off lights, locks doors, arms security, says goodbye

3. "Claudette, what's going on in the house?"
   → Summarises: who's home, what's on, temperature, any alerts

4. "Claudette, set up for dinner"
   → Runs "dinner" scene: dining lights, music low, kitchen bright

5. "Claudette, remind me to take the bins out at 8"
   → Sets reminder, delivers it through nearest panel at 8pm

---

## Architecture

```
[Wake word: "Claudette"] 
  → [Local STT: Whisper on Workshop]
  → [Claudette brain: OpenClaw + Claude]
    ↕ [Home context: HA state injected every 30s]
  → [Intent → HA service call]
  → [TTS response: Gemini Kore]
  → [Panel speaker output]
```

---

## Non-Goals (v1)

- NOT open source (proprietary product)
- NOT building our own device protocols (use HA)
- NOT multi-tenant cloud product yet (local first)

---

## Success Metrics

- Wake word response latency < 2 seconds
- Correct intent resolution > 90% of commands
- Mattie stops using Alexa within 30 days of v1 install
- "Claudette, turn on the lights" works reliably 100% of the time
