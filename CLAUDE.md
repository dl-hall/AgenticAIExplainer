# Agentic AI Explainer

## What this is

A visual demonstration app that explains how agentic AI works to non-expert audiences. Uses a local Ministral 3B LLM as the agent's "brain," with a web-based UI that makes each step of the agentic process visible.

## Hardware

- **Development**: RTX 5090 desktop (32GB VRAM, CUDA 13.2 driver)
- **Demo**: RTX 3500 Ada laptop (12GB VRAM) — all code must run on this machine
- Transfer via GitHub

## Architecture

- Backend: Python + FastAPI + WebSocket
- Frontend: Plain HTML/CSS/JS (no build step, no npm)
- Models: Ministral 3B via KaggleHub, two variants switchable at runtime:
  - `ministral-3-3b-instruct-2512` (no thinking)
  - `ministral-3-3b-reasoning-2512` (with thinking blocks)
- One model loaded at a time (swap on switch)

## Key constraints

- The audience are non-experts. Every design decision should prioritize clarity of explanation over capability.
- The presenter controls pacing — the UI must support starting new prompts, resetting, and switching models at any time.
- No npm/node dependencies. Frontend is static files served by FastAPI.
- VRAM budget: ~6GB for model at bfloat16, leaving headroom on the 12GB laptop GPU.

## Dependencies

- `requirements.txt` — primary, uses cu128 (works on both RTX 5090 and RTX 3500 Ada)
- `requirements-laptop-cu124.txt` — fallback if the laptop's NVIDIA driver doesn't support CUDA 12.8
