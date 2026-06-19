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

## File structure

```
app/
  main.py           # FastAPI app, WebSocket endpoint, serves static files
  agent.py          # Agent loop, event emission, tool call parsing
  tools.py          # Tool definitions (calculator, read_file, list_files) and execution
  model_manager.py  # Model loading/switching, tokenization, streaming generation
static/
  index.html        # Dashboard layout (two-panel: context window + agent activity)
  style.css         # Dark theme, color-coded roles, CSS grid layout
  app.js            # WebSocket client, UI rendering, event handling
demo_data/          # .txt files the agent can read (program_info, team_roster, project_status)
```

## Running

Always use the workspace virtualenv at `.venv` — `torch`/`transformers` and the
other deps are installed there, NOT in system Python. On Windows the interpreter
is `.\.venv\Scripts\python.exe` (use it directly rather than relying on shell
activation):

```
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
.\.venv\Scripts\python.exe -m pytest tests/
```

Do NOT use `--reload` — it spawns a child process that conflicts with CUDA model loading.

## Mistral-specific details

- Chat template uses special tokens: `[SYSTEM_PROMPT]`, `[AVAILABLE_TOOLS]`, `[INST]`, `[TOOL_CALLS]`, `[ARGS]`, `[TOOL_RESULTS]`
- Tool call output format: `[TOOL_CALLS]function_name[ARGS]{"key": "value"}</s>`
- `tool_call_id` must be exactly 9 alphanumeric characters (a-z, A-Z, 0-9)
- `MistralCommonBackend.apply_chat_template` does NOT accept a dict-return kwarg (no `return_dict`/`return_dist`). When tools are present, pass `tools=`; when there are none, pass neither — a bare tensor result is wrapped into `{"input_ids": ...}` in `model_manager.tokenize`, which is all `generate()` needs.
- `</s>` is the EOS token; `<s>` is BOS (prepended to prompt by the template)

## Dependencies

- `requirements.txt` — primary, uses cu128 (works on both RTX 5090 and RTX 3500 Ada)
- `requirements-laptop-cu124.txt` — fallback if the laptop's NVIDIA driver doesn't support CUDA 12.8
