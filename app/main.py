import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.model_manager import ModelManager
from app.agent import Agent, AgentEvent
from app.tools import tool_names

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Origins the served page reports now that we bind to loopback on port 8000.
# A browser-based Cross-Site WebSocket Hijacking attempt arrives with the
# attacker's Origin, which won't be in this set and is rejected before accept().
# NOTE: tied to host/port below — update this if the demo's port changes.
ALLOWED_ORIGINS = {"http://localhost:8000", "http://127.0.0.1:8000"}

# Generous upper bounds to keep a misbehaving/malicious client from driving the
# agent into a very long loop or feeding a pathologically large prompt. Normal
# demo usage stays well under these.
MAX_ITERATIONS_CAP = 20
MAX_PROMPT_CHARS = 10000

model_manager = ModelManager()
agent = Agent(model_manager)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    # Reject browser-originated cross-site connections before accepting. A
    # missing Origin (non-browser clients: CLI tools, tests) is allowed —
    # browsers always send Origin on WebSocket handshakes, so this does not
    # weaken the protection against a malicious web page.
    origin = ws.headers.get("origin")
    if origin is not None and origin not in ALLOWED_ORIGINS:
        await ws.close(code=1008)  # policy violation
        return

    await ws.accept()

    async def emit(event: AgentEvent):
        await ws.send_text(json.dumps(event.to_dict(), default=str))

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                # A malformed/non-JSON frame must not tear down the session.
                await emit(AgentEvent("error", {"text": "Invalid message: expected JSON."}))
                continue
            action = msg.get("action")

            if action == "load_model":
                model_key = msg.get("model", "instruct")
                try:
                    loop = asyncio.get_running_loop()

                    def on_progress(text):
                        asyncio.run_coroutine_threadsafe(
                            emit(AgentEvent("model_loading", {"text": text})),
                            loop,
                        )

                    await asyncio.to_thread(model_manager.load_model, model_key, on_progress)
                    temp, addendum_on = agent.apply_model_defaults(model_key)
                    await emit(AgentEvent("model_ready", {
                        "model": model_key,
                        "temperature": temp,
                        "reasoning_addendum_enabled": addendum_on,
                    }))
                except Exception as e:
                    await emit(AgentEvent("error", {"text": str(e)}))

            elif action == "send_prompt":
                prompt = msg.get("prompt", "")
                if not prompt.strip():
                    continue
                if len(prompt) > MAX_PROMPT_CHARS:
                    await emit(AgentEvent("error", {
                        "text": f"Prompt too long (max {MAX_PROMPT_CHARS} characters).",
                    }))
                    continue
                if model_manager.model is None:
                    await emit(AgentEvent("error", {"text": "No model loaded. Load a model first."}))
                    continue
                try:
                    await agent.run(prompt, emit)
                except Exception as e:
                    await emit(AgentEvent("error", {"text": f"Agent error: {e}"}))

            elif action == "reset":
                agent.reset()
                await emit(AgentEvent("reset", {}))

            elif action == "set_tools":
                # Intersect with known names so a bad payload can't inject
                # unknown tools. Takes effect on the next prompt.
                requested = msg.get("enabled_tools", tool_names())
                agent.enabled_tools = set(requested) & set(tool_names())
                await emit(AgentEvent("tools_updated", {
                    "enabled_tools": sorted(agent.enabled_tools),
                    "all_tools": tool_names(),
                }))

            elif action == "get_tools":
                await emit(AgentEvent("tools_updated", {
                    "enabled_tools": sorted(agent.enabled_tools),
                    "all_tools": tool_names(),
                }))

            elif action == "set_system_prompt":
                agent.system_prompt = msg.get("system_prompt", "")
                await emit(AgentEvent("system_prompt_updated", {
                    "system_prompt": agent.system_prompt,
                }))

            elif action == "set_temperature":
                agent.temperature = float(msg.get("temperature", agent.temperature))
                await emit(AgentEvent("temperature_updated", {
                    "temperature": agent.temperature,
                }))

            elif action == "set_reasoning_addendum":
                agent.reasoning_addendum_enabled = bool(msg.get("enabled", False))
                await emit(AgentEvent("reasoning_addendum_updated", {
                    "enabled": agent.reasoning_addendum_enabled,
                }))

            elif action == "set_max_iterations":
                agent.max_iterations = min(
                    MAX_ITERATIONS_CAP,
                    max(1, int(msg.get("max_iterations", agent.max_iterations))),
                )
                await emit(AgentEvent("max_iterations_updated", {
                    "max_iterations": agent.max_iterations,
                }))

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
