"""Security tests for the /ws WebSocket endpoint (findings 2b, 4, 5).

These use FastAPI's TestClient and only exercise handlers that do NOT touch the
model (get_tools, set_max_iterations, malformed frames), so no GPU / model load
is required. The Origin check runs during the handshake, before any model use.

Run with:  python -m pytest tests/
"""

import pytest
from starlette.websockets import WebSocketDisconnect
from fastapi.testclient import TestClient

from app.main import app, MAX_ITERATIONS_CAP

client = TestClient(app)


# --- Finding 2b: Origin check -------------------------------------------------

def test_cross_site_origin_is_rejected():
    """A browser-originated connection from another site is closed before
    accept(), so the handshake never completes."""
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as ws:
            ws.receive_json()


def test_allowed_origin_connects():
    with client.websocket_connect("/ws", headers={"origin": "http://localhost:8000"}) as ws:
        ws.send_json({"action": "get_tools"})
        data = ws.receive_json()
        assert data["type"] == "tools_updated"


def test_missing_origin_connects():
    """Non-browser clients (no Origin header) are allowed."""
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "get_tools"})
        data = ws.receive_json()
        assert data["type"] == "tools_updated"


# --- Finding 4: max_iterations is clamped -------------------------------------

def test_max_iterations_is_capped():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "set_max_iterations", "max_iterations": 999})
        data = ws.receive_json()
        assert data["type"] == "max_iterations_updated"
        assert data["max_iterations"] == MAX_ITERATIONS_CAP


def test_max_iterations_floor_is_one():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"action": "set_max_iterations", "max_iterations": 0})
        data = ws.receive_json()
        assert data["max_iterations"] == 1


# --- Finding 5: malformed frame does not kill the session ---------------------

def test_malformed_frame_yields_error_and_keeps_socket_open():
    with client.websocket_connect("/ws") as ws:
        ws.send_text("this is not json")
        data = ws.receive_json()
        assert data["type"] == "error"

        # Socket is still usable for a following valid message.
        ws.send_json({"action": "get_tools"})
        data = ws.receive_json()
        assert data["type"] == "tools_updated"
