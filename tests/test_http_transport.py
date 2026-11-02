"""Exercise HTTP through the SDK client, including image delivery and session cleanup."""

import base64
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import anyio
import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ImageContent

from blender_mcp import server, trajectory


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_server_options_keep_blender_and_http_addresses_separate():
    args = server.parse_server_args([
        "--host", "localhost", "--port", "9877",
        "--transport", "streamable-http", "--http-host", "0.0.0.0",
        "--http-port", "8765", "--http-allowed-host", "blender.local:8765",
        "--http-allowed-host", "192.0.2.10:8765",
    ])
    assert (args.host, args.port) == ("localhost", 9877)
    assert (args.transport, args.http_host, args.http_port) == ("streamable-http", "0.0.0.0", 8765)


def test_wildcard_listener_requires_an_explicit_allowed_host():
    with pytest.raises(SystemExit):
        server.parse_server_args(["--transport", "streamable-http", "--http-host", "0.0.0.0"])


def test_default_entry_point_still_runs_stdio(monkeypatch):
    run = Mock()
    monkeypatch.setattr(server.sys, "argv", ["mcp-for-blender"])
    monkeypatch.setattr(server.mcp, "run", run)
    monkeypatch.setattr(server.context_log, "install", Mock())
    server.main()
    run.assert_called_once_with()


@pytest.mark.anyio
async def test_http_round_trip_images_host_checks_and_overlapping_sessions(monkeypatch):
    # The addon is the only fake: MCP discovery, HTTP, images and session lifespans
    # all use the real server and SDK transport.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6uQAAAAASUVORK5CYII="
    )
    blender = Mock()

    def send_command(command, params=None, **kwargs):
        if command == "get_viewport_screenshot":
            Path(params["filepath"]).write_bytes(png)
            return {}
        return {"result": "HTTP round trip"}

    blender.send_command.side_effect = send_command

    def connection():
        server._blender_connection = blender
        return blender

    monkeypatch.setenv("DISABLE_TELEMETRY", "true")
    monkeypatch.setattr(server, "get_blender_connection", connection)
    monkeypatch.setattr(server, "check_addon_status_on_startup", lambda: SimpleNamespace(needs_action=False, message=""))
    monkeypatch.setattr(server, "record_startup", Mock())
    monkeypatch.setattr(trajectory, "get_trajectory_recorder", lambda: Mock())
    monkeypatch.setattr(server, "_active_sessions", 0)
    monkeypatch.setattr(server, "_blender_connection", None)
    monkeypatch.setattr(server.mcp, "settings", server.mcp.settings.model_copy(deep=True))
    monkeypatch.setattr(server.mcp, "_session_manager", None)
    args = server.parse_server_args([
        "--transport", "streamable-http", "--http-host", "0.0.0.0",
        "--http-allowed-host", "blender.local",
    ])
    server.configure_http(args)
    app = server.mcp.streamable_http_app()
    url = "http://blender.local:8765/mcp"

    @asynccontextmanager
    async def session(client):
        async with streamable_http_client(url, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as connected:
                await connected.initialize()
                yield connected

    with anyio.fail_after(10):
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as client:
                response = await client.post(url, headers={"Host": "untrusted.example"}, json={})
                assert response.status_code == 421
                response = await client.post(url, headers={"Origin": "https://untrusted.example"}, json={})
                assert response.status_code == 403

                async with session(client) as first:
                    assert "look" in {tool.name for tool in (await first.list_tools()).tools}
                    async with session(client) as second:
                        reply = await second.call_tool("execute_blender_code", {"code": "print('HTTP round trip')"})
                        assert not reply.isError
                        assert "HTTP round trip" in reply.content[0].text
                    # Closing one client must not disconnect another client's addon.
                    blender.disconnect.assert_not_called()
                    reply = await first.call_tool("look", {"mode": "viewport"})
                    assert not reply.isError
                    [image] = [item for item in reply.content if isinstance(item, ImageContent)]
                    assert base64.b64decode(image.data) == png

    blender.disconnect.assert_called_once_with()
    assert server._blender_connection is None
    assert server._active_sessions == 0
