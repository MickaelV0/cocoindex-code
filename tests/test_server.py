import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from cocoindex_code import client as daemon_client
from cocoindex_code._version import __version__
from cocoindex_code.protocol import IndexResponse, SearchResponse
from cocoindex_code.server import create_mcp_server


async def test_mcp_server_uses_v2_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        daemon_client,
        "search",
        lambda **kwargs: SearchResponse(success=True, offset=kwargs["offset"]),
    )

    server = create_mcp_server(".")
    async with Client(server, raise_exceptions=True) as client:
        tools = await client.list_tools()
        result = await client.call_tool(
            "search",
            {"query": "authentication", "refresh_index": False},
        )

    assert [tool.name for tool in tools.tools] == ["search"]
    assert result.structured_content == {
        "success": True,
        "results": [],
        "total_returned": 0,
        "offset": 0,
        "message": None,
    }


async def test_mcp_server_reports_own_version() -> None:
    """The handshake advertises our version, not the SDK's or an empty string."""
    server = create_mcp_server(".")
    async with Client(server, raise_exceptions=True) as client:
        assert client.server_info is not None
        assert client.server_info.name == "cocoindex-code"
        assert client.server_info.version == __version__


def _make_project(root: Path) -> Path:
    (root / ".cocoindex_code").mkdir(parents=True)
    (root / ".cocoindex_code" / "settings.yml").write_text("include_patterns: []\n")
    return root


class _DaemonCalls:
    """Records the project root of every daemon call the MCP server makes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.index_roots: list[str] = []
        self.search_roots: list[str] = []

        def _index(project_root: str, **_: Any) -> IndexResponse:
            self.index_roots.append(project_root)
            return IndexResponse(success=True)

        def _search(**kwargs: Any) -> SearchResponse:
            self.search_roots.append(kwargs["project_root"])
            return SearchResponse(success=True)

        monkeypatch.setattr(daemon_client, "index", _index)
        monkeypatch.setattr(daemon_client, "search", _search)


async def test_search_project_path_selects_project_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subdirectory of another project routes both indexing and search there."""
    project_a = _make_project(tmp_path / "a")
    project_b = _make_project(tmp_path / "b")
    sub = project_b / "src" / "pkg"
    sub.mkdir(parents=True)
    calls = _DaemonCalls(monkeypatch)

    server = create_mcp_server(str(project_a))
    async with Client(server, raise_exceptions=True) as client:
        other = await client.call_tool("search", {"query": "q", "project_path": str(sub)})
        default = await client.call_tool("search", {"query": "q"})

    assert other.structured_content is not None and other.structured_content["success"] is True
    assert default.structured_content is not None and default.structured_content["success"] is True
    assert calls.index_roots == [str(project_b), str(project_a)]
    assert calls.search_roots == [str(project_b), str(project_a)]


async def test_search_project_path_without_project_fails_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No project at project_path is an error, never the startup project's results."""
    project_a = _make_project(tmp_path / "a")
    bare = tmp_path / "bare"
    bare.mkdir()
    calls = _DaemonCalls(monkeypatch)

    server = create_mcp_server(str(project_a))
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("search", {"query": "q", "project_path": str(bare)})

    assert result.structured_content is not None
    assert result.structured_content["success"] is False
    message = result.structured_content["message"]
    assert str(bare) in message and "ccc init" in message
    assert calls.index_roots == [] and calls.search_roots == []


async def test_search_without_any_project_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server started outside a project needs project_path on every call."""
    calls = _DaemonCalls(monkeypatch)

    server = create_mcp_server()
    async with Client(server, raise_exceptions=True) as client:
        result = await client.call_tool("search", {"query": "q"})

    assert result.structured_content is not None
    assert result.structured_content["success"] is False
    assert "project_path" in result.structured_content["message"]
    assert calls.index_roots == [] and calls.search_roots == []


async def test_ccc_mcp_starts_outside_a_project(tmp_path: Path) -> None:
    """`ccc mcp` launched in a non-project directory serves MCP instead of exiting."""
    ccc_dir = tmp_path / "ccc"
    ccc_dir.mkdir()
    (ccc_dir / "global_settings.yml").write_text("embedding:\n  model: test\n  provider: litellm\n")
    bare = tmp_path / "bare"
    bare.mkdir()
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", "from cocoindex_code.cli import app; app()", "mcp"],
        cwd=bare,
        env={"COCOINDEX_CODE_DIR": str(ccc_dir), "HOME": str(tmp_path), "PATH": ""},
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("search", {"query": "q"})

    assert [tool.name for tool in tools.tools] == ["search"]
    assert result.structured_content is not None
    assert result.structured_content["success"] is False
