"""OMP marketplace/plugin manifests stay valid, launch `ccc mcp`, and declare index hooks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_omp_marketplace_catalog_points_at_repo_root() -> None:
    catalog = json.loads((REPO_ROOT / ".omp-plugin" / "marketplace.json").read_text())
    assert catalog["name"] == "cocoindex-code"
    assert catalog["owner"]["name"] == "CocoIndex"
    assert catalog["plugins"][0]["name"] == "cocoindex-code"
    assert catalog["plugins"][0]["source"] == "./"


def test_omp_plugin_manifest_replaces_mcp_with_ccc_mcp() -> None:
    manifest = json.loads((REPO_ROOT / ".omp-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "cocoindex-code"
    assert manifest["mcpServers"] == "./.mcp.json"
    mcp = json.loads((REPO_ROOT / ".mcp.json").read_text())
    assert mcp["mcpServers"]["cocoindex-code"] == {"command": "ccc", "args": ["mcp"]}


def test_omp_package_declares_ccc_index_extension() -> None:
    package = json.loads((REPO_ROOT / "package.json").read_text())
    entries = package["omp"]["extensions"]
    assert entries == ["./extensions/ccc-index.ts"]
    extension = REPO_ROOT / "extensions" / "ccc-index.ts"
    assert extension.is_file()
    source = extension.read_text()
    assert 'pi.on("session_start"' in source
    assert 'pi.on("tool_result"' in source


def test_claude_hooks_cover_session_start_and_post_edit() -> None:
    hooks = json.loads((REPO_ROOT / "hooks" / "hooks.json").read_text())["hooks"]
    assert "SessionStart" in hooks
    assert "PostToolUse" in hooks
    matcher = hooks["PostToolUse"][0]["matcher"]
    assert "Edit" in matcher
    assert "Write" in matcher


def _hook_commands() -> dict[str, str]:
    hooks = json.loads((REPO_ROOT / "hooks" / "hooks.json").read_text())["hooks"]
    return {
        event: h["command"] for event, entries in hooks.items() for e in entries for h in e["hooks"]
    }


@pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="hooks are bash commands"
)
@pytest.mark.parametrize("command", list(_hook_commands().values()), ids=list(_hook_commands()))
def test_hook_indexes_only_dirs_with_project_settings(command: str, tmp_path: Path) -> None:
    """A dir holding only the user ``global_settings.yml`` is not a project; the hook
    must not run ``ccc index`` there (e.g. ``$HOME``). A dir with ``settings.yml`` is.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "ccc.log"
    fake_ccc = bin_dir / "ccc"
    fake_ccc.write_text(f'#!/bin/sh\necho "$PWD $*" >> "{log}"\n')
    fake_ccc.chmod(0o755)

    user_dir = tmp_path / "user_only"
    (user_dir / ".cocoindex_code").mkdir(parents=True)
    (user_dir / ".cocoindex_code" / "global_settings.yml").write_text("embedding: {model: x}\n")
    project = tmp_path / "project"
    (project / ".cocoindex_code").mkdir(parents=True)
    (project / ".cocoindex_code" / "settings.yml").write_text("include_patterns: []\n")

    def run(root: Path) -> None:
        env = {
            **os.environ,
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            "CLAUDE_PROJECT_DIR": str(root),
        }
        subprocess.run(["bash", "-c", command], env=env, check=True, cwd=tmp_path)

    run(user_dir)
    assert not log.exists()

    run(project)
    assert log.read_text().strip() == f"{project} index"
