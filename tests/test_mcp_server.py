"""GPU-free tests for the oh_my_mlip MCP adapter (oh_my_mlip/mcp_server.py).

These verify the THIN-ADAPTER contract without a GPU, conda env, torch, or any
model weights:

  * importing oh_my_mlip succeeds WITHOUT mcp installed (guarded-import proof);
  * the server registers EXACTLY the expected tool names;
  * the GPU-free tools (list_models / describe_model / model_status) return data
    that matches the real models.json;
  * run_singlepoint's structure->Atoms conversion works for every documented
    input form;
  * a compute tool (run_singlepoint) returns a graceful "env not installed"
    error path when the model's env is absent (no traceback).

Tests that need the mcp SDK are skipped cleanly when it is not installed; the
guarded-import test always runs.
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from oh_my_mlip import mcp_server

REPO_ROOT = Path(__file__).resolve().parent.parent

def _mcp_available() -> bool:
    try:
        return importlib.util.find_spec("mcp") is not None
    except ImportError:
        # A meta-path finder may raise to block mcp (used by the guarded-import
        # proof) — treat that as "not available".
        return False


mcp_installed = _mcp_available()
requires_mcp = pytest.mark.skipif(
    not mcp_installed, reason="mcp SDK not installed (optional extra)"
)


# ── guarded-import proof (ALWAYS runs, even without mcp) ─────────────────────
def test_oh_my_mlip_imports_without_mcp():
    """Importing oh_my_mlip must NOT require mcp.

    Run in a subprocess with mcp made unimportable so the proof holds even on a
    host where mcp IS installed (mirrors how torch/hf are guarded elsewhere).
    """
    code = (
        "import sys, importlib.abc, importlib.machinery\n"
        "class _Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'mcp' or name.startswith('mcp.'):\n"
        "            raise ImportError('mcp blocked for test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import oh_my_mlip\n"
        "assert 'mcp' not in sys.modules, 'mcp must not be imported'\n"
        "print('OK', len(oh_my_mlip.__all__))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("OK")


def test_mcp_server_module_constants_present():
    """The tool-name surface is declared on the module (no mcp needed to read)."""
    assert mcp_server.TOOL_NAMES == (
        "list_models",
        "describe_model",
        "model_status",
        "run_singlepoint",
        "run_relax",
        "install_model",
        "run_catbench",
    )
    assert mcp_server.GPU_FREE_TOOLS == {
        "list_models",
        "describe_model",
        "model_status",
    }


# ── structure -> Atoms conversion (GPU-free; needs ase, present everywhere) ──
def test_structure_to_atoms_simple_dict():
    pytest.importorskip("ase")
    atoms = mcp_server.structure_to_atoms(
        {
            "symbols": "H2O",
            "positions": [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        }
    )
    assert list(atoms.get_chemical_symbols()) == ["H", "H", "O"]
    assert atoms.get_positions().shape == (3, 3)


def test_structure_to_atoms_with_cell_and_pbc():
    pytest.importorskip("ase")
    atoms = mcp_server.structure_to_atoms(
        {
            "symbols": "Cu",
            "positions": [[0, 0, 0]],
            "cell": [[3.6, 0, 0], [0, 3.6, 0], [0, 0, 3.6]],
            "pbc": [True, True, True],
        }
    )
    assert all(atoms.pbc)
    assert atoms.cell[0][0] == pytest.approx(3.6)


def test_structure_to_atoms_todict_roundtrip():
    pytest.importorskip("ase")
    from ase.build import bulk

    original = bulk("Cu", "fcc", a=3.61, cubic=True)
    restored = mcp_server.structure_to_atoms(original.todict())
    assert list(restored.get_chemical_symbols()) == list(
        original.get_chemical_symbols()
    )
    assert restored.get_positions() == pytest.approx(original.get_positions())


def test_structure_to_atoms_path(tmp_path):
    pytest.importorskip("ase")
    from ase.build import bulk

    p = tmp_path / "cu.xyz"
    bulk("Cu", "fcc", a=3.61, cubic=True).write(str(p))
    atoms = mcp_server.structure_to_atoms(str(p))
    assert len(atoms) == 4


def test_structure_to_atoms_rejects_bad_input():
    with pytest.raises(ValueError):
        mcp_server.structure_to_atoms(42)
    with pytest.raises(ValueError):
        mcp_server.structure_to_atoms({"symbols": "H"})  # no positions
    with pytest.raises(ValueError):
        mcp_server.structure_to_atoms("/no/such/structure/file")


# ── server registers exactly the expected tools ──────────────────────────────
@requires_mcp
def test_server_registers_expected_tools():
    import asyncio

    server = mcp_server.build_server()
    tools = asyncio.run(server.list_tools())
    names = sorted(t.name for t in tools)
    assert names == sorted(mcp_server.TOOL_NAMES)


def _tool(server, name):
    """Return the underlying python callable for a registered tool."""
    return server._tool_manager.get_tool(name).fn


# ── GPU-free tools return correct data vs the real models.json ───────────────
@requires_mcp
def test_list_models_tool_matches_registry():
    from oh_my_mlip import registry

    server = mcp_server.build_server()
    out = _tool(server, "list_models")()
    frameworks = [m["framework"] for m in out["models"]]
    assert frameworks == registry.list_models()
    mace = next(m for m in out["models"] if m["framework"] == "MACE")
    assert mace["default_version"] == "MACE-MPA-0"
    assert "MACE-MPA-0" in mace["versions"]


@requires_mcp
def test_describe_model_tool_returns_codegen_dict():
    server = mcp_server.build_server()
    spec = _tool(server, "describe_model")("MACE", "MACE-MPA-0")
    assert spec["model"] == "MACE"
    assert spec["env"] == "mace"
    assert spec["gated"] is False
    for key in ("python", "imports", "inference", "env_run", "validation"):
        assert key in spec
    assert "${OH_MY_MLIP_HOME}" not in spec["python"]


@requires_mcp
def test_describe_model_default_version():
    server = mcp_server.build_server()
    spec = _tool(server, "describe_model")("MACE")
    assert spec["version"] == "MACE-MPA-0"


@requires_mcp
def test_model_status_tool_matches_status_table():
    server = mcp_server.build_server()
    out = _tool(server, "model_status")()
    rows = out["rows"]
    # One row per (framework, version) in models.json.
    from oh_my_mlip import registry

    models = registry.load_models()
    total = sum(
        len(info.get("versions", {}))
        for name, info in models.items()
        if not name.startswith("_")
    )
    assert len(rows) == total
    uma = [r for r in rows if r["framework"] == "UMA"]
    assert uma and all(r["gated"] for r in uma)
    mace = [r for r in rows if r["framework"] == "MACE"]
    # Derived from dist_manifest.json: published (<rev>) once the tarball is
    # live, upload-pending before that. Accept either so the test tracks the
    # mechanism, not the current publication state.
    assert all(
        r["v1_tarball"].startswith(("published (", "upload-pending"))
        for r in mace
    )
    # validation labels are humanized
    assert any(r["validation"] == "validated (sm89)" for r in rows)


# ── compute tool: graceful "env not installed" path (no GPU, env absent) ─────
@requires_mcp
def test_run_singlepoint_env_absent_graceful(monkeypatch):
    """When the model env is not materialized, run_singlepoint returns an
    actionable error (not a traceback) and never reaches oh_my_mlip.run."""
    server = mcp_server.build_server()
    run_sp = _tool(server, "run_singlepoint")

    # Force the env-present check to report absent regardless of host state.
    monkeypatch.setattr(mcp_server, "_env_python_exists", lambda *a, **k: False)

    called = {"run": False}

    def _boom(*a, **k):
        called["run"] = True
        raise AssertionError("run() must not be called when env is absent")

    import oh_my_mlip

    monkeypatch.setattr(oh_my_mlip, "run", _boom)

    out = run_sp("MACE", {"symbols": "Cu", "positions": [[0, 0, 0]]})
    assert out["ok"] is False
    assert "install" in out["error"].lower()
    assert called["run"] is False


@requires_mcp
def test_run_singlepoint_bad_structure_graceful():
    server = mcp_server.build_server()
    run_sp = _tool(server, "run_singlepoint")
    out = run_sp("MACE", 12345)
    assert out["ok"] is False
    assert "structure" in out["error"].lower()
