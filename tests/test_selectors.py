"""Multi-head selectors: the requested name exists, and a fallback is refused.

A multi-head checkpoint is told its head by one calculator keyword (MACE
`head`, UMA `task_name`, SevenNet `modal`). mace-torch does not refuse an
unknown head: it logs a warning and computes with the checkpoint's last head.
A registry that asked MACE-MH-1 for head='oc20' (no such head) therefore ran
the omat_pbe head under the OC20 name. So every selector in models.json is
declared with the names the checkpoint lists, and its inference lines refuse
a calculator that settled on another name.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import provider, registry  # noqa: E402
from oh_my_mlip.registry import SELECTOR_KWARGS, selector_check_line  # noqa: E402

MODELS = registry.load_models()
INFERENCE_KEYS = ("inference", "inference_sm86", "inference_sm89")
SELECTOR_CALL = re.compile(r"\b(" + "|".join(SELECTOR_KWARGS) + r")\s*=\s*['\"]([^'\"]+)['\"]")
VERSIONS = [(family, version, entry) for family, info in MODELS.items()
            if not family.startswith("_") and isinstance(info, dict)
            for version, entry in (info.get("versions") or {}).items()]


def _selector_calls(lines: list[str]) -> list[tuple[int, str, str]]:
    return [(i, m.group(1), m.group(2)) for i, line in enumerate(lines) for m in SELECTOR_CALL.finditer(line)]


@pytest.mark.parametrize("family,version,entry", VERSIONS, ids=[v for _, v, _ in VERSIONS])
def test_every_selector_is_declared_valid_and_checked(family, version, entry):
    sel = entry.get("selector")
    seen = False
    for key in INFERENCE_KEYS:
        lines = entry.get(key)
        if not lines:
            continue
        calls = _selector_calls(lines)
        if not calls:
            continue
        seen = True
        assert len(calls) == 1, f"{version} {key}: one selector per calculator, got {calls}"
        (i, kwarg, value), = calls
        assert sel, (f"{version}: {kwarg}={value!r} needs a \"selector\" block with the names the checkpoint lists "
                     "(see oh_my_mlip/registry.py SELECTOR_KWARGS)")
        assert (sel["kwarg"], sel["value"]) == (kwarg, value), f"{version}: selector block != the inference line"
        assert value in sel["choices"], f"{version}: {kwarg}={value!r} is not one of the checkpoint's {sel['choices']}"
        assert lines[i + 1:i + 2] == [selector_check_line(version, kwarg, value)], (
            f"{version} {key}: the line after the calculator must be selector_check_line({version!r}, {kwarg!r}, "
            f"{value!r})")
    if sel:
        assert seen, f"{version}: a selector block without a {sel['kwarg']}= argument in any inference line"


def test_the_registry_carries_the_multihead_variants_it_should():
    declared = {v: e["selector"]["value"] for _, v, e in VERSIONS if e.get("selector")}
    assert declared["MACE-MH-1-OC20"] == "oc20_usemppbe" and declared["MACE-MH-1-OMAT"] == "omat_pbe"
    mace = MODELS["MACE"]["versions"]["MACE-MH-1-OC20"]["selector"]["choices"]
    assert "oc20" not in mace   # the misspelling that made MACE-MH-1 silently compute omat_pbe


class _MaceLike:
    """mace-torch 0.3.15 MACECalculator head handling: an unknown head is a
    warning and the last head is used."""
    def __init__(self, head):
        self.available_heads = ["matpes_r2scan", "mp_pbe_refit_add", "spice_wB97M", "oc20_usemppbe", "omol",
                                "omat_pbe"]
        self.head = head if head in self.available_heads else self.available_heads[-1]


def test_the_check_line_refuses_a_silent_mace_fallback():
    line = selector_check_line("MACE-MH-1-OC20", "head", "oc20")
    with pytest.raises(RuntimeError, match=r"selected head 'omat_pbe', not 'oc20'; available: \['matpes_r2scan'"):
        exec(line, {"calc": _MaceLike("oc20")})  # noqa: S102
    exec(selector_check_line("MACE-MH-1-OC20", "head", "oc20_usemppbe"), {"calc": _MaceLike("oc20_usemppbe")})  # noqa: S102


@pytest.mark.parametrize("kwarg", ["task_name", "modal"])
def test_the_check_line_reads_the_attribute_named_by_the_kwarg(kwarg):
    calc = type("Calc", (), {kwarg: "omat"})()
    exec(selector_check_line("X-1", kwarg, "omat"), {"calc": calc})  # noqa: S102
    with pytest.raises(RuntimeError, match=f"selected {kwarg} 'omat', not 'oc20'"):
        exec(selector_check_line("X-1", kwarg, "oc20"), {"calc": calc})  # noqa: S102
    with pytest.raises(RuntimeError, match=f"selected {kwarg} None"):   # a calculator without the attribute
        exec(selector_check_line("X-1", kwarg, "oc20"), {"calc": object()})  # noqa: S102


def test_an_unknown_kwarg_is_a_registry_error():
    with pytest.raises(registry.RegistryError, match="not one of"):
        selector_check_line("X-1", "fidelity", "a")


def test_get_calculator_runs_the_check(monkeypatch):
    """get_calculator (and with it run(), the Worker and setup_verify) exec the
    inference lines, check line included."""
    spec = {"version": "MACE-MH-1-OC20", "imports": [],
            "inference": ["calc = _MaceLike('oc20')", selector_check_line("MACE-MH-1-OC20", "head", "oc20")]}
    monkeypatch.setattr(provider.registry, "resolve", lambda *a, **k: spec)
    monkeypatch.setattr(provider, "ensure_weights", lambda *a, **k: None)
    monkeypatch.setattr(provider, "_check_device_honoured", lambda *a, **k: None)
    monkeypatch.setitem(__builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__, "_MaceLike",
                        _MaceLike)
    with pytest.raises(RuntimeError, match="selected head 'omat_pbe', not 'oc20'"):
        provider.get_calculator("MACE", "MACE-MH-1-OC20")
