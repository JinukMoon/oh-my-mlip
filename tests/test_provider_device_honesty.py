"""A device the registry line cannot deliver must be refused, not faked.

The inference lines run verbatim and today they pin the device: device='cuda'
in most, cpu=False in a few. `device` only entered the exec namespace, so
get_calculator(device="cpu") built a CUDA calculator and every caller above it
-- run(), Worker, setup_verify's old-driver path -- reported a CPU run that
never happened. AGENTS.md ground rule 6 and docs/host_requirements.md promised
that fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from oh_my_mlip import provider  # noqa: E402

CUDA_LITERAL = ["calc = mace_mp(model='medium-mpa-0', device='cuda')"]
CPU_FLAG = ["calc = UCalculator(checkpoint_path='/w/EquFlashV2.pt', cpu=False)"]
PARAMETERIZED = ["calc = Thing(device=device)"]
NO_TOKEN = ["calc = SevenNetCalculator('7net-mf-ompa', modal='mpa', enable_oeq=True)"]


def test_cpu_request_against_a_cuda_pinned_line_is_refused():
    with pytest.raises(provider.DeviceUnavailableError, match="cannot run on 'cpu'"):
        provider._check_device_honoured("MACE", "MACE-MPA-0", CUDA_LITERAL, "cpu")
    with pytest.raises(provider.DeviceUnavailableError, match="cannot run on 'cpu'"):
        provider._check_device_honoured("EquFlash", "EquFlashV2", CPU_FLAG, "cpu")


def test_the_gpu_path_every_variant_uses_today_is_untouched():
    for lines in (CUDA_LITERAL, NO_TOKEN, PARAMETERIZED):
        provider._check_device_honoured("X", "X-1", lines, "cuda")   # must not raise


def test_a_parameterized_line_honours_either_device():
    provider._check_device_honoured("X", "X-1", PARAMETERIZED, "cpu")


def test_a_line_without_a_device_argument_cannot_promise_cpu():
    with pytest.raises(provider.DeviceUnavailableError, match="no device argument"):
        provider._check_device_honoured("SevenNet", "SevenNet-MF-OMPA", NO_TOKEN, "cpu")


def test_the_registry_lines_this_hub_ships_are_all_gpu_runnable():
    """Guards the fix above: if a real line ever stops answering the default
    device, every model breaks at once rather than in one variant's report."""
    from oh_my_mlip import registry

    models = registry.load_models()
    for family, spec in models.items():
        if family.startswith("_"):
            continue
        for version in spec.get("versions", {}):
            resolved = registry.resolve(version)
            provider._check_device_honoured(family, version, resolved["inference"], "cuda")
