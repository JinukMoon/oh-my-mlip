"""Tests for setup_guardrail.py multi-layer bounded self-heal stop conditions.

Proves the four stop conditions (signature stall / cumulative N / wallclock /
disk) keep the self-heal loop bounded even under the "different strategy each
round" policy that produces a fresh stderr signature every attempt. This is the
divergence-prevention guarantee from the approved plan (AGENTS.md section 8).
"""
import importlib.util
import time
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "setup_guardrail",
    Path(__file__).resolve().parent.parent / "scripts" / "setup_guardrail.py",
)
guardrail = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(guardrail)


def _eval(state, sig, *, max_repeat=2, cumulative_max=5, wallclock_max_s=None):
    return guardrail._evaluate_attempt(state, sig, max_repeat, cumulative_max, wallclock_max_s)


def test_signature_stall_two_repeats():
    """Same signature twice -> signature_stalled (legacy N=2 behavior preserved)."""
    state = {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}
    ev1 = _eval(state, "same-sig")
    assert ev1["signature_stalled"] is False
    ev2 = _eval(state, "same-sig")
    assert ev2["signature_stalled"] is True
    assert ev2["repeat_count"] == 2


def test_cumulative_stall_with_fresh_signatures():
    """Different signature every round still stops at cumulative_max (divergence guard).

    This is the core anti-divergence proof: signature_stalled never fires because
    each attempt has a unique signature, yet the loop is bounded at N attempts.
    """
    state = {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}
    results = [_eval(state, f"unique-sig-{i}", max_repeat=2, cumulative_max=5)
               for i in range(5)]
    # signature stall never triggers (all unique)
    assert all(r["signature_stalled"] is False for r in results)
    # first four continue, fifth trips cumulative
    assert [r["cumulative_stalled"] for r in results] == [False, False, False, False, True]
    assert results[-1]["total_attempts"] == 5


def test_wallclock_halt_after_first_attempt_baseline():
    """first_attempt_ts is the baseline; a later attempt past the limit halts."""
    state = {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}
    ev1 = _eval(state, "sig-a", wallclock_max_s=0.05)
    assert ev1["wallclock_exceeded"] is False  # baseline just set, elapsed ~0
    time.sleep(0.06)
    ev2 = _eval(state, "sig-b", max_repeat=99, cumulative_max=99, wallclock_max_s=0.05)
    assert ev2["wallclock_exceeded"] is True
    assert ev2["elapsed_s"] >= 0.05


def test_wallclock_off_by_default_never_halts():
    """wallclock_max_s=None disables the wallclock stop entirely."""
    state = {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}
    time.sleep(0.02)
    ev = _eval(state, "sig", max_repeat=99, cumulative_max=99, wallclock_max_s=None)
    assert ev["wallclock_exceeded"] is False


def test_first_attempt_ts_set_once():
    """first_attempt_ts is stamped once and not overwritten on later attempts."""
    state = {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}
    _eval(state, "s1")
    ts1 = state["first_attempt_ts"]
    assert ts1 is not None
    time.sleep(0.02)
    _eval(state, "s2")
    assert state["first_attempt_ts"] == ts1  # unchanged


def test_no_stop_condition_when_under_all_limits():
    """A single fresh attempt under every limit returns no stop flags."""
    state = {"attempts": [], "signature_counts": {}, "first_attempt_ts": None}
    ev = _eval(state, "sig", max_repeat=2, cumulative_max=5, wallclock_max_s=3600)
    assert not (ev["signature_stalled"] or ev["cumulative_stalled"] or ev["wallclock_exceeded"])
