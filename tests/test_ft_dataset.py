"""GPU-free tests for scripts/ft_dataset.py.

The converter's whole point (C9): a SINGLE extxyz writer attaches a
`SinglePointCalculator` AND duplicates the same numbers into
`REF_energy`/`REF_forces`, so MACE (reads `REF_*`) and SevenNet/NequIP/
Allegro/GRACE (read the calculator) are fed from one file. These tests assert
that duplication actually round-trips, that the train/valid split is
seed-deterministic, and that the four documented-not-implemented targets
refuse cleanly instead of silently falling back to something else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")  # GPU-free CI has no numpy/ase
pytest.importorskip("ase")
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ft_dataset  # noqa: E402


def _make_frames(n: int = 5, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        a = bulk("Cu", "fcc", a=3.61, cubic=True)
        a.rattle(stdev=0.02, seed=seed * 100 + i)
        energy = -3.5 * len(a) + float(rng.normal(0, 0.01))
        forces = rng.normal(0, 0.1, size=(len(a), 3))
        a.calc = SinglePointCalculator(a, energy=energy, forces=forces)
        frames.append(a)
    return frames


@pytest.fixture()
def synthetic_traj(tmp_path) -> Path:
    path = tmp_path / "synthetic5.traj"
    write(path, _make_frames(5, seed=0))
    return path


# ── extxyz-canonical: calculator AND REF_* agree ─────────────────────────────
@pytest.mark.parametrize("alias", ["mace", "sevennet", "extxyz"])
def test_extxyz_calculator_and_ref_keys_agree(tmp_path, synthetic_traj, alias):
    out = tmp_path / f"out_{alias}"
    manifest = ft_dataset.convert(
        [str(synthetic_traj)], ":", "energy", "forces", 0.8, 0, alias, out,
    )
    assert manifest["canonical_target"] == "extxyz-canonical"
    train = read(out / "train.xyz", index=":")
    assert len(train) == manifest["n_train"]
    for atoms in train:
        calc_e = atoms.get_potential_energy()
        calc_f = atoms.get_forces()
        assert atoms.info["REF_energy"] == pytest.approx(calc_e)
        assert np.allclose(atoms.arrays["REF_forces"], calc_f)


def test_conversion_manifest_written(tmp_path, synthetic_traj):
    out = tmp_path / "out"
    ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.8, 0, "mace", out)
    manifest = json.loads((out / "conversion.json").read_text())
    assert manifest["n_frames_total"] == 5
    assert manifest["n_train"] + manifest["n_valid"] == 5
    assert manifest["seed"] == 0
    assert manifest["elements"] == ["Cu"]


# ── split is seed-deterministic ──────────────────────────────────────────────
def test_split_is_seed_deterministic(tmp_path, synthetic_traj):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.8, 7, "mace", out_a)
    ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.8, 7, "mace", out_b)
    train_a = [a.get_potential_energy() for a in read(out_a / "train.xyz", index=":")]
    train_b = [a.get_potential_energy() for a in read(out_b / "train.xyz", index=":")]
    assert train_a == train_b


def test_empty_validation_split_fails_fast(tmp_path, synthetic_traj):
    """m3: 5 frames x 0.95 rounds to 5/0 -- refused here, not deep inside the
    trainer; --split 1.0 is the explicit opt-out and still writes no valid file."""
    with pytest.raises(SystemExit) as exc:
        ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.95, 0, "mace", tmp_path / "e")
    assert "EMPTY" in str(exc.value) and "--split 1.0" in str(exc.value)
    assert not (tmp_path / "e" / "train.xyz").exists()
    with pytest.raises(SystemExit) as exc:
        ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.0, 0, "mace", tmp_path / "z")
    assert "0 of 5 frames for training" in str(exc.value)
    m = ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 1.0, 0, "mace", tmp_path / "one")
    assert m["n_valid"] == 0 and m["outputs"]["valid"] is None and m["n_train"] == 5


def test_split_differs_across_seeds_deterministically(tmp_path, synthetic_traj):
    out_a = tmp_path / "seed0"
    out_b = tmp_path / "seed1"
    ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.6, 0, "mace", out_a)
    ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.6, 1, "mace", out_b)
    train_a = {a.get_potential_energy() for a in read(out_a / "train.xyz", index=":")}
    train_b = {a.get_potential_energy() for a in read(out_b / "train.xyz", index=":")}
    # Not asserting inequality unconditionally would be flaky in principle,
    # but with 5 frames split 3/2 two different seeds landing on the exact
    # same subset is exactly what we want to rule out here.
    assert train_a != train_b


# ── deepmd writer: numpy round-trip ──────────────────────────────────────────
def test_deepmd_writer_round_trips_energy_and_forces(tmp_path, synthetic_traj):
    out = tmp_path / "out_deepmd"
    manifest = ft_dataset.convert(
        [str(synthetic_traj)], ":", "energy", "forces", 0.8, 0, "deepmd", out,
    )
    assert manifest["canonical_target"] == "deepmd"
    train_systems = manifest["outputs"]["train_systems"]
    assert train_systems
    sys_dir = Path(train_systems[0])
    assert (sys_dir / "type_map.raw").read_text().strip().splitlines() == ["Cu"]
    coord = np.load(sys_dir / "set.000" / "coord.npy")
    energy = np.load(sys_dir / "set.000" / "energy.npy").reshape(-1)  # dpdata writes (nframes, 1)
    force = np.load(sys_dir / "set.000" / "force.npy")
    box = np.load(sys_dir / "set.000" / "box.npy")
    n_frames = manifest["n_train"]
    assert coord.shape[0] == energy.shape[0] == force.shape[0] == box.shape[0] == n_frames
    assert coord.shape[1] == force.shape[1] == 4 * 3  # 4-atom Cu cell
    assert box.shape[1] == 9

    # Cross-check against the original frames' own recorded energies.
    original = read(synthetic_traj, index=":")
    all_energies = sorted(a.get_potential_energy() for a in original)
    written_energies = sorted(energy.tolist())
    # train+valid should reconstruct the original energy set exactly.
    valid_systems = manifest["outputs"]["valid_systems"]
    valid_energies = []
    for sd in valid_systems:
        valid_energies.extend(np.load(Path(sd) / "set.000" / "energy.npy").reshape(-1).tolist())
    assert sorted(written_energies + valid_energies) == pytest.approx(all_energies)


# ── documented-not-implemented targets refuse cleanly (exit 2) ─────────────
@pytest.mark.parametrize("alias", ["orb", "nequix", "aselmdb", "alphanet"])
def test_not_implemented_targets_exit_2(tmp_path, synthetic_traj, alias):
    out = tmp_path / f"out_{alias}"
    with pytest.raises(SystemExit) as exc:
        ft_dataset.convert([str(synthetic_traj)], ":", "energy", "forces", 0.8, 0, alias, out)
    assert exc.value.code == 2


def test_unknown_target_is_a_usage_error(tmp_path, synthetic_traj):
    with pytest.raises(SystemExit):
        ft_dataset.canonical_target("not-a-real-target")


def test_chgnet_is_an_extxyz_alias_consumed_by_the_driver():
    """chgnet 0.4.0 has no on-disk training format; ft_run.py's driver reads
    the canonical extxyz and builds pymatgen Structures in memory."""
    assert ft_dataset.canonical_target("chgnet") == "extxyz-canonical"


# ── normalize_frame: calculator vs info/array fallback ───────────────────────
def test_stress_is_kept_on_the_calculator_ref_key_and_deepmd_virial(tmp_path):
    from ase.build import bulk
    frames = []
    for i in range(3):
        a = bulk("Cu", "fcc", a=3.61 + 0.01 * i, cubic=True)
        s = np.array([0.01, 0.02, 0.03, 0.004, 0.005, 0.006]) * (i + 1)
        a.calc = SinglePointCalculator(a, energy=-10.0 - i, forces=np.zeros((4, 3)), stress=s)
        frames.append(a)
    normalized = [ft_dataset.normalize_frame(a, "REF_energy", "REF_forces") for a in frames]
    assert np.allclose(normalized[1].get_stress(), frames[1].calc.results["stress"])
    path = tmp_path / "out.xyz"
    ft_dataset.write_extxyz_canonical(normalized, path)
    back = read(path, ":")
    assert np.allclose(back[2].get_stress(), frames[2].calc.results["stress"])
    assert np.allclose(back[2].info["REF_stress"], frames[2].calc.results["stress"])
    # the numpy deepmd writer stores virial = -volume * stress (3x3), as dpdata does
    ft_dataset._write_deepmd_system_numpy(normalized, ["Cu"], tmp_path / "sys")
    virial = np.load(tmp_path / "sys" / "set.000" / "virial.npy")
    s = frames[0].calc.results["stress"]
    s33 = np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
    assert np.allclose(virial[0], (-frames[0].get_volume() * s33).reshape(-1))


def test_frame_stress_reads_info_and_full_tensors():
    from ase.build import bulk
    a = bulk("Cu", "fcc", a=3.61, cubic=True)
    a.info["REF_stress"] = [[1, 6, 5], [6, 2, 4], [5, 4, 3]]
    assert np.allclose(ft_dataset.frame_stress(a), [1, 2, 3, 4, 5, 6])
    assert ft_dataset.frame_stress(bulk("Cu", "fcc", a=3.61)) is None


def test_normalize_frame_from_info_arrays_fallback():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    atoms.info["energy"] = -14.0
    atoms.new_array("forces", np.zeros((len(atoms), 3)))
    out = ft_dataset.normalize_frame(atoms, "energy", "forces")
    assert out.get_potential_energy() == pytest.approx(-14.0)
    assert np.allclose(out.get_forces(), 0.0)


def test_normalize_frame_missing_energy_raises():
    atoms = bulk("Cu", "fcc", a=3.61, cubic=True)
    with pytest.raises(SystemExit):
        ft_dataset.normalize_frame(atoms, "energy", "forces")
