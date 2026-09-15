#!/usr/bin/env python3
"""ft_dataset.py — one converter, canonical extxyz in, per-framework layout out.

This is the hub's core added value for fine-tuning: a SINGLE canonical
extxyz writer feeds five frameworks at once. MACE defaults to `REF_energy`/
`REF_forces` in `info`/`arrays` and does NOT read the `SinglePointCalculator`;
SevenNet, NequIP, Allegro and GRACE (plus MatterSim/PET/TACE, which also read
the calculator) do. Writing a `SinglePointCalculator` AND duplicating the same
numbers into `REF_energy`/`REF_forces` on the same file satisfies all of them
from one artifact — no framework-specific re-export.

Pipeline: read (anything ``ase.io.read`` handles, one or more files) ->
normalize every frame to a plain ``Atoms`` + ``SinglePointCalculator`` (energy
and forces resolved either from an existing calculator or from
``--energy-key``/``--force-key`` info/array entries) -> deterministic
train/valid split (``random.Random(seed)``, so re-running with the same
``--seed`` reproduces the same split byte-for-byte) -> write the requested
target. A ``conversion.json`` manifest records inputs, counts, keys and seed
next to the written files so the split is auditable, not just reproducible.

Target table (every target this hub's fine-tune path could need is named
here, whether or not v1 implements it — an unimplemented target must read as
unimplemented, never as silently absent):

  Target              Format                                          v1 status
  ------------------  ----------------------------------------------  --------------------------
  extxyz-canonical    calculator + REF_energy/REF_forces (info/       IMPLEMENTED
                       arrays) -- feeds MACE, SevenNet, NequIP,
                       Allegro, GRACE, MatterSim, PET, TACE, and
                       CHGNet (whose python-API driver emitted by
                       ft_run.py converts each frame to a pymatgen
                       Structure + eV/atom in memory -- chgnet 0.4.0
                       has no on-disk training format of its own)
  deepmd              npy system layout (type_map.raw/type.raw +      IMPLEMENTED (demo path)
                       set.000/{coord,box,energy,force}.npy) --
                       feeds DeePMD, DPA4. Uses ``dpdata`` when it is
                       importable in the interpreter running THIS
                       script; otherwise a direct numpy writer
                       produces the identical on-disk layout without
                       the dependency (dpdata was not installed in
                       any local env at the time this was written --
                       see scripts/upstream_finetune.py's DeePMD/DPA4
                       blockers).
  orb                 ASE sqlite ``.db``, 1-indexed dense ids         documented, NOT implemented
  aselmdb             UMA / EquiformerV3 / Nequix                     documented, NOT implemented
  alphanet            custom pickle, needs ``virial`` in `atoms.info` documented, NOT implemented

Requesting one of the three not-implemented targets exits 2 with a message
naming the target and pointing back at this table -- it never silently no-ops
or falls back to a different target.

Usage:
  python3 scripts/ft_dataset.py --input traj.traj --to mace --out ft_mace/data
  python3 scripts/ft_dataset.py --input a.xyz b.xyz --index ":" \\
      --energy-key energy --force-key forces --split 0.9 --seed 0 \\
      --to deepmd --out ft_deepmd/data
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

# ── target aliasing ──────────────────────────────────────────────────────────
# Family/framework names the fine-tuning skill and ft_run.py speak, mapped to
# the ONE canonical writer that satisfies them. Keeping the alias table here
# (not duplicated in ft_run.py) is what makes "one canonical extxyz writer
# feeds five frameworks" a fact about the code, not just prose.
_EXTXYZ_ALIASES = {
    "extxyz", "extxyz-canonical", "mace", "sevennet", "nequip", "allegro",
    "grace", "mattersim", "pet", "tace",
    # chgnet: ft_run.py's driver converts the canonical extxyz to pymatgen
    # Structures + eV/atom in memory (no on-disk chgnet format exists)
    "chgnet",
}
_DEEPMD_ALIASES = {"deepmd", "dpa4"}
_NOT_IMPLEMENTED = {
    "orb": "ASE sqlite .db, 1-indexed dense ids",
    "nequix": "aselmdb (UMA / EquiformerV3 / Nequix)",
    "aselmdb": "aselmdb (UMA / EquiformerV3 / Nequix)",
    "alphanet": "custom pickle, needs 'virial' in atoms.info",
}

TARGET_CHOICES = sorted(_EXTXYZ_ALIASES | _DEEPMD_ALIASES | set(_NOT_IMPLEMENTED))


def canonical_target(name: str) -> str:
    """Map any accepted --to alias to 'extxyz-canonical' or 'deepmd'.

    Raises SystemExit(2) for a documented-not-implemented target -- this is
    the ONLY place that decision is made, so a caller can never silently fall
    through to the wrong writer.
    """
    key = name.lower()
    if key in _EXTXYZ_ALIASES:
        return "extxyz-canonical"
    if key in _DEEPMD_ALIASES:
        return "deepmd"
    if key in _NOT_IMPLEMENTED:
        print(
            f"[ft_dataset] --to {name!r} is documented but NOT implemented in v1 "
            f"({_NOT_IMPLEMENTED[key]}). See the target table in this module's "
            f"docstring; convert manually or use --to extxyz/deepmd instead.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(f"[ft_dataset] unknown --to target {name!r}; choices: {TARGET_CHOICES}")


# ── read + normalize ─────────────────────────────────────────────────────────
def read_frames(inputs: list[str], index: str) -> list[Atoms]:
    frames: list[Atoms] = []
    for path in inputs:
        obj = read(path, index=index)
        frames.extend(obj if isinstance(obj, list) else [obj])
    if not frames:
        raise SystemExit(f"[ft_dataset] no frames read from {inputs!r}")
    return frames


def normalize_frame(atoms: Atoms, energy_key: str, force_key: str) -> Atoms:
    """One frame -> a fresh Atoms carrying a SinglePointCalculator.

    Energy/forces come from an existing calculator when present (the common
    case for a teacher-labeled .traj), else from ``atoms.info[energy_key]`` /
    ``atoms.arrays[force_key]`` (the common case for a hand-built extxyz).
    """
    energy = None
    forces = None
    if atoms.calc is not None:
        try:
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        except Exception:
            energy = None
            forces = None
    if energy is None:
        energy = atoms.info.get(energy_key)
    if forces is None:
        forces = atoms.arrays.get(force_key)
    if energy is None or forces is None:
        raise SystemExit(
            f"[ft_dataset] frame missing energy/forces -- no calculator result and "
            f"no info[{energy_key!r}]/arrays[{force_key!r}] present"
        )
    out = Atoms(
        numbers=atoms.numbers.copy(),
        positions=atoms.positions.copy(),
        cell=atoms.cell.copy(),
        pbc=atoms.pbc.copy(),
    )
    stress = frame_stress(atoms)
    results = {"energy": float(energy), "forces": np.asarray(forces, dtype=float)}
    if stress is not None:
        results["stress"] = stress
    out.calc = SinglePointCalculator(out, **results)
    return out


def frame_stress(atoms: Atoms) -> np.ndarray | None:
    """The frame's stress as ASE Voigt-6 (xx, yy, zz, yz, xz, xy; eV/A^3, ASE sign),
    from the calculator when it has one, else from ``info["REF_stress"]`` or
    ``info["stress"]`` (6 or 9 components); None when the frame carries none."""
    stress = None
    if atoms.calc is not None and "stress" in (getattr(atoms.calc, "results", None) or {}):
        stress = atoms.calc.results["stress"]
    else:
        for key in ("REF_stress", "stress"):
            if key in atoms.info:
                stress = atoms.info[key]
                break
    if stress is None:
        return None
    stress = np.asarray(stress, dtype=float).reshape(-1)
    if stress.size == 9:
        s = stress.reshape(3, 3)
        stress = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
    if stress.size != 6:
        raise SystemExit(f"[ft_dataset] stress with {stress.size} components; expected 6 (Voigt) or 9 (3x3)")
    return stress


def deterministic_split(n: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Shuffle indices with a seeded RNG, then cut at `fraction` -- the same
    `n`/`fraction`/`seed` always produces the same two index lists."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_train = round(n * fraction)
    return sorted(idx[:n_train]), sorted(idx[n_train:])


# ── extxyz-canonical writer ──────────────────────────────────────────────────
def write_extxyz_canonical(frames: list[Atoms], path: Path) -> None:
    """Attach the calculator's numbers into info/arrays REF_* keys too, on the
    SAME atoms object the calculator is attached to -- one file, both readers."""
    out_frames = []
    for atoms in frames:
        a = atoms.copy()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = frame_stress(atoms)
        results = {"energy": float(energy), "forces": forces}
        if stress is not None:
            results["stress"] = stress
        a.calc = SinglePointCalculator(a, **results)
        a.info["REF_energy"] = float(energy)
        a.new_array("REF_forces", np.asarray(forces, dtype=float))
        if stress is not None:   # MACE reads REF_stress; the calculator serves the other readers
            a.info["REF_stress"] = stress
        out_frames.append(a)
    write(path, out_frames, format="extxyz")


# ── deepmd writer ─────────────────────────────────────────────────────────────
def _group_by_composition(frames: list[Atoms]) -> dict[tuple[int, ...], list[Atoms]]:
    """Group frames that share an identical (ordered) atomic-number sequence --
    a deepmd 'system' needs the same atom count AND identity/order in every
    frame. Simplification, documented here: frames whose composition matches
    but whose ATOM ORDER differs land in different systems (order is not
    canonicalized)."""
    groups: dict[tuple[int, ...], list[Atoms]] = {}
    for atoms in frames:
        key = tuple(int(z) for z in atoms.numbers)
        groups.setdefault(key, []).append(atoms)
    return groups


def _write_deepmd_system_numpy(frames: list[Atoms], type_map: list[str], sys_dir: Path) -> None:
    sys_dir.mkdir(parents=True, exist_ok=True)
    symbol_to_idx = {s: i for i, s in enumerate(type_map)}
    natoms = len(frames[0])
    type_idx = [symbol_to_idx[s] for s in frames[0].get_chemical_symbols()]
    (sys_dir / "type_map.raw").write_text("\n".join(type_map) + "\n")
    (sys_dir / "type.raw").write_text("\n".join(str(i) for i in type_idx) + "\n")

    coord = np.stack([f.get_positions().reshape(-1) for f in frames]).astype(np.float64)
    box = np.stack([np.asarray(f.get_cell()).reshape(-1) for f in frames]).astype(np.float64)
    # Shape (nframes, 1), matching dpdata's own `to_deepmd_npy` output -- the
    # two writer paths must agree on-disk regardless of which one ran.
    energy = np.array([f.get_potential_energy() for f in frames], dtype=np.float64).reshape(-1, 1)
    force = np.stack([f.get_forces().reshape(-1) for f in frames]).astype(np.float64)
    assert coord.shape == (len(frames), natoms * 3)

    set_dir = sys_dir / "set.000"
    set_dir.mkdir(exist_ok=True)
    np.save(set_dir / "coord.npy", coord)
    np.save(set_dir / "box.npy", box)
    np.save(set_dir / "energy.npy", energy)
    np.save(set_dir / "force.npy", force)
    # virial = -volume * stress (3x3), as dpdata's ase plugin converts it; only when every
    # frame of the system carries stress
    stresses = [frame_stress(f) for f in frames]
    if all(s is not None for s in stresses):
        virial = []
        for f, s in zip(frames, stresses):
            s33 = np.array([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])
            virial.append((-f.get_volume() * s33).reshape(-1))
        np.save(set_dir / "virial.npy", np.stack(virial).astype(np.float64))


def _write_deepmd_system_dpdata(frames: list[Atoms], type_map: list[str], sys_dir: Path) -> bool:
    """Best-effort dpdata path; returns False (caller falls back to the numpy
    writer) if dpdata is not importable in THIS interpreter, or the
    conversion raises. dpdata's 'ase/structure' format takes a live
    ``ase.Atoms`` object directly (not a file path) and one frame at a time;
    frames are concatenated with dpdata's own System `+` (passing a written extxyz PATH under fmt='ase/structure'
    does not work: dpdata does not accept it)."""
    try:
        import dpdata
    except ImportError:
        return False
    try:
        combined = None
        for atoms in frames:
            frame_sys = dpdata.LabeledSystem(atoms, fmt="ase/structure", type_map=type_map)
            combined = frame_sys if combined is None else combined + frame_sys
        combined.to_deepmd_npy(str(sys_dir), set_size=len(frames))
        return True
    except Exception:
        return False


def write_deepmd(frames: list[Atoms], out_dir: Path, type_map: list[str]) -> list[str]:
    groups = _group_by_composition(frames)
    written = []
    for i, group_frames in enumerate(groups.values()):
        sys_dir = out_dir / f"system_{i:03d}"
        if not _write_deepmd_system_dpdata(group_frames, type_map, sys_dir):
            _write_deepmd_system_numpy(group_frames, type_map, sys_dir)
        written.append(str(sys_dir))
    return written


# ── CLI ───────────────────────────────────────────────────────────────────────
def convert(
    inputs: list[str],
    index: str,
    energy_key: str,
    force_key: str,
    split: float,
    seed: int,
    to: str,
    out: Path,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = canonical_target(to)

    raw_frames = read_frames(inputs, index)
    frames = [normalize_frame(a, energy_key, force_key) for a in raw_frames]
    train_idx, valid_idx = deterministic_split(len(frames), split, seed)
    # Fail here, not inside the trainer: an empty split is handed on as
    # "no validation file" and several trainers (NequIP's unconditional
    # val-metric monitor, SevenNet's checkpoint_best) then die or never
    # write the designated checkpoint. A caller who wants no validation
    # says so with --split 1.0.
    if not train_idx:
        raise SystemExit(f"[ft_dataset] --split {split} leaves 0 of {len(frames)} frames for training")
    if not valid_idx and split < 1.0:
        raise SystemExit(f"[ft_dataset] --split {split} of {len(frames)} frames rounds to an EMPTY "
                         f"validation set -- supply more frames, lower --split, or pass --split 1.0 "
                         f"to opt out of validation explicitly")
    train_frames = [frames[i] for i in train_idx]
    valid_frames = [frames[i] for i in valid_idx]
    elements = sorted({s for a in frames for s in a.get_chemical_symbols()})

    manifest: dict = {
        "inputs": inputs,
        "index": index,
        "energy_key": energy_key,
        "force_key": force_key,
        "n_frames_total": len(frames),
        "n_train": len(train_frames),
        "n_valid": len(valid_frames),
        "split": split,
        "seed": seed,
        "to": to,
        "canonical_target": target,
        "elements": elements,
    }

    if target == "extxyz-canonical":
        train_path = out / "train.xyz"
        write_extxyz_canonical(train_frames, train_path)
        # A 0-frame validation file would be handed to the trainer and fail
        # deep inside it -- mirror the deepmd branch: no frames, no file.
        valid_path = None
        if valid_frames:
            valid_path = out / "valid.xyz"
            write_extxyz_canonical(valid_frames, valid_path)
        manifest["outputs"] = {"train": str(train_path),
                               "valid": str(valid_path) if valid_path else None}
    elif target == "deepmd":
        train_systems = write_deepmd(train_frames, out / "train_systems", elements)
        valid_systems = write_deepmd(valid_frames, out / "valid_systems", elements) if valid_frames else []
        manifest["outputs"] = {"train_systems": train_systems, "valid_systems": valid_systems}
    else:  # pragma: no cover -- canonical_target() only returns the two above
        raise SystemExit(f"[ft_dataset] internal error: unhandled canonical target {target!r}")

    (out / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", nargs="+", required=True, help="one or more ase.io.read-able files")
    ap.add_argument("--index", default=":", help="ase.io.read index/slice per input file (default: ':' = all frames)")
    ap.add_argument("--energy-key", default="energy", help="atoms.info key when no calculator is attached")
    ap.add_argument("--force-key", default="forces", help="atoms.arrays key when no calculator is attached")
    ap.add_argument("--split", type=float, default=0.9, help="train fraction (default 0.9)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for the train/valid shuffle (default 0)")
    ap.add_argument("--to", required=True, choices=TARGET_CHOICES, help="target framework/format")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    args = ap.parse_args()

    manifest = convert(
        args.input, args.index, args.energy_key, args.force_key,
        args.split, args.seed, args.to, args.out,
    )
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
