#!/usr/bin/env python3
"""Compile the per-GPU-arch MACE-SOYO .pt2 that inference loads.

MACESoyoCalculator loads only an AOTInductor package (.pt2) compiled for one GPU
architecture, not the released training checkpoint (.pth). install.sh runs this
with the ENV interpreter after the build (`prepare_mace_soyo_weights.py
--target-root models/mace_soyo`):

  1. download mace_soyo_oam.pth from the upstream GitHub release to
     <target_root>/, checked against its size and sha256;
  2. write the export example structure next to it;
  3. `python -m mace_soyo.export.AOTI_export` (upstream README) into
     $OH_MY_MLIP_HOME/models/compiled/<arch>/MACE-SOYO-OAM_<arch>.pt2.

The node and edge bounds are left at upstream's default (None, no upper bound),
so one package serves gas molecules and slabs alike.

Existing non-empty outputs are kept (per-arch artifacts; rerun after deleting).
On a host without a usable GPU the checkpoint is downloaded and the export is
left for a GPU node: the script prints the command to run there and exits 4
(DEFERRED), which install.sh accepts. The export reads only the local files, so
it works on an offline GPU node.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _weight_download import download_first_available, is_complete  # noqa: E402
from prepare_nequip_weights import DEFERRED, compile_env, host_arch  # noqa: E402

VERSION = "MACE-SOYO-OAM"
CKPT_NAME = "mace_soyo_oam.pth"
CKPT_URL = "https://github.com/PrincessHakuryu/MACE-SOYO/releases/download/v0.1/mace_soyo_oam.pth"
# The release asset as GitHub serves it (not published upstream; measured on the
# download and identical on two hosts).
CKPT_SIZE = 25960795
CKPT_SHA256 = "180784bad2fdee6aec58dae2cdb0d581ad1c4f3d2272d2857bce6315ea43b82b"
EXAMPLE_NAME = "export_example_pt111_h2o.xyz"


def fetch_checkpoint(target_root: Path) -> Path:
    ckpt = target_root / CKPT_NAME
    if not is_complete(ckpt, CKPT_SIZE):
        target_root.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {CKPT_NAME} -> {ckpt}", flush=True)
        download_first_available([("the MACE-SOYO GitHub release", CKPT_URL)], ckpt,
                                 size=CKPT_SIZE, sha256=CKPT_SHA256, label=VERSION)
    return ckpt


def write_example(target_root: Path) -> Path:
    """The export traces one realistic structure: Pt(111) 3x3x4 with H2O on top."""
    from ase.build import add_adsorbate, fcc111, molecule
    from ase.io import write

    path = target_root / EXAMPLE_NAME
    if not path.is_file():
        slab = fcc111("Pt", size=(3, 3, 4), vacuum=10.0)
        add_adsorbate(slab, molecule("H2O"), 2.2, "ontop")
        write(path, slab)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-root", required=True, help="$OH_MY_MLIP_HOME/models/mace_soyo")
    ap.add_argument("--dry-run", action="store_true", help="print the command only")
    args = ap.parse_args(argv)

    target_root = Path(args.target_root).resolve()
    home = target_root.parent.parent
    prefix = Path(sys.executable).resolve().parent.parent
    arch = host_arch() if not args.dry_run else "${ARCH}"
    ckpt = target_root / CKPT_NAME
    if not args.dry_run:
        try:
            ckpt = fetch_checkpoint(target_root)
        except Exception as exc:  # noqa: BLE001 - reported with the rerun hint
            print(f"  {VERSION}: download failed ({exc}); rerun to resume it", file=sys.stderr)
            return 1
    if arch is None:
        cmd = f"{sys.executable} {Path(__file__).resolve()} --target-root {target_root}"
        print(f"  no usable GPU on this host: {CKPT_NAME} downloaded, not exported. MACE-SOYO loads a "
              f"model compiled for one GPU architecture, so export it on a GPU node (no internet needed "
              f"there), e.g. as the first step of the GPU job:\n    {cmd}", flush=True)
        return DEFERRED

    out_dir = home / "models" / "compiled" / arch
    out = out_dir / f"{VERSION}_{arch}.pt2"
    if not args.dry_run and out.is_file() and out.stat().st_size > 0:
        print(f"  {VERSION}: {out} already present, kept")
        return 0
    example = target_root / EXAMPLE_NAME if args.dry_run else write_example(target_root)
    cmd = [sys.executable, "-m", "mace_soyo.export.AOTI_export", "--ckpt-path", str(ckpt),
           "--structure-file", str(example), "--output-path", str(out),
           "--device", "cuda:0", "--dtype", "float32"]
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  exporting {VERSION} for {arch} -> {out} (several minutes)", flush=True)
    rc = subprocess.run(cmd, env=compile_env(prefix), cwd=target_root).returncode
    if rc != 0 or not (out.is_file() and out.stat().st_size > 0):
        # a failed export leaves nothing behind: a partial .pt2 would look present
        leftovers = [p for p in out_dir.glob(f"{VERSION}_{arch}*") if p.is_file()]
        for path in leftovers:
            path.unlink(missing_ok=True)
        note = f"; removed {len(leftovers)} partial file(s)" if leftovers else ""
        print(f"  {VERSION}: AOTI export failed (rc={rc}){note}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
