#!/usr/bin/env python3
"""Compile the per-GPU-arch NequIP / Allegro .pt2 that inference loads.

NequIPCalculator only loads via from_compiled_model, so a fresh build is not
usable until `nequip-compile` has produced
$OH_MY_MLIP_HOME/models/compiled/<arch>/<Model>_<arch>.nequip.pt2 on THIS GPU.
install.sh runs this with the ENV interpreter after the build
(`prepare_<env>_weights.py --target-root models/<env>`). The commands mirror
scripts/upstream_recipes.py (NequIP / Allegro fetch recipes), except that the
model package is downloaded here first (see MODELS) and compiled from disk.

Existing non-empty outputs are kept (per-arch artifacts; rerun after deleting).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _weight_download import download_first_available, is_complete  # noqa: E402

# (nequip.net URI, registry version name, extra nequip-compile args,
#  zenodo package file name, zenodo md5, zenodo size, sha256). nequip.net resolves each
# URI to the zenodo record file named here; md5 and size are zenodo's published
# values. The package is downloaded by this script (resumable, md5-checked) and
# compiled from the local file: nequip-compile's own download of the URI cannot
# resume, and zenodo can be slow enough for it to break partway.
#
# Sources, tried in order and all held to the same size and md5:
#  1. Zenodo record 10.5281/zenodo.18775904 ("NequIP & Allegro Foundation
#     Potentials", CC-BY-4.0, Kavanagh, S. R.; MIR Group @ Harvard), which holds
#     all three packages;
#  2. a byte-identical copy on Hugging Face, used only when Zenodo fails or
#     stays below MIN_RATE (the license permits redistribution with attribution;
#     the mirror's model card credits the authors and the record).
ZENODO = "https://zenodo.org/api/records/18775904/files/{name}/content"
MIRROR = "https://huggingface.co/JinukMoon/oh-my-mlip-mirror-nequip/resolve/main/{name}"
MIN_RATE = 50_000  # bytes/s averaged over a minute; below this, try the mirror
MODELS = {
    "nequip": [
        ("nequip.net:mir-group/NequIP-OAM-XL:0.1", "NequIP-OAM-XL", ["--modifiers", "enable_OpenEquivariance"],
         "NequIP-OAM-XL-0.1.nequip.zip", "3d2369c7238eb83a23141abdcb055a8f", 259627903,
         "99c3799b28026f1ecf66c413292038a27a0749d4d4d7cd0b3a642f2e68df9e9c"),
        ("nequip.net:mir-group/NequIP-OAM-L:0.1", "NequIP-OAM-L", ["--modifiers", "enable_OpenEquivariance"],
         "NequIP-OAM-L-0.1.nequip.zip", "67144367c710a70a53a8e21acf331980", 78464590,
         "5d01a4fab228abb3cdb6ace0033f93993729956bca6a42234a2a8816825b9a0f"),
    ],
    # Allegro's recommended kernel is CuEquivariance (upstream allegro docs,
    # "Inference with CuEquivariance"): AOT Inductor + enable_CuEquivarianceContracter.
    # AOTI with cueq does not support float64 models; the OAM checkpoints are float32.
    "allegro": [
        ("nequip.net:mir-group/Allegro-OAM-L:0.1", "Allegro-OAM-L", ["--modifiers", "enable_CuEquivarianceContracter"],
         "Allegro-OAM-L-0.1.nequip.zip", "0db7f9b3c3a62e74d78b3fcf2973c462", 80738705,
         "3f0d3ca7bb136d4c2ee76278170fcbe756436f037e16c673a86e2c57d271c64e"),
    ],
}


def local_package(zip_dir: str, name: str, md5: str) -> str | None:
    """A pre-downloaded zenodo package in OMM_NEQUIP_ZIP_DIR, accepted only when
    its md5 equals zenodo's published checksum (a partial or different file is
    a hard error, not a silent download)."""
    import hashlib

    path = Path(zip_dir) / name
    if not path.is_file():
        return None
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != md5:
        raise SystemExit(f"prepare_nequip_weights: {path} md5 {digest.hexdigest()} != zenodo {md5}")
    return str(path)


def host_arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def compile_env(prefix: Path) -> dict:
    env = dict(os.environ)
    # ninja for the openequivariance JIT build must come from the env's bin.
    env["PATH"] = str(prefix / "bin") + os.pathsep + env.get("PATH", "")
    # MAX_JOBS=4: oeq's first import JIT-builds at nproc and can wedge on a stale lock.
    env.setdefault("MAX_JOBS", "4")
    # nvrtc.h (and other CUDA headers) ship in the env's pip nvidia-* wheels.
    # crt/host_defines.h etc. come from the conda CUDA toolkit (<env>/targets/*/include).
    includes = (sorted(glob.glob(str(prefix / "lib" / "python3*" / "site-packages" / "nvidia" / "*" / "include")))
                + sorted(glob.glob(str(prefix / "targets" / "*" / "include"))))
    if includes:
        env["CPATH"] = os.pathsep.join(includes) + (os.pathsep + env["CPATH"] if env.get("CPATH") else "")
    # -lcuda / -lnvrtc at link time: conda CUDA libs + the libcuda link stub.
    libs = (sorted(glob.glob(str(prefix / "targets" / "*" / "lib")))
            + sorted(glob.glob(str(prefix / "targets" / "*" / "lib" / "stubs"))))
    if libs:
        env["LIBRARY_PATH"] = os.pathsep.join(libs) + (os.pathsep + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else "")
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target-root", required=True, help="$OH_MY_MLIP_HOME/models/<env> (env name = basename)")
    ap.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = ap.parse_args(argv)

    target_root = Path(args.target_root).resolve()
    env_name = target_root.name
    if env_name not in MODELS:
        print(f"prepare_nequip_weights: no compile recipe for env {env_name!r}", file=sys.stderr)
        return 2
    home = target_root.parent.parent
    prefix = Path(sys.executable).resolve().parent.parent
    compiler = prefix / "bin" / "nequip-compile"
    arch = host_arch() if not args.dry_run else "${ARCH}"
    out_dir = home / "models" / "compiled" / arch
    env = compile_env(prefix)

    zip_dir = os.environ.get("OMM_NEQUIP_ZIP_DIR")
    failed = 0
    for uri, version, extra, zip_name, md5, size, sha256 in MODELS[env_name]:
        out = out_dir / f"{version}_{arch}.nequip.pt2"
        if not args.dry_run and out.is_file() and out.stat().st_size > 0:
            print(f"  {version}: {out} already present, kept")
            continue
        source = local_package(zip_dir, zip_name, md5) if zip_dir else None
        if source is None:
            package = target_root / zip_name
            if not args.dry_run and not is_complete(package, size):
                sources = [("Zenodo", ZENODO.format(name=zip_name)),
                           ("the oh-my-mlip mirror on Hugging Face", MIRROR.format(name=zip_name))]
                print(f"  downloading {zip_name} -> {package}", flush=True)
                try:
                    used = download_first_available(sources, package, size=size, md5=md5,
                                                    sha256=sha256, label=version, min_rate=MIN_RATE)
                    print(f"  {version}: package from {used}", flush=True)
                except Exception as exc:  # noqa: BLE001 - reported, the other models still run
                    print(f"  {version}: download failed ({exc}); rerun to resume it, or set "
                          f"OMM_NEQUIP_ZIP_DIR to a directory holding {zip_name}", file=sys.stderr)
                    failed += 1
                    continue
            source = str(package)
        cmd = [str(compiler), source, str(out), "--mode", "aotinductor", "--device", "cuda", "--target", "ase", *extra]
        if args.dry_run:
            print(" ".join(cmd))
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  compiling {version} for {arch} -> {out}", flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0 or not (out.is_file() and out.stat().st_size > 0):
            print(f"  {version}: nequip-compile failed (rc={rc})", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
