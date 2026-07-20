#!/usr/bin/env bash
# release_env.sh — AUTHOR-SIDE deterministic release driver for ONE env's
# conda-pack tarball: rebuild -> pack -> LOCAL RELOCATION GATE -> publish.
#
# This encodes the exact author choreography (host-proven 2026-07-20) so a
# release is one command with hard gates, not session improvisation:
#   1. install.sh <env>           clean recipe build (adopt-or-heal safe)
#   2. build_conda_pack.sh        strip arch artifacts, emit tarball+sidecars
#   3. RELOCATION GATE            unpack to a temp prefix, conda-unpack, then
#                                 run the registry's own import+inference lines
#                                 (resolve() codegen) with the UNPACKED
#                                 interpreter — energy must be finite. A
#                                 tarball that fails here is NEVER uploaded.
#   4. publish_hf.py              upload + revision tag + dist_manifest entry
#                                 (write-credential guard lives in that script:
#                                 HF_TOKEN -> HF_TOKEN_PATH -> login cache)
#
# Usage:
#   scripts/release_env.sh <env> <model> <hf-repo> <revision> [--dry-run]
#     env        recipe/env name (e.g. mace)
#     model      registry family for the gate compute (e.g. MACE)
#     hf-repo    e.g. JinukMoon/oh-my-mlip-env-mace
#     revision   immutable tag to pin (e.g. v1); never 'main'
#     --dry-run  print the exact commands and exit without running them
#
# Requires: conda-pack on PATH (pack step), a GPU (gate step), and a
# write-capable HF credential (publish step). Runs from the hub root.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <env> <model> <hf-repo> <revision> [--dry-run]" >&2
  exit 2
fi
ENV_NAME="$1"; MODEL="$2"; HF_REPO="$3"; REVISION="$4"
DRY="${5:-}"

HUB="$(cd "$(dirname "$0")/.." && pwd)"
export OH_MY_MLIP_HOME="$HUB"
DIST="$HUB/dist"
GATE_DIR="${OMM_RELEASE_GATE_DIR:-$HUB/dist/.reloc_gate}/$ENV_NAME"

STEPS=(
  "$HUB/install.sh $ENV_NAME"
  "bash $HUB/scripts/build_conda_pack.sh $HUB/envs/$ENV_NAME $ENV_NAME $DIST"
  "# relocation gate: unpack + conda-unpack + registry-codegen compute (below)"
  "python3 $HUB/scripts/publish_hf.py --env $ENV_NAME --tarball $DIST/$ENV_NAME.tar.gz --hf-repo $HF_REPO --revision $REVISION --manifest $HUB/dist_manifest.json"
)
if [ "$DRY" = "--dry-run" ]; then
  printf '%s\n' "${STEPS[@]}"
  exit 0
fi

echo "== release[$ENV_NAME] 1/4 clean recipe build"
"$HUB/install.sh" "$ENV_NAME"

echo "== release[$ENV_NAME] 2/4 conda-pack"
bash "$HUB/scripts/build_conda_pack.sh" "$HUB/envs/$ENV_NAME" "$ENV_NAME" "$DIST"

echo "== release[$ENV_NAME] 3/4 RELOCATION GATE (unpack + compute with the unpacked interpreter)"
rm -rf "$GATE_DIR" && mkdir -p "$GATE_DIR"
tar -xzf "$DIST/$ENV_NAME.tar.gz" -C "$GATE_DIR"
"$GATE_DIR/bin/python" "$GATE_DIR/bin/conda-unpack" 2>/dev/null || "$GATE_DIR/bin/conda-unpack"
# Generate the compute check from the registry itself (resolve() codegen —
# the SAME verbatim import+inference lines every user gets), executed with
# the UNPACKED interpreter instead of the resolved one.
python3 - "$MODEL" > "$GATE_DIR/_gate_check.py" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["OH_MY_MLIP_HOME"])
from oh_my_mlip import resolve
spec = resolve(sys.argv[1])
lines = (
    ["from ase.build import bulk"]
    + spec["imports"]
    + spec["inference"]
    + [
        "atoms = bulk('Cu', 'fcc', a=3.61, cubic=True)",
        "atoms.calc = calc",
        "e = atoms.get_potential_energy()",
        "import math; assert math.isfinite(e), 'non-finite energy'",
        "print('RELOCATION_GATE_PASS energy_ev=%.6f' % e)",
    ]
)
sys.stdout.write("\n".join(lines) + "\n")
PYEOF
"$GATE_DIR/bin/python" "$GATE_DIR/_gate_check.py"

echo "== release[$ENV_NAME] 4/4 publish + manifest"
python3 "$HUB/scripts/publish_hf.py" \
  --env "$ENV_NAME" --tarball "$DIST/$ENV_NAME.tar.gz" \
  --hf-repo "$HF_REPO" --revision "$REVISION" \
  --manifest "$HUB/dist_manifest.json"

rm -rf "$GATE_DIR"
echo "== release[$ENV_NAME] DONE — commit dist_manifest.json to ship the pin."
