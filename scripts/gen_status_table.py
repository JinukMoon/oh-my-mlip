#!/usr/bin/env python3
"""gen_status_table.py — generate the README `## Models & status` table.

Reads `models.json` and emits a deterministic markdown table (one row per
model+version) for the block between the stable markers

    <!-- STATUS_TABLE_START -->
    <!-- STATUS_TABLE_END -->

in `README.md`. The table is derived ENTIRELY from the registry's per-version
`validation` / `weights` / `gated` fields, so the README can never drift from
`models.json` (the source of truth). This guards against an
"all-validated" overclaim.

Modes:
  (default)   print the generated table to stdout.
  --check     compare the generated table against the current README block and
              exit non-zero if they differ (this is what CI calls).

No heavy imports: this script reads JSON directly and never imports torch / ase
/ the oh_my_mlip package, so it runs on any host.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = _REPO_ROOT / "models.json"
README = _REPO_ROOT / "README.md"

START_MARKER = "<!-- STATUS_TABLE_START -->"
END_MARKER = "<!-- STATUS_TABLE_END -->"

# Human-readable validation labels. The registry carries machine codes; the
# table renders them so a reader sees ship state at a glance.
_VALIDATION_LABEL = {
    "validated_sm86": "validated (sm86)",
    "validated_sm89": "validated (sm89)",
    "gpu_pending": "gpu pending",
    "cpu_only": "cpu only",
}


def _validation_label(code: str) -> str:
    return _VALIDATION_LABEL.get(code, code)


def _rows(models: dict) -> list[tuple[str, str, str, str, str, str]]:
    """Build (Model, Framework, Weights, Validation, Gated, v1 tarball) rows in a
    stable order: registry framework order, then version order within each
    framework. The `v1 tarball` column is taken from `_meta.shipped_v1` (the
    frameworks whose conda-pack distribution is authored for v1 = MACE +
    SevenNet); their tarballs are `upload-pending` the compute checkpoint, every
    other framework is a Phase-2 target. This is kept distinct from the per-model
    GPU `validation` state."""
    shipped = set(models.get("_meta", {}).get("shipped_v1", []))
    rows: list[tuple[str, str, str, str, str, str]] = []
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        for version, vinfo in info.get("versions", {}).items():
            rows.append(
                (
                    vinfo.get("mlip_name", version),
                    framework,
                    vinfo.get("weights", "bundled"),
                    _validation_label(vinfo.get("validation", "unknown")),
                    "yes" if vinfo.get("gated", False) else "no",
                    "upload-pending" if framework in shipped else "Phase 2",
                )
            )
    return rows


def render_table(models: dict) -> str:
    """Render the markdown table (no trailing newline)."""
    header = "| Model | Framework | Weights | Validation | Gated | v1 tarball |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for model, framework, weights, validation, gated, shipped in _rows(models):
        lines.append(
            f"| {model} | {framework} | {weights} | {validation} | "
            f"{gated} | {shipped} |"
        )
    return "\n".join(lines)


def _load_models() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def _readme_block() -> str:
    """Extract the current table text between the markers (markers excluded,
    surrounding blank lines stripped). Raises if the markers are missing."""
    text = README.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        raise SystemExit(
            f"README is missing the {START_MARKER} / {END_MARKER} markers"
        )
    inner = text.split(START_MARKER, 1)[1].split(END_MARKER, 1)[0]
    return inner.strip("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare against the README block; exit non-zero if they differ",
    )
    args = ap.parse_args(argv)

    table = render_table(_load_models())

    if not args.check:
        print(table)
        return 0

    current = _readme_block()
    if current == table:
        print("gen_status_table: README table is up to date.")
        return 0
    print(
        "gen_status_table: README `## Models & status` table is OUT OF DATE.\n"
        "Regenerate it: python scripts/gen_status_table.py > /tmp/t && "
        "paste between the markers in README.md\n",
        file=sys.stderr,
    )
    print("--- expected (generator) ---", file=sys.stderr)
    print(table, file=sys.stderr)
    print("--- found (README) ---", file=sys.stderr)
    print(current, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
