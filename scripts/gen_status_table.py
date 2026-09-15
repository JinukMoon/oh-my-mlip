#!/usr/bin/env python3
"""gen_status_table.py — generate the supported-MLIPs lists from `models.json`.

`models.json` is the single source of truth. This script renders two views of it
and keeps both byte-for-byte in sync (CI runs `--check`):

  1. A LIGHT `| Framework | Models |` list for the README `## Supported MLIPs`
     section, between the stable markers

         <!-- STATUS_TABLE_START -->
         <!-- STATUS_TABLE_END -->

     in `README.md`. One row per framework, variant names comma-joined — just
     enough to see what is available, scannable and short.

  2. The FULL detailed table (Model / Framework / Weights / Gated / v1 tarball /
     Code — one row per model+version) for `docs/model_status.md`, between

         <!-- STATUS_TABLE_DETAILED_START -->
         <!-- STATUS_TABLE_DETAILED_END -->

Both are derived ENTIRELY from the registry (`weights` / `gated` per version,
`dist_manifest.json` for tarballs), so neither view can drift from
`models.json`. The per-version `validation` field stays in the registry (and in
the MCP `model_status` tool) but is not rendered in the public docs.

Modes:
  (default)   print both generated blocks to stdout (clearly delimited).
  --check     compare each generated block against its committed file and exit
              non-zero if EITHER differs (this is what CI calls).

No heavy imports: this script reads JSON directly and never imports torch / ase
/ the oh_my_mlip package, so it runs on any host.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from oh_my_mlip import registry  # noqa: E402  (stdlib-only module, not a heavy import)

_REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_JSON = _REPO_ROOT / "models.json"
README = _REPO_ROOT / "README.md"
MODEL_STATUS_DOC = _REPO_ROOT / "docs" / "model_status.md"

# Light README list (Framework | Models).
START_MARKER = "<!-- STATUS_TABLE_START -->"
END_MARKER = "<!-- STATUS_TABLE_END -->"

# Detailed per-version table in docs/model_status.md.
DETAILED_START_MARKER = "<!-- STATUS_TABLE_DETAILED_START -->"
DETAILED_END_MARKER = "<!-- STATUS_TABLE_DETAILED_END -->"

def _detailed_rows(models: dict) -> list[tuple[str, str, str, str, str]]:
    """Build (Model, Framework, Weights, Gated, v1 tarball) rows in a stable
    order: registry framework order, then version order within each framework.
    The `v1 tarball` column is derived from dist_manifest.json via
    registry.published_envs() — `published (<rev>)` when the tarball is
    actually live on the Hub, `upload-pending` for v1-authored frameworks
    (`_meta.shipped_v1`) not yet uploaded, `Phase 2` otherwise."""
    shipped = set(models.get("_meta", {}).get("shipped_v1", []))
    published = registry.published_envs()
    rows: list[tuple[str, str, str, str, str]] = []
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        for version, vinfo in info.get("versions", {}).items():
            rows.append(
                (
                    vinfo.get("mlip_name", version),
                    framework,
                    vinfo.get("weights", "bundled"),
                    "yes" if vinfo.get("gated", False) else "no",
                    f"published ({published[info['env']]})"
                    if info.get("env") in published
                    else ("upload-pending" if framework in shipped else "Phase 2"),
                )
            )
    return rows


def _framework_rows(models: dict) -> list[tuple[str, str]]:
    """Build (Framework, comma-joined model variant names) rows in registry
    order — one row per framework. This is the LIGHT view for the README."""
    rows: list[tuple[str, str]] = []
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        names = [
            vinfo.get("mlip_name", version)
            for version, vinfo in info.get("versions", {}).items()
        ]
        rows.append((framework, ", ".join(names)))
    return rows


def render_simple_list(models: dict) -> str:
    """Render the light README `| Framework | Models |` table (no trailing
    newline)."""
    lines = ["| Framework | Models |", "|---|---|"]
    for framework, names in _framework_rows(models):
        lines.append(f"| {framework} | {names} |")
    return "\n".join(lines)


def render_detailed_table(models: dict) -> str:
    """Render the full detailed markdown table (no trailing newline). The last
    column links each framework's upstream repository (`repo` in models.json)
    as a small button: an `attr_list` class styled in docs/stylesheets/extra.css."""
    repos = {fw: info.get("repo") for fw, info in models.items() if not fw.startswith("_")}
    header = "| Model | Framework | Weights | Gated | v1 tarball | Code |"
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for model, framework, weights, gated, shipped in _detailed_rows(models):
        repo = repos.get(framework)
        code = f"[GitHub]({repo}){{ .md-button .omm-repo }}" if repo else "-"
        lines.append(f"| {model} | {framework} | {weights} | {gated} | {shipped} | {code} |")
    return "\n".join(lines)


def _load_models() -> dict:
    return json.loads(MODELS_JSON.read_text(encoding="utf-8"))


def _extract_block(path: Path, start: str, end: str, what: str) -> str:
    """Extract the text between markers (markers excluded, surrounding blank
    lines stripped) in `path`. Raises if the file or markers are missing."""
    if not path.exists():
        raise SystemExit(f"{what}: file {path} does not exist")
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise SystemExit(f"{what} is missing the {start} / {end} markers")
    inner = text.split(start, 1)[1].split(end, 1)[0]
    return inner.strip("\n")


def _check_block(path: Path, start: str, end: str, expected: str, what: str) -> bool:
    """Return True if the committed block in `path` matches `expected`, else
    print a diagnostic and return False."""
    current = _extract_block(path, start, end, what)
    if current == expected:
        print(f"gen_status_table: {what} is up to date.")
        return True
    print(
        f"gen_status_table: {what} is OUT OF DATE.\n"
        "Regenerate it: python scripts/gen_status_table.py "
        "(see CONTRIBUTING.md for which block goes where)\n",
        file=sys.stderr,
    )
    print(f"--- expected (generator) [{what}] ---", file=sys.stderr)
    print(expected, file=sys.stderr)
    print(f"--- found ({path.name}) [{what}] ---", file=sys.stderr)
    print(current, file=sys.stderr)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare both committed blocks against the generator; exit "
        "non-zero if EITHER differs",
    )
    args = ap.parse_args(argv)

    models = _load_models()
    simple = render_simple_list(models)
    detailed = render_detailed_table(models)

    if not args.check:
        print(f"{START_MARKER} (README.md — ## Supported MLIPs)")
        print(simple)
        print(END_MARKER)
        print()
        print(f"{DETAILED_START_MARKER} (docs/model_status.md)")
        print(detailed)
        print(DETAILED_END_MARKER)
        return 0

    ok_readme = _check_block(
        README, START_MARKER, END_MARKER, simple,
        "README `## Supported MLIPs` list",
    )
    ok_doc = _check_block(
        MODEL_STATUS_DOC, DETAILED_START_MARKER, DETAILED_END_MARKER, detailed,
        "docs/model_status.md detailed table",
    )
    return 0 if (ok_readme and ok_doc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
