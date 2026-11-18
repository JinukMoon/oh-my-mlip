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

  2. The FULL detailed table (Model / Framework / Weights / Gated / Code — one
     row per model+version) for `docs/model_status.md`, between

         <!-- STATUS_TABLE_DETAILED_START -->
         <!-- STATUS_TABLE_DETAILED_END -->

Both are derived ENTIRELY from the registry (`weights` / `gated` per version,
`repo` per framework), so neither view can drift from `models.json`. The
per-version `validation` field stays in the registry and in the MCP
`model_status` tool but is not rendered in the public docs.

Newer GPU architectures get their own column (`ARCH_COLUMNS`, today sm120):
a base recipe validated on older GPUs says nothing about them, so each variant
shows what was measured there -- `arch_validation.<arch>` in models.json
(set only after setup_verify passed on such a GPU) and the recipe table
envs/_arch_recipes.json (unsupported there) -- and "not checked" otherwise.

Modes:
  (default)   print both generated blocks to stdout (clearly delimited).
  --check     compare each generated block against its committed file and exit
              non-zero if EITHER differs (this is what CI calls).
  --write     replace both blocks in place (README.md, docs/model_status.md).

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
MODEL_STATUS_DOC = _REPO_ROOT / "docs" / "model_status.md"

# Light README list (Framework | Models).
START_MARKER = "<!-- STATUS_TABLE_START -->"
END_MARKER = "<!-- STATUS_TABLE_END -->"

# Detailed per-version table in docs/model_status.md.
DETAILED_START_MARKER = "<!-- STATUS_TABLE_DETAILED_START -->"
DETAILED_END_MARKER = "<!-- STATUS_TABLE_DETAILED_END -->"

ARCH_RECIPES = _REPO_ROOT / "envs" / "_arch_recipes.json"
# Architectures shown as their own column. Older GPUs use the base recipes.
ARCH_COLUMNS = ("sm120",)


def _arch_table() -> dict:
    return json.loads(ARCH_RECIPES.read_text(encoding="utf-8")) if ARCH_RECIPES.is_file() else {}


def arch_state(models: dict, framework: str, version: str, arch: str, table: dict | None = None) -> str:
    """One variant's state on `arch`: validated / failed / unsupported / not checked."""
    table = _arch_table() if table is None else table
    info = models[framework]
    entry = (table.get(arch) or {}).get(info.get("env"))
    if isinstance(entry, dict) and not entry.get("recipe"):
        return "unsupported"
    state = (info["versions"][version].get("arch_validation") or {}).get(arch)
    return {"validated": "validated", "failed": "failed"}.get(state, "not checked")


def _framework_arch_state(models: dict, framework: str, arch: str, table: dict) -> str:
    states = {arch_state(models, framework, v, arch, table) for v in models[framework].get("versions", {})}
    if states == {"validated"}:
        return "validated"
    if "validated" in states:
        return "partly validated"
    if states == {"unsupported"}:
        return "unsupported"
    if "failed" in states:
        return "failed"
    return "not checked"

def _detailed_rows(models: dict) -> list[tuple]:
    """Build (Model, Framework, Weights, Gated) rows in a stable order: registry
    framework order, then version order within each framework. Always uses the
    version key (resolvable via resolve()) rather than mlip_name, so users can
    copy it directly into their code."""
    rows: list[tuple] = []
    table = _arch_table()
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        for version, vinfo in info.get("versions", {}).items():
            rows.append(
                (
                    version,
                    framework,
                    vinfo.get("weights", "bundled"),
                    "yes" if vinfo.get("gated", False) else "no",
                    *(arch_state(models, framework, version, a, table) for a in ARCH_COLUMNS),
                )
            )
    return rows


def _framework_rows(models: dict) -> list[tuple]:
    """Build (Framework, comma-joined model variant names) rows in registry
    order — one row per framework. This is the LIGHT view for the README.
    Uses version key when it differs from mlip_name, so users can copy and run."""
    rows: list[tuple] = []
    table = _arch_table()
    for framework, info in models.items():
        if framework.startswith("_"):
            continue
        names = [
            version for version in info.get("versions", {})
        ]
        rows.append((framework, ", ".join(names),
                     *(_framework_arch_state(models, framework, a, table) for a in ARCH_COLUMNS)))
    return rows


def render_simple_list(models: dict) -> str:
    """Render the light README `| Framework | Models |` table (no trailing
    newline)."""
    lines = ["| Framework | Models | " + " | ".join(ARCH_COLUMNS) + " |",
             "|---|---|" + "---|" * len(ARCH_COLUMNS)]
    for framework, names, *arch in _framework_rows(models):
        lines.append(f"| {framework} | {names} | " + " | ".join(arch) + " |")
    return "\n".join(lines)


def render_detailed_table(models: dict) -> str:
    """Render the full detailed markdown table (no trailing newline). The last
    column links each framework's upstream repository (`repo` in models.json)
    as a small button: an `attr_list` class styled in docs/stylesheets/extra.css."""
    repos = {fw: info.get("repo") for fw, info in models.items() if not fw.startswith("_")}
    header = "| Model | Framework | Weights | Gated | " + " | ".join(ARCH_COLUMNS) + " | Code |"
    sep = "|---|---|---|---|" + "---|" * len(ARCH_COLUMNS) + "---|"
    lines = [header, sep]
    for model, framework, weights, gated, *arch in _detailed_rows(models):
        repo = repos.get(framework)
        code = f"[GitHub]({repo}){{ .md-button .omm-repo }}" if repo else "-"
        lines.append(f"| {model} | {framework} | {weights} | {gated} | " + " | ".join(arch) + f" | {code} |")
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
    ap.add_argument("--write", action="store_true",
                    help="replace both generated blocks in their files in place")
    args = ap.parse_args(argv)

    models = _load_models()
    simple = render_simple_list(models)
    detailed = render_detailed_table(models)

    if args.write:
        for path, start, end, block in ((README, START_MARKER, END_MARKER, simple),
                                        (MODEL_STATUS_DOC, DETAILED_START_MARKER, DETAILED_END_MARKER, detailed)):
            text = path.read_text(encoding="utf-8")
            head, rest = text.split(start, 1)
            _, tail = rest.split(end, 1)
            path.write_text(f"{head}{start}\n{block}\n{end}{tail}", encoding="utf-8")
            print(f"gen_status_table: wrote {path.relative_to(_REPO_ROOT)}")
        return 0

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
