#!/usr/bin/env python3
"""gen_finetune.py — render `docs/finetune.md` from `scripts/upstream_finetune.py`
(family-level upstream FT procedures) + `models.json` (per-variant `finetune`
classification, P2a/P2b).

Sibling generator to `gen_recipes.py`, same discipline: research table (.py)
-> generator with `--write/--check` -> generated doc -> pytest + CI step.
When upstream changes its documented fine-tuning procedure, edit
`upstream_finetune.py`; when a variant's classification changes, edit
`models.json`; never hand-edit the generated doc.

Per family the doc carries four blocks:

  A. what upstream documents    verbatim `cmd`, with the doc `src` linked.
  B. dataset format             what the trainer actually reads (`data_format`).
  C. the command this hub generates   one `ft_run.py <Variant> ...` invocation
                                        per variant, gated by that variant's own
                                        `finetune` block:
                                          - documented + runnable_as_installed:
                                            the real invocation (licence-gated:
                                            EquFlash and MACE-MH-1 rows print
                                            the checkpoint licence + URL right
                                            above the command, then proceed);
                                          - documented + NOT runnable_as_installed:
                                            the same invocation, commented as
                                            BLOCKED with the live blocker list
                                            (ft_run.py itself exits 3 for this
                                            case);
                                          - not-supported / code-excavation-needed:
                                            a refusal line (ft_run.py exits 2)
                                            with the reason, plus evidence URLs
                                            for not-supported (that status
                                            asserts an ABSENCE and needs a
                                            citation).
  D. per-variant support matrix   status / runnable_as_installed / demonstrated
                                    / licence, one row per variant.

Two guards, mirroring gen_recipes.py:

  * orphans      every models.json family must have an UPSTREAM_FT entry and
                  vice versa -- a new family can never silently ship an empty
                  fine-tuning section, and a stale UPSTREAM_FT entry can never
                  silently reference a family that no longer exists.
  * src coverage every family's upstream `src` URL must appear in the
                  rendered text, so a citation can never be quietly dropped.

Every variant must resolve to a classification (its `finetune.status` is
read straight from models.json, which the tests/schema/models.schema.json
`required` gate already enforces at the JSON level).

The output is HERMETIC: no host-specific or home-directory paths, no host
GPU arch (arch never enters this doc at all -- fine-tuning commands are not
arch-pinned the way compiled-accelerator recipes are). `ft_run.py` invocation
lines use `$OMM` for this clone, never an absolute path -- so `--check` is
meaningful on any machine.

Modes:
  (default)  print the rendered document to stdout
  --write    write it to docs/finetune.md
  --check    compare against the committed docs/finetune.md; exit non-zero on diff
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from upstream_finetune import UPSTREAM_FT  # noqa: E402
from ft_run import _LICENCE_URLS  # noqa: E402

DOC = REPO / "docs" / "finetune.md"

# Doc-honesty states that mean "upstream has a documented fine-tuning path",
# regardless of the parenthetical qualifier a family may carry (inherited /
# config-level / generic-plus-excavation). Anything else (not-supported,
# code-excavation-needed) is a refusal row in block C.
_DOCUMENTED_PREFIX = "documented"


def load_models() -> dict:
    return json.loads((REPO / "models.json").read_text())


def canonical(text: str) -> str:
    """Collapse this clone's absolute path so the doc is host-independent
    (same rule as gen_recipes.py's canonical() -- defensive here, since every
    string this module renders is already written in ${OH_MY_MLIP_HOME}/$OMM
    or host-independent form, never a literal absolute path)."""
    return text.replace(str(REPO), "$OMM").replace("${OH_MY_MLIP_HOME}", "$OMM")


def _slug(version: str) -> str:
    return version.lower().replace("_", "-")


def families(models: dict) -> list[str]:
    return [k for k in models if not k.startswith("_")]


def variants(models: dict, fam: str) -> list[str]:
    return list(models[fam]["versions"])


def variant_finetune(models: dict, fam: str, version: str) -> dict:
    ft = models[fam]["versions"][version].get("finetune")
    if ft is None:
        raise SystemExit(f"gen_finetune: {fam}/{version} has no 'finetune' block")
    return ft


def render_command_block(fam: str, version: str, ft: dict) -> list[str]:
    """Block C for one variant: the ft_run.py invocation this hub generates,
    gated by that variant's own classification."""
    status = ft.get("status") or ""
    out: list[str] = []

    if not status.startswith(_DOCUMENTED_PREFIX):
        # not-supported / code-excavation-needed -- ft_run.py's refusal path
        # (exit 2). Render the reason; not-supported additionally cites
        # evidence for the ABSENCE it asserts.
        out.append(f"`{version}` -- **{status}**. `ft_run.py {version}` refuses "
                    f"(exit 2): {ft.get('reason') or 'no reason recorded'}")
        for url in ft.get("evidence") or []:
            out.append(f"  - evidence: {url}")
        return out

    invocation = (f"python3 $OMM/scripts/ft_run.py {version} "
                  f"--dataset <your-dataset> --out ft_{_slug(version)}")
    licence = ft.get("licence")
    lines: list[str] = []
    if licence:
        url = _LICENCE_URLS.get(licence, "")
        lines.append(f"# LICENCE: {licence}" + (f" -- {url}" if url else ""))
        lines.append("#   Disclosed to users — oh-my-mlip is MIT and redistributes no")

    if ft.get("runnable_as_installed"):
        lines.append(invocation)
    else:
        lines.append("# BLOCKED -- ft_run.py exits 3 until:")
        for b in ft.get("blockers") or []:
            lines.append(f"#   - {b}")
        lines.append(invocation)

    out.append(f"`{version}`:")
    out.append("")
    out.append("```bash")
    out.extend(lines)
    out.append("```")
    return out


def render_matrix(models: dict, fam: str) -> list[str]:
    out = ["| variant | status | runnable_as_installed | licence | demonstrated |",
           "|---|---|---|---|---|"]
    for v in variants(models, fam):
        ft = variant_finetune(models, fam, v)
        out.append(
            f"| `{v}` | {ft.get('status')} | {ft.get('runnable_as_installed')} | "
            f"{ft.get('licence') or '—'} | {ft.get('demonstrated') or '—'} |"
        )
    return out


def render(models: dict) -> str:
    out: list[str] = []
    w = out.append

    w("# Fine-tuning recipes — continuing training from a foundation checkpoint")
    w("")
    w("<!-- GENERATED FILE — do not edit by hand. -->")
    w("<!-- Regenerate: python3 scripts/gen_finetune.py --write   ·   CI: --check -->")
    w("")
    w("One section per framework: **A** is what upstream's own fine-tuning docs say")
    w("(verbatim command, source linked); **B** is the dataset format the trainer")
    w("actually reads; **C** is the single command this hub generates per variant —")
    w("`scripts/ft_run.py` resolves the variant's `finetune` block in `models.json`,")
    w("converts the dataset, writes the patched config/command, and (when runnable)")
    w("executes it; **D** is the per-variant support matrix. `status` reflects")
    w("documented fine-tuning support, not an execution guarantee — see `demonstrated` for that.")
    w("")
    w("```bash")
    w("export OMM=$(pwd)          # this clone")
    w("source $OMM/env.sh         # shared caches + D3/CUDA environment, once per shell")
    w("```")
    w("")
    w("A variant refused with `exit 2` is `not-supported` or `code-excavation-needed`")
    w("— no amount of retrying fixes it, see the reason and evidence in block D/C.")
    w("A variant refused with `exit 3` is `documented` but `runnable_as_installed:")
    w("false` — the listed blockers are fixable (usually one `pip install`, or a git")
    w("clone); fix them and rerun the same command.")
    w("")
    w("---")
    w("")

    for fam in families(models):
        u = UPSTREAM_FT[fam]
        env = models[fam]["env"]
        w(f"## {fam} — env `{env}`")
        w("")
        w("### A. What upstream documents")
        w("")
        w(f"Upstream ([source]({u['src'].split()[0]})):")
        w("")
        if u.get("cmd"):
            w("```bash")
            for ln in u["cmd"]:
                w(canonical(ln))
            w("```")
        else:
            w("_No fine-tuning command exists upstream — see block D for why._")
        w("")
        if u.get("note"):
            w(f"> **Note:** {u['note']}")
            w("")

        w("### B. Dataset format")
        w("")
        w(canonical(u.get("data_format") or "n/a."))
        w("")

        w("### C. The command this hub generates")
        w("")
        for v in variants(models, fam):
            ft = variant_finetune(models, fam, v)
            for ln in render_command_block(fam, v, ft):
                w(ln)
            w("")

        w("### D. Per-variant support matrix")
        w("")
        for ln in render_matrix(models, fam):
            w(ln)
        w("")
        w("---")
        w("")

    return "\n".join(out).rstrip() + "\n"


def coverage_line(models: dict) -> str:
    total = demonstrated = 0
    for fam in families(models):
        for v in variants(models, fam):
            ft = variant_finetune(models, fam, v)
            total += 1
            if ft.get("demonstrated"):
                demonstrated += 1
    return f"{total}/{total} classified · {demonstrated}/{total} demonstrated"


def check_orphans(models: dict) -> None:
    fams = set(families(models))
    missing = sorted(fams - set(UPSTREAM_FT))
    extra = sorted(set(UPSTREAM_FT) - fams)
    if missing:
        raise SystemExit(f"gen_finetune: models.json families with no UPSTREAM_FT entry: {missing}")
    if extra:
        raise SystemExit(f"gen_finetune: UPSTREAM_FT entries for unknown families: {extra}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {DOC.relative_to(REPO)}")
    ap.add_argument("--check", action="store_true",
                     help="compare against the committed doc; exit non-zero on diff")
    args = ap.parse_args(argv)

    models = load_models()
    check_orphans(models)
    text = render(models)

    for fam in families(models):
        src = UPSTREAM_FT[fam]["src"].split()[0]
        if src not in text:
            raise SystemExit(f"gen_finetune: {fam}'s src URL is missing from the rendered doc")

    if args.check:
        if not DOC.is_file():
            print(f"gen_finetune: {DOC.relative_to(REPO)} is missing — run --write", file=sys.stderr)
            return 1
        if DOC.read_text() != text:
            print(f"gen_finetune: {DOC.relative_to(REPO)} is STALE — "
                  f"run `python3 scripts/gen_finetune.py --write`", file=sys.stderr)
            return 1
        print(f"gen_finetune: {DOC.relative_to(REPO)} in sync with models.json "
              f"({coverage_line(models)})")
        return 0

    if args.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(text)
        print(f"wrote {DOC.relative_to(REPO)} ({len(text.splitlines())} lines) "
              f"({coverage_line(models)})")
        return 0

    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
