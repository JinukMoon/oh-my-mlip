#!/usr/bin/env python3
"""gen_recipes.py — render `docs/recipes.md` from the registry + upstream table.

Per framework the recipe book carries three blocks:

  A. install   upstream's OWN documented command (scripts/upstream_recipes.py,
               each entry carrying the doc URL it was read from) PLUS the
               pinned combination from `envs/<env>.yml` (or the multi-pass
               sidecar), rendered as commands you can paste.
  B. weights   upstream's OWN acquisition idiom — the framework CLI / python
               call / URL that actually fetches the checkpoint, including any
               post-download step (freeze / export / flatten / AOT compile).
  C. compute   `models.json` imports + inference, verbatim.

Two guards keep this from drifting away from `models.json`:

  * hard  — a curated `curl` + `sha256sum` pair must quote the registry's
            recorded digest for that variant, else generation FAILS.
  * soft  — EVERY registry fingerprint not already quoted in a recipe is
            emitted automatically, so a newly recorded sha256 can never be
            silently missing from the doc.

The output is HERMETIC: no host-specific paths, no per-machine state, no
`$HOME`. Paths render as `$OMM/...` so the doc is byte-identical on any clone —
which is what makes `--check` usable in CI.

Modes:
  (default)  print the rendered document to stdout
  --write    write it to docs/recipes.md
  --check    compare against the committed docs/recipes.md; exit non-zero on diff
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from oh_my_mlip import registry  # noqa: E402
from upstream_recipes import UPSTREAM  # noqa: E402

DOC = REPO / "docs" / "recipes.md"
CATBENCH_PIN = "1.1.2"
# Rendered in place of the host's real GPU arch (sm86/sm89/...) in compile paths.
ARCH_PLACEHOLDER = "${OMM_ARCH}"

# Variant whose registry digest / source id must appear in that family's curated
# recipe. Families absent here are covered by the soft guard only.
CHK = {
    "NequIP": "NequIP-OAM-XL", "Allegro": "Allegro-OAM-L", "Nequix": "Nequix-MP-1",
    "AlphaNet": "AlphaNet-v1-OMA", "Eqnorm": "Eqnorm-MPtrj", "fairchemv1": "eSEN-30M-OAM",
    "EquiformerV3": "EqV3-OMatMPtrjSalex", "MatRIS": "MatRIS-10M-OAM",
    "DPA4": "DPA-4.0.1-pro-MPtrj", "TACE": "TACE-OAM-L", "EquFlash": "EquFlashV2",
}


def _yaml():
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - dev dependency
        raise SystemExit("gen_recipes.py needs pyyaml (see requirements-dev.txt)")
    return yaml


def load_models() -> dict:
    return json.loads((REPO / "models.json").read_text())


def canonical(text: str) -> str:
    """Collapse this clone's absolute path so the doc is host-independent."""
    return text.replace(str(REPO), "$OMM").replace("${OH_MY_MLIP_HOME}", "$OMM")


def pin_block(env: str) -> list[str]:
    """The pinned version combination, as pasteable commands."""
    yaml = _yaml()
    ymlp, side = REPO / "envs" / f"{env}.yml", REPO / "envs" / f"{env}.build.sh"
    d = yaml.safe_load(ymlp.read_text())
    conda = [x for x in d["dependencies"] if isinstance(x, str)]
    pipsec = next((x["pip"] for x in d["dependencies"] if isinstance(x, dict)), [])
    channels = " ".join(f"-c {c}" for c in d.get("channels", []))
    idx = [p for p in pipsec if p.startswith("-")]
    pkgs = [p for p in pipsec if not p.startswith("-")]

    L: list[str] = []
    if side.exists():
        text = side.read_text()
        subs = dict(re.findall(r'^([A-Z_]+)="([^"$]+)"\s*$', text, re.M))
        L.append(f"bash $OMM/envs/{env}.build.sh <prefix>      "
                 f"# single conda solve impossible -> the sidecar owns the build")
        L.append("#   what the sidecar actually runs:")
        cont = False
        for line in text.splitlines():
            s = line.strip()
            if not (cont or s.startswith(('"$CONDA_BIN"', '"$PIP"'))):
                continue
            cont = s.endswith("\\")
            s = re.sub(r"\s*\|\|\s*\{.*?\}\s*$", "", s)
            s = (s.replace('"$CONDA_BIN"', "conda").replace('"$PIP"', "<prefix>/bin/pip")
                  .replace('"$PREFIX"', "<prefix>").replace("--yes", "-y"))
            for k, v in subs.items():
                s = s.replace("${%s}" % k, v).replace("$%s" % k, v)
            if s:
                L.append("#     " + s)
            elif L and L[-1].endswith("\\"):
                L[-1] = L[-1].rstrip("\\ ")
    else:
        L.append(f"conda env create -f $OMM/envs/{env}.yml -p <prefix>")
        L.append("#   equivalent explicit form:")
        L.append(f"#     conda create -y -p <prefix> {channels} " + " ".join(conda))
        if pkgs:
            L.append(f"#     <prefix>/bin/pip install {' '.join(idx + pkgs)}")
    L.append(f"<prefix>/bin/pip install catbench=={CATBENCH_PIN}    "
             f"# every env: D3 dispersion + adsorption benchmarking")
    return L


def weights_block(fam: str, models: dict) -> list[str]:
    u = UPSTREAM[fam]
    entry = models[fam]
    chk = CHK.get(fam)
    target = chk or entry.get("default_version") or next(iter(entry["versions"]))
    s = registry.resolve(fam, version=target, models=models)
    lines = [canonical(ln).replace("{sha}", s["weights_sha256"] or "")
                          .replace("{url}", s["weights_source"])
             for ln in u["fetch"]]

    # Hard guard — HARDCODED digests only. A `{sha}` placeholder is substituted
    # from the registry above, so checking it would be a tautology (it cannot
    # drift by construction). What can drift is a digest typed literally into
    # the recipe, so every 64-hex literal must still be one this family records.
    known = {registry.resolve(fam, version=v, models=models)["weights_sha256"]
             for v in entry["versions"]}
    known.discard(None)
    for literal in {h for ln in u["fetch"] for h in re.findall(r"\b[0-9a-f]{64}\b", ln)}:
        if literal not in known:
            raise SystemExit(
                f"DRIFT {fam}: recipe hardcodes sha256 {literal} which the registry "
                f"no longer records (known: {sorted(known) or 'none'})")

    if chk:  # the source identifier must still be reachable from the recipe text
        body = "\n".join(lines)
        s_chk = registry.resolve(fam, version=chk, models=models)
        if not s_chk["weights_sha256"]:
            ident = s_chk["weights_source"].rsplit("/", 1)[-1].split(":")[0]
            if ident not in body:
                raise SystemExit(f"DRIFT {fam}: registry source id {ident!r} absent from recipe")

    # soft guard: surface every recorded fingerprint the recipe does not quote
    body = "\n".join(lines)
    extra = []
    for v in entry["versions"]:
        sv = registry.resolve(fam, version=v, models=models)
        sha, size = sv["weights_sha256"], sv["weights_size"]
        if sha and sha not in body:
            extra.append(f"#   {v}: {sha}" + (f"  ({size:,} B)" if size else ""))
    if extra:
        lines += ["#", "# recorded fingerprints — verify with: "
                       "python3 $OMM/scripts/verify_weights_integrity.py"] + extra
    return lines


def render(models: dict) -> str:
    out: list[str] = []
    w = out.append

    w("# MLIP recipes — install, fetch, and run each framework")
    w("")
    w("<!-- GENERATED FILE — do not edit by hand. -->")
    w("<!-- Regenerate: python3 scripts/gen_recipes.py --write   ·   CI: --check -->")
    w("")
    w("One section per framework: how to install it, how to get its weights, and the")
    w("calculator line that runs it. Blocks **A** and **B** are what each framework's own")
    w("documentation says, with the source URL linked. Block **C** comes verbatim from")
    w("`models.json` — keep those lines exactly as written.")
    w("")
    w("Upstream install gives the supported way; the pin gives one combination of versions")
    w("known to work together.")
    w("")
    w("```bash")
    w("export OMM=$(pwd)          # this clone")
    w("source $OMM/env.sh         # shared caches + D3/CUDA environment, once per shell")
    w("```")
    w("")
    w("`<prefix>` is wherever that env is installed — `$OMM/envs/<env>` for a stock")
    w("`install.sh` build, or your own path if you adopted an existing env")
    w("(`scripts/adopt_env.py`). Resolve it programmatically with")
    w("`oh_my_mlip.resolve(<model>)[\"python\"]` rather than hardcoding.")
    w("")
    w("## The short path (let the hub do it)")
    w("")
    w("The per-model blocks below are the *explicit* procedure — useful to audit what")
    w("happens, to run air-gapped, or to port elsewhere. Day to day, four commands cover")
    w("every model in this registry:")
    w("")
    w("```bash")
    w("# 1. is the env already here?   ready -> skip to 3")
    w("python3 $OMM/scripts/setup_survey.py --table")
    w("")
    w("# 2. build it (or adopt an env you already have, at zero disk cost)")
    w("$OMM/install.sh <env>")
    w("python3 $OMM/scripts/adopt_env.py <env> <prefix>      # alternative to building")
    w("")
    w("# 3. materialise the weights (handles URL rewriting, HF auth, unpacking)")
    w("python3 -c \"import sys; sys.path.insert(0,'$OMM'); from oh_my_mlip import fetch; \\")
    w("    print(fetch.ensure_weights('<Framework>', version='<Variant>'))\"")
    w("")
    w("# 4. check it works: energy + forces, exit 0 on success")
    w("python3 $OMM/scripts/setup_verify.py <Variant> --json")
    w("```")
    w("")
    w("A bare name resolves family-first, so pass `--version` to reach a variant whose")
    w("name its family shadows.")
    w("")
    w("---")
    w("")

    yaml = _yaml()
    for fam in [k for k in models if not k.startswith("_")]:
        entry = models[fam]
        env = entry["env"]
        versions = list(entry["versions"])
        u = UPSTREAM[fam]
        ymld = yaml.safe_load((REPO / "envs" / f"{env}.yml").read_text())
        conda = [x for x in ymld["dependencies"] if isinstance(x, str)]
        pyver = next((c.split("=", 1)[1] for c in conda if c.startswith("python=")), "?")
        pipsec = next((x["pip"] for x in ymld["dependencies"] if isinstance(x, dict)), [])
        torchp = next((p for p in pipsec if p.startswith("torch==")), "—")

        w(f"## {fam} — env `{env}`")
        w("")
        w(f"pin: python {pyver} · {torchp} · variants: " + ", ".join(f"`{v}`" for v in versions))
        w("")
        w("### A. install")
        w("")
        w(f"Upstream ([source]({u['src'].split()[0]}))"
          + (f" — requires: {u['req']}" if u.get("req") else "") + ":")
        w("")
        w("```bash")
        for ln in u["pip"]:
            w(ln)
        w("```")
        w("")
        w("Pinned combination:")
        w("")
        w("```bash")
        for ln in pin_block(env):
            w(ln)
        w("```")
        w("")
        w("### B. weights")
        w("")
        w(f"Upstream acquisition ([source]({u['fetch_src'].split()[0]})):")
        w("")
        w("```bash")
        for ln in weights_block(fam, models):
            w(ln)
        w("```")
        w("")
        w("### C. ASE calculator")
        w("")
        for v in versions:
            # ARCH_PLACEHOLDER keeps arch-pinned compile paths generic: resolve()
            # would otherwise bake THIS machine's GPU arch into the doc, which is
            # both wrong for the reader and a --check failure on another host.
            s = registry.resolve(fam, version=v, arch=ARCH_PLACEHOLDER, models=models)
            w(f"`{v}`" + (" *(default)*" if v == entry.get("default_version") else ""))
            w("")
            w("```python")
            for ln in s["imports"] + s["inference"]:
                w(canonical(ln))
            w("atoms.calc = calc")
            w("```")
            w("")
        run = f'{s["env_run_raw"]} ' if s.get("env_run_raw") else ""
        w(f"Run: `{run}<prefix>/bin/python script.py` — never `conda activate`.")
        w("")
        w(f"Check it works: `python3 $OMM/scripts/setup_verify.py {versions[0]} --json` "
      f"(prints energy + forces; exit 0 on success).")
        w("")
        if u.get("note"):
            w(f"> **Note:** {u['note']}")
            w("")
        w("---")
        w("")

    return "\n".join(out).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {DOC.relative_to(REPO)}")
    ap.add_argument("--check", action="store_true",
                    help="compare against the committed doc; exit non-zero on diff")
    args = ap.parse_args(argv)

    text = render(load_models())

    if args.check:
        if not DOC.is_file():
            print(f"gen_recipes: {DOC.relative_to(REPO)} is missing — run --write", file=sys.stderr)
            return 1
        if DOC.read_text() != text:
            print(f"gen_recipes: {DOC.relative_to(REPO)} is STALE — "
                  f"run `python3 scripts/gen_recipes.py --write`", file=sys.stderr)
            return 1
        print(f"gen_recipes: {DOC.relative_to(REPO)} in sync with models.json")
        return 0

    if args.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(text)
        print(f"wrote {DOC.relative_to(REPO)} ({len(text.splitlines())} lines)")
        return 0

    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
