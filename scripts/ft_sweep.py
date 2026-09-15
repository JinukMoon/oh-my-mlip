#!/usr/bin/env python3
"""ft_sweep.py -- per-variant fine-tuning sweep for ONE env, ledger-first.

For every variant of the env it classifies, then (when attemptable) runs the
real chain `ft_run.py` (which itself converts the dataset via ft_dataset.py,
writes the rerunnable finetune_<variant>.sh and executes it) -> checkpoint
discovery -> `ft_verify.py <ckpt> --device cuda --json` (reload + energy/
forces witness). Every phase appends one JSONL row; the terminal row of each
variant carries exactly one `state` from this vocabulary:

  passed            train rc 0 AND a real (non-stub) builder AND ft_run's
                    ft_run.json provenance record AND a checkpoint the train
                    step NEWLY wrote AND ft_verify pass
  unsupported       upstream has no training path -- VALID ONLY WITH A
                    CITATION (models.json `evidence` or the campaign audit's
                    `citation`); an uncited claim is failed(uncited_unsupported)
  access-blocked    access prerequisite missing/unverified: gated checkpoint
                    and no HF token source PRESENT (presence checked only,
                    contents never read); nothing is fetched, so no remote
                    denial is observed or claimed
  resource-blocked  free disk below the floor at attempt time
  failed(<class>)   refused:exit2 | not_runnable_as_installed | no_builder |
                    seed_unhonoured (ft_run rc 5: refusal before any write) |
                    train | no_provenance_record | no_checkpoint | reload |
                    ft_verify:no_gpu_witness | audit_missing |
                    uncited_unsupported | ft_verify:no_loader
                    (| generic_stub: defensive only, see below)

The verify row records ft_verify's own `reason` and `version` verbatim
(`ft_verify_reason`, `ft_verify_version`; None when not reported).

ft_verify's checkpoint-loader capability is QUERIED, never assumed: first
`ft_verify.py --list-loaders --json` ({"loaders": [family, ...]}), else the
module's `loader_families()` / `_LOADER_TEMPLATE` keys, as shipped in the
runtime copy. A family the shipped ft_verify cannot reload is
failed(ft_verify:no_loader) -- an implementation gap, never `unsupported`.
A family without a real builder is reported by ft_run's exit code 4 (rc 2 =
refused status, rc 3 = not runnable as installed): failed(no_builder), with
nothing written. The older `_generic_stub` marker in finetune_config.json
(ft_run's former best-effort rendering, since removed upstream) is still
checked DEFENSIVELY: should any rendering ever label itself a stub again, it
is failed(generic_stub) rather than a pass on a stub's leftover file. It is
not an expected path in the current contract.

Classification source: models.json `finetune.status`. The two
`code-excavation-needed` candidates and the two `not-supported` variants are
re-audited per sweep run through `--audit PATH` (JSON {variant: {supported:
bool, citation: str}}); a candidate found `supported: true` is ATTEMPTED and
joins the required set (evidence_report.py reads the same file). Without an
audit entry a code-excavation-needed variant is failed(audit_missing) -- the
sweep never silently demotes it to unsupported.

A generated stub or emitted config is never a pass: the state is decided by
the ft_verify verdict on a checkpoint that the train step actually wrote.

Usage:
  python3 scripts/ft_sweep.py --env mace --dataset ft_demo.traj --ledger .ft/mace.jsonl
  python3 scripts/ft_sweep.py --env mace --dataset D --ledger L --audit audit.json \\
      --campaign-id C --manifest-sha256 SHA

Test-only: --ft-run-cmd / --ft-verify-cmd replace the real commands (explicit
injection, no monkeypatching); the variant / checkpoint is appended as the
first positional argument exactly as the real scripts take it.
"""
from __future__ import annotations

import argparse
import hashlib
import ast
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import resolve_home, utc_now  # noqa: E402
from setup_survey import token_source  # noqa: E402

STDERR_TAIL_CHARS = 2000
GIB = 1024 ** 3
STUB_MARKER = "_generic_stub"

# Checkpoint discovery per family: ordered glob preferences under <out>.
# The authoritative table is the SHIPPED ft_run.py's FAMILY_CHECKPOINT_GLOBS
# (the builders' own output conventions, each source-cited),
# read from the runtime copy without importing it (`family_checkpoint_globs`).
# This local table is only the fallback for a shipped ft_run.py that does not
# export the table (unreadable, or without FAMILY_CHECKPOINT_GLOBS). It is a
# MIRROR of the shipped table (one designating pattern per family) and is pinned to
# it by tests/test_ft_sweep.py whenever the shipped file is readable, so a
# second drift fails loudly instead of surviving behind a generic glob.
# DeePMD/DPA4: `model.ckpt.pt` is a SYMLINK the discovery filter rejects; the
# real files are model.ckpt-<step>.pt. Families without an entry use the
# generic list (newest match wins).
CKPT_GLOBS = {
    "MACE": ["{version}.model"],
    "SevenNet": ["checkpoint_best.pth"],
    "DeePMD": ["model.ckpt-*.pt"],
    "DPA4": ["model.ckpt-*.pt"],
    "GRACE": ["seed/*/final_model/saved_model.pb"],
    "PET": ["model-ft.pt"],
    "NequIP": ["checkpoints/last.ckpt"],
    "Allegro": ["checkpoints/last.ckpt"],
    "MatterSim": ["results/best_model.pth"],
    "TACE": ["checkpoints_epoch/last.ckpt"],
    "CHGNet": ["chgnet_ft/bestE_*.pth.tar"],
    "UMA": ["runs/ft/checkpoints/final/inference_ckpt.pt"],
    "fairchemv1": ["runs/checkpoints/ft/checkpoint.pt"],
    "EquFlash": ["runs/ft/checkpoints/checkpoint.pt"],
    "Nequix": ["wandb/offline-run-*/files/checkpoint.nqx"],
    "ORB": ["ckpts/checkpoint_epoch*.ckpt"],
}
GENERIC_CKPT_GLOBS = ["*.model", "*.pth", "*.pt", "*.ckpt"]
# Family-specific glob prefixes kept IN FRONT of the shipped table (a more
# specific preference, never a replacement).
CKPT_GLOB_PREFIX = {"MACE": ["{version}.model"]}
# A discovered checkpoint whose file name is one of these is one file of a
# BUNDLE (TensorFlow SavedModel: saved_model.pb + variables/ + assets/ ...):
# the whole parent directory is hashed and preserved, the named file alone
# would not reload.
CKPT_BUNDLE_MARKERS = {"saved_model.pb"}


def family_checkpoint_globs(home: Path) -> dict:
    """The shipped ft_run.py's FAMILY_CHECKPOINT_GLOBS, read by parsing the
    file (never importing it: ft_run loads the registry at import time).
    Returns {"globs": {family: [glob, ...]}, "source": str}; a script without
    the table (or an unparsable one) yields the local fallback with its
    source named so the plan row says which rules were used."""
    script = home / "scripts" / "ft_run.py"
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except (OSError, SyntaxError) as exc:
        return {"globs": dict(CKPT_GLOBS), "source": f"ft_sweep.CKPT_GLOBS (ft_run.py unreadable: {exc})"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FAMILY_CHECKPOINT_GLOBS"
                                                for t in node.targets):
            try:
                table = ast.literal_eval(node.value)
            except ValueError:
                break
            if isinstance(table, dict) and all(isinstance(v, list) and all(isinstance(g, str) for g in v)
                                               for v in table.values()):
                merged = {fam: [g for g in CKPT_GLOB_PREFIX.get(fam, []) if g not in globs] + list(globs)
                          for fam, globs in table.items()}
                return {"globs": merged, "source": f"{script}:FAMILY_CHECKPOINT_GLOBS"}
            break
    return {"globs": dict(CKPT_GLOBS), "source": "ft_sweep.CKPT_GLOBS (ft_run.py exports no FAMILY_CHECKPOINT_GLOBS)"}

# Fallback probe when ft_verify.py has no --list-loaders yet: import the
# SHIPPED module and read its real loader table (a public loader_families()
# wins when present). Runs in a subprocess so ft_verify's import-time
# registry loading never touches this process.
_LOADER_PROBE = r'''
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ft_verify", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
fn = getattr(m, "loader_families", None)
fams = fn() if callable(fn) else list(getattr(m, "_LOADER_TEMPLATE", {}))
print(json.dumps({"loaders": sorted(set(fams)), "source": "loader_families()" if callable(fn) else "_LOADER_TEMPLATE"}))
'''


def loader_families(home: Path, python: str | None = None) -> dict:
    """The families the SHIPPED ft_verify.py can actually reload, queried
    from the script itself. Returns {"loaders": set, "source": str}; on any
    probe failure the set is EMPTY (every family then fails no_loader --
    fail closed, never a guessed capability)."""
    script = home / "scripts" / "ft_verify.py"
    py = python or sys.executable
    attempts = [([py, str(script), "--list-loaders", "--json"], "ft_verify --list-loaders"),
                ([py, "-c", _LOADER_PROBE, str(script)], "import probe")]
    errors = []
    for cmd, label in attempts:
        try:
            proc = subprocess.run(cmd, cwd=str(home), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label}: {exc}")
            continue
        payload = last_json_line(proc.stdout) if proc.returncode == 0 else None
        if payload and isinstance(payload.get("loaders"), list):
            return {"loaders": set(payload["loaders"]), "source": payload.get("source") or label}
        errors.append(f"{label}: rc={proc.returncode} {proc.stderr.strip()[-300:]}")
    return {"loaders": set(), "source": "unavailable", "errors": errors}


def stub_marker(out: Path) -> str | None:
    """ft_run's own honesty label: finetune_config.json[_generic_stub]."""
    cfg = out / "finetune_config.json"
    if not cfg.is_file():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, dict) and STUB_MARKER in data:
        return str(data[STUB_MARKER])
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_variants(env: str, home: Path) -> list[tuple[str, str, dict]]:
    """[(family, version, version_spec)] for every variant living in `env`."""
    registry = json.loads((home / "models.json").read_text(encoding="utf-8"))
    out = []
    for family, spec in registry.items():
        if family.startswith("_") or spec.get("env") != env:
            continue
        for version, vspec in (spec.get("versions") or {}).items():
            out.append((family, version, vspec or {}))
    return out


def load_audit(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data.get("variants", data)
    if not isinstance(entries, dict):
        raise SystemExit("[ft_sweep] audit file must be a JSON object {variant: {supported, citation}}")
    return entries


def classify(version: str, vspec: dict, audit: dict, token_missing: bool) -> dict:
    """Pre-attempt decision, pure. Returns {action: attempt|record, state,
    citation, reason}."""
    ft = vspec.get("finetune") or {}
    status = ft.get("status") or ""
    entry = audit.get(version) or {}
    citation = entry.get("citation") or ""
    if status == "not-supported":
        cite = citation or "; ".join(ft.get("evidence") or [])
        if not cite:
            return {"action": "record", "state": "failed(uncited_unsupported)", "citation": "",
                    "reason": f"{version}: not-supported without any citation"}
        if entry.get("supported"):
            return {"action": "attempt", "state": None, "citation": citation,
                    "reason": "campaign audit found a training path; attempting"}
        return {"action": "record", "state": "unsupported", "citation": cite,
                "reason": ft.get("reason") or "not-supported"}
    if status == "code-excavation-needed":
        if not entry:
            return {"action": "record", "state": "failed(audit_missing)", "citation": "",
                    "reason": f"{version}: code-excavation-needed needs a current-campaign audit entry"}
        if entry.get("supported"):
            return {"action": "attempt", "state": None, "citation": citation,
                    "reason": "campaign audit found a training path; attempting"}
        if not citation:
            return {"action": "record", "state": "failed(uncited_unsupported)", "citation": "",
                    "reason": f"{version}: audit says unsupported but gives no citation"}
        return {"action": "record", "state": "unsupported", "citation": citation,
                "reason": ft.get("reason") or "audit: no shipped checkpoint-consuming train path"}
    if vspec.get("gated") and token_missing:
        # M8: the ACCESS PREREQUISITE is missing/unverified -- no token source
        # is present (presence only; a token's contents are never read) and
        # nothing was fetched, so no remote denial (401/403) was observed
        # and none is claimed
        return {"action": "record", "state": "access-blocked", "citation": "",
                "reason": "access prerequisite missing/unverified: gated checkpoint and no HF token source "
                          "present (HF_TOKEN / HF_TOKEN_PATH / hf cache / OMM_HF_TOKEN_FILE); "
                          "no remote denial observed (nothing fetched)",
                "access": {"prerequisite": "hf_token_source", "status": "missing/unverified",
                           "checked": "presence_only", "remote_denial_observed": False, "fetched": False}}
    return {"action": "attempt", "state": None, "citation": citation, "reason": ""}


def find_checkpoint(out: Path, family: str, version: str, extra_glob: str | None = None,
                    not_before: float | None = None, globs: dict | None = None) -> Path | None:
    """Newest matching regular, non-empty file under `out` written at or
    after `not_before` (the train start): a stale artifact from an earlier
    attempt or a symlink to something outside never counts as this run's
    checkpoint. `globs` is the per-family table (the shipped ft_run.py's,
    see family_checkpoint_globs); default: the local fallback."""
    table = globs if globs is not None else CKPT_GLOBS
    patterns = ([extra_glob] if extra_glob else []) + table.get(family, GENERIC_CKPT_GLOBS)
    for pat in patterns:
        matches = [p for p in out.rglob(pat.format(version=version))
                   if p.is_file() and not p.is_symlink() and p.stat().st_size > 0
                   and "_compiled" not in p.name and "/data/" not in str(p)
                   and (not_before is None or p.stat().st_mtime >= not_before)]
        if matches:
            return max(matches, key=lambda p: p.stat().st_mtime)
    return None


def witness_ok(verdict: dict | None, device: str) -> tuple[bool, str | None]:
    """The reload witness must carry a finite energy AND a forces shape; on
    cuda the verifier's own MEASURED GPU report must be present and True
    (`gpu_used` is True and `device`, when reported, is cuda). A cuda verify
    whose witness lacks `gpu_used` is `no_gpu_witness` -- a failure, never a
    pass with an unreported witness."""
    if not verdict or not verdict.get("pass"):
        return False, "verifier_failed"
    energy = verdict.get("energy_ev")
    if not isinstance(energy, (int, float)) or isinstance(energy, bool) or energy != energy \
            or energy in (float("inf"), float("-inf")):
        return False, "non_finite_energy"
    if not verdict.get("forces_shape"):
        return False, "forces_missing"
    if device == "cuda":
        if verdict.get("gpu_used") is not True:
            return False, "no_gpu_witness" if verdict.get("gpu_used") is None else "gpu_not_used"
        if verdict.get("device") and verdict["device"] != "cuda":
            return False, "gpu_not_used"
    return True, None


# ft_run refusal gate exit codes (contract with ft_run.py, e2e-finetune):
# 2 refused status, 3 not runnable as installed, 4 no real builder for this
# family (implementation gap: nothing written), 5 seed unhonoured (an
# explicit --seed on a data-split-only family without --allow-partial-seed:
# a refusal before any write, never a training failure -- the sweep passes
# no --seed and never --allow-partial-seed). Anything else is a real
# training failure.
FT_RUN_EXIT_CLASSES = {2: "refused:exit2", 3: "not_runnable_as_installed", 4: "no_builder", 5: "seed_unhonoured"}
# ft_run's per-run provenance record (schema ft_run.json/1: args, seed
# control, designated checkpoint, artifacts, rematerialize argv), REQUIRED
# after a successful train step and preserved with the checkpoint.
FT_RUN_PROVENANCE = "ft_run.json"


def run(command: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, errors="replace", env=env)
    return proc.returncode, proc.stdout, proc.stderr[-STDERR_TAIL_CHARS:]


def last_json_line(stdout: str) -> dict | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


class Ledger:
    def __init__(self, path: Path, common: dict):
        self.path = path
        self.common = common
        self.seq = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, **row) -> dict:
        self.seq += 1
        record = {"seq": self.seq, "at": utc_now(), "kind": "finetune", **self.common, **row}
        record.setdefault("state", None)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        return record


def sweep(env: str, home: Path, ledger_path: Path, dataset: Path, out_root: Path, *,
          audit: dict, campaign_id: str | None, manifest_sha256: str | None,
          epochs: int = 2, device: str = "cuda", min_free_gib: float = 5.0,
          ft_run_cmd: list[str] | None = None, ft_verify_cmd: list[str] | None = None,
          ckpt_glob: str | None = None) -> list[dict]:
    common = {"env": env, "campaign_id": campaign_id, "manifest_sha256": manifest_sha256}
    ledger = Ledger(ledger_path, common)
    variants = env_variants(env, home)
    # capability of the SHIPPED verifier -- an injected test verifier is taken
    # to handle whatever it is handed (its fakes decide the state).
    caps = {"loaders": None, "source": "injected --ft-verify-cmd"} if ft_verify_cmd else loader_families(home)
    # checkpoint discovery rules: the SHIPPED ft_run.py's own table (its
    # builders' output conventions), never a copy that could drift
    glob_table = family_checkpoint_globs(home)
    ledger.append(family=None, variant=None, phase="plan",
                  targets=[v for _, v, _ in variants], dataset=str(dataset), epochs=epochs, device=device,
                  ft_verify_loaders=sorted(caps["loaders"]) if caps["loaders"] is not None else None,
                  ft_verify_loaders_source=caps["source"], unit="GiB", min_free_gib=min_free_gib,
                  ckpt_globs=glob_table["globs"], ckpt_globs_source=glob_table["source"])
    token_missing = token_source() == "none"
    rows: list[dict] = []
    for family, version, vspec in variants:
        decision = classify(version, vspec, audit, token_missing)
        if decision["action"] == "record":
            rows.append(ledger.append(family=family, variant=version, phase="audit",
                                      state=decision["state"], returncode=None, stderr_tail="",
                                      evidence={"citation": decision["citation"], "reason": decision["reason"],
                                                "status": (vspec.get("finetune") or {}).get("status"),
                                                **({"access": decision["access"]} if decision.get("access") else {})}))
            continue

        # measured free space where the outputs will land, GiB, at attempt time
        free_gib = shutil.disk_usage(out_root if out_root.exists() else home).free / GIB
        if free_gib < min_free_gib:
            rows.append(ledger.append(family=family, variant=version, phase="skipped_disk",
                                      state="resource-blocked", returncode=None,
                                      stderr_tail=f"free disk {free_gib:.2f} GiB < {min_free_gib:.2f} GiB floor",
                                      evidence={"unit": "GiB", "free_gib": round(free_gib, 3),
                                                "min_free_gib": min_free_gib}))
            continue

        out = out_root / version
        out.mkdir(parents=True, exist_ok=True)
        base = ft_run_cmd or [sys.executable, str(home / "scripts" / "ft_run.py")]
        train = base + [version, "--dataset", str(dataset), "--out", str(out),
                        "--epochs", str(epochs), "--device", device]
        train_started = time.time() - 1.0  # 1 s slack for coarse mtime resolution
        rc, out_text, err = run(train, home)
        sh = out / f"finetune_{version}.sh"
        evidence = {"out": str(out), "sh": str(sh) if sh.exists() else None,
                    "command": train, "citation": decision["citation"], "device": device,
                    "train_started_utc": utc_now()}
        if rc != 0:
            klass = FT_RUN_EXIT_CLASSES.get(rc, "train")
            rows.append(ledger.append(family=family, variant=version, phase="train",
                                      state=f"failed({klass})", returncode=rc, stderr_tail=err,
                                      evidence=evidence))
            continue
        stub = stub_marker(out)
        evidence["builder"] = "generic_stub" if stub is not None else "real"
        ledger.append(family=family, variant=version, phase="train", returncode=rc,
                      stderr_tail=err, evidence=evidence)
        if stub is not None:
            rows.append(ledger.append(family=family, variant=version, phase="train",
                                      state="failed(generic_stub)", returncode=rc,
                                      stderr_tail=f"ft_run rendered its self-labelled generic stub for {family}: {stub}",
                                      evidence=evidence))
            continue
        # the provenance record ft_run writes for this run is REQUIRED (it is
        # what makes the run rematerializable) and is preserved with the
        # checkpoint; a train step without one is not a pass
        prov = out / FT_RUN_PROVENANCE
        prov_error = None
        if not prov.is_file() or prov.is_symlink():
            prov_error = f"ft_run wrote no {FT_RUN_PROVENANCE} under {out} (provenance record required)"
        else:
            try:
                prov_doc = json.loads(prov.read_text(encoding="utf-8"))
                if not isinstance(prov_doc, dict):
                    prov_error = f"{prov} is not a JSON object"
            except (OSError, json.JSONDecodeError) as exc:
                prov_error = f"{prov} unreadable or not JSON: {exc}"
        if prov_error is not None:
            rows.append(ledger.append(family=family, variant=version, phase="train",
                                      state="failed(no_provenance_record)", returncode=rc,
                                      stderr_tail=prov_error, evidence=evidence))
            continue
        evidence["ft_run_json"] = str(prov)
        evidence["ft_run_json_sha256"] = sha256_file(prov)
        evidence["ft_run_json_schema"] = prov_doc.get("schema")

        ckpt = find_checkpoint(out, family, version, ckpt_glob, not_before=train_started, globs=glob_table["globs"])
        if ckpt is None:
            rows.append(ledger.append(family=family, variant=version, phase="checkpoint",
                                      state="failed(no_checkpoint)", returncode=None,
                                      stderr_tail=f"no checkpoint newly written under {out} (pre-existing files ignored)",
                                      evidence=evidence))
            continue
        evidence["ckpt"] = str(ckpt)
        evidence["ckpt_sha256"] = sha256_file(ckpt)
        evidence["ckpt_bytes"] = ckpt.stat().st_size
        if ckpt.name in CKPT_BUNDLE_MARKERS:
            # one file of a SavedModel bundle: record (and later preserve)
            # every regular file of the bundle directory, hashed
            bundle_dir = ckpt.parent
            files = [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
                     for p in sorted(bundle_dir.rglob("*")) if p.is_file() and not p.is_symlink()]
            # every directory of the bundle too (an empty assets/ is legal
            # SavedModel layout): the preserver recreates them, files or not
            dirs = [str(p) for p in sorted(bundle_dir.rglob("*")) if p.is_dir() and not p.is_symlink()]
            evidence["ckpt_bundle"] = {"dir": str(bundle_dir), "files": files, "dirs": dirs,
                                       "bytes": sum(f["bytes"] for f in files), "file_count": len(files)}

        if caps["loaders"] is not None and family not in caps["loaders"]:
            rows.append(ledger.append(family=family, variant=version, phase="verify",
                                      state="failed(ft_verify:no_loader)", returncode=None,
                                      stderr_tail=f"shipped ft_verify.py has no checkpoint loader for {family} "
                                                  f"(loaders: {sorted(caps['loaders'])}, source: {caps['source']})",
                                      evidence=evidence))
            continue
        vbase = ft_verify_cmd or [sys.executable, str(home / "scripts" / "ft_verify.py")]
        # the VARIANT key, so variant-specific reload details (modal, ...) resolve
        verify = vbase + [str(ckpt), "--model", version, "--device", device, "--json"]
        rc, out_text, err = run(verify, home)
        verdict = last_json_line(out_text)
        evidence["verdict"] = verdict
        ok, why = witness_ok(verdict, device) if rc == 0 else (False, "verifier_failed")
        # the MEASURED witness as reported (True/False) or None when the verifier
        # did not report one; a cuda pass requires True (checked in witness_ok)
        evidence["gpu_witness"] = (verdict or {}).get("gpu_used") if device == "cuda" else "n/a(cpu)"
        # ft_verify's own verdict fields, VERBATIM (None when the verifier did
        # not report them): `reason` is its failure vocabulary ("" on a pass;
        # non_finite_energy, bad_forces_shape, non_finite_forces,
        # witness_record_missing, witness_inconsistent, gpu_not_used) and
        # `version` the exact variant it resolved
        evidence["ft_verify_reason"] = (verdict or {}).get("reason")
        evidence["ft_verify_version"] = (verdict or {}).get("version")
        if ok:
            state = "passed"
        elif why == "no_gpu_witness":
            state = "failed(ft_verify:no_gpu_witness)"
        else:
            state = "failed(reload)"
        vreason = evidence["ft_verify_reason"]
        tail = err if ok else (f"{why} (ft_verify reason: {vreason})" if why and vreason else (why or err))
        rows.append(ledger.append(family=family, variant=version, phase="verify", state=state,
                                  returncode=rc, stderr_tail=tail, evidence=evidence))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--env", required=True, help="env name (mace, sevennet, ...)")
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--out-root", default=None, help="default: $OH_MY_MLIP_HOME/.ft/<env>")
    ap.add_argument("--audit", default=None, help="FT support audit JSON for this run")
    ap.add_argument("--campaign-id", default=None)
    ap.add_argument("--manifest-sha256", default=None)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--min-free-gib", type=float, default=5.0, help="measured free-space floor, GiB (2^30)")
    ap.add_argument("--ckpt-glob", default=None)
    ap.add_argument("--ft-run-cmd", default=None, help="TEST ONLY")
    ap.add_argument("--ft-verify-cmd", default=None, help="TEST ONLY")
    args = ap.parse_args()

    home = resolve_home()
    out_root = Path(args.out_root) if args.out_root else home / ".ft" / args.env
    rows = sweep(args.env, home, args.ledger, args.dataset.resolve(), out_root,
                 audit=load_audit(Path(args.audit) if args.audit else None),
                 campaign_id=args.campaign_id, manifest_sha256=args.manifest_sha256,
                 epochs=args.epochs, device=args.device, min_free_gib=args.min_free_gib,
                 ft_run_cmd=args.ft_run_cmd.split() if args.ft_run_cmd else None,
                 ft_verify_cmd=args.ft_verify_cmd.split() if args.ft_verify_cmd else None,
                 ckpt_glob=args.ckpt_glob)
    for r in rows:
        print(f"  {r['variant']:<24} {r['state']}")
    print(json.dumps({"ft_sweep": True, "env": args.env, "ledger": str(args.ledger),
                      "states": {r["variant"]: r["state"] for r in rows}}))
    return 0  # completion is the contract; per-variant facts live in the ledger


if __name__ == "__main__":
    raise SystemExit(main())
