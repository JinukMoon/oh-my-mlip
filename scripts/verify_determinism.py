#!/usr/bin/env python3
"""verify_determinism.py - fail CI on nondeterministic recipe requirements.

Checks every env recipe's conda AND pip dependencies without importing torch,
ase, conda, or any model package.

A conda dependency is deterministic only when it is one of:

  - pkg=version=build          (fully exact)
  - pkg=maj.min.patch          (conda prefix-match on a complete version)
  - pkg=maj.min[.*]            (SERIES pin — allowed only for SERIES_PIN_OK
                                packages whose patch level is deliberately
                                floated to follow the host toolchain)
  - a bare package in BARE_OK  (the pip provider only)

Bare packages and range constraints (>=, <=, ~=, >, <, !=) are failures. This
half of the gate exists because pip pins alone did NOT make a rebuild
reproducible: a floating conda ``- ase`` silently rode the newest release into
the env and broke mattersim/eqnorm, and pinning ase per env was the fix — a
hole this checker was previously blind to.

A pip requirement is deterministic only when it is one of:

  - pkg==version
  - pkg @ git+...@<7-40 hex sha>
  - -e git+...@<7-40 hex sha>#egg=...
  - --extra-index-url / --find-links / -f flag line
  - an http(s) *.whl URL
  - a documented private file:// source in a candidate recipe

Bare packages, range specifiers, wildcard pins, and git URLs without an exact
commit SHA are failures.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ENVS_DIR = REPO_ROOT / "envs"

_STATUS_RE = re.compile(r"^#\s*build_status:\s*(\w+)", re.IGNORECASE | re.MULTILINE)
_REASON_RE = re.compile(
    r"^#\s*candidate-reason:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
_PIP_START_RE = re.compile(r"^(?P<indent>\s*)-\s+pip\s*:\s*(?:#.*)?$")
_LIST_ITEM_RE = re.compile(r"^(?P<indent>\s*)-\s+(?P<value>.+?)\s*$")
_COMMENT_SPLIT_RE = re.compile(r"\s+#")

_SHA = r"[0-9A-Fa-f]{7,40}"
_PKG = r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_.?, -]+\])?"

_EXACT_RE = re.compile(rf"^{_PKG}==(?P<version>[^\s;]+)(?:\s*;.+)?$")
_DIRECT_GIT_RE = re.compile(
    rf"^{_PKG}\s*@\s*git\+\S+@{_SHA}(?:[/?#]\S*)?(?:\s*;.+)?$"
)
_EDITABLE_GIT_RE = re.compile(
    rf"^-e\s+git\+\S+@{_SHA}\S*#egg=[A-Za-z0-9_.-]+(?:\[[^\]]+\])?(?:\s*;.+)?$"
)
_FLAG_RE = re.compile(
    r"^(?:--extra-index-url(?:\s+|=)\S+|--find-links(?:\s+|=)\S+|-f\s+\S+)$"
)
_WHEEL_URL_RE = re.compile(
    rf"^(?:{_PKG}\s*@\s*)?https?://\S+\.whl(?:[#?]\S*)?(?:\s*;.+)?$",
    re.IGNORECASE,
)


# ── conda dependency rules ───────────────────────────────────────────────────
# `pip` is the provider entry that makes the pip block installable at all; its
# own version does not select any model code, and every recipe declares it.
BARE_OK = {"pip"}
# Packages whose PATCH level is deliberately floated to track the host CUDA
# toolchain. The major.minor pair is what the wheels are built against, and
# that pair IS pinned — a patch bump inside the series is a compatible driver
# component, not a model change.
SERIES_PIN_OK = {"cuda-nvcc"}

_CONDA_DEPS_START_RE = re.compile(r"^(?P<indent>\s*)dependencies\s*:\s*(?:#.*)?$")
_CONDA_PKG = r"[A-Za-z0-9][A-Za-z0-9_.\-]*"
_CONDA_EXACT_BUILD_RE = re.compile(rf"^{_CONDA_PKG}=[^=<>!~\s]+=[^=\s]+$")
_CONDA_FULL_VERSION_RE = re.compile(rf"^{_CONDA_PKG}=\d+\.\d+\.\d+[A-Za-z0-9.\-+]*$")
_CONDA_SERIES_RE = re.compile(rf"^(?P<pkg>{_CONDA_PKG})=(?:\d+\.\d+(?:\.\*)?|\d+\.\*)$")
_CONDA_RANGE_RE = re.compile(rf"^{_CONDA_PKG}\s*(?:>=|<=|~=|!=|>|<)")
_CONDA_BARE_RE = re.compile(rf"^{_CONDA_PKG}$")


@dataclass
class PipEntry:
    lineno: int
    requirement: str
    raw: str
    comment: str = ""


@dataclass
class Offender:
    lineno: int
    requirement: str
    reason: str


@dataclass
class RecipeReport:
    name: str
    path: Path
    build_status: str
    candidate_reason: str
    offenders: list[Offender] = field(default_factory=list)
    private_file_docs: list[int] = field(default_factory=list)

    @property
    def deterministic(self) -> bool:
        return not self.offenders

    @property
    def candidate_with_private_source(self) -> bool:
        return self.build_status == "candidate" and bool(self.private_file_docs)


def _read_build_status(text: str) -> str:
    m = _STATUS_RE.search(text)
    return m.group(1).strip().lower() if m else "unknown"


def _read_candidate_reason(text: str) -> str:
    m = _REASON_RE.search(text)
    return m.group(1).strip() if m else ""


def _strip_wrapping_quotes(value: str) -> str:
    """Strip one matching pair of surrounding quotes from a YAML scalar so a
    quoted pip requirement (e.g. a git+URL with a #subdirectory fragment) is
    classified on its content, not the quote character."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _split_inline_comment(value: str) -> tuple[str, str]:
    """Split a YAML scalar's trailing comment without treating #egg/#sha as a
    comment marker."""
    match = _COMMENT_SPLIT_RE.search(value)
    if not match:
        return _strip_wrapping_quotes(value.strip()), ""
    return _strip_wrapping_quotes(value[: match.start()].strip()), value[match.start() + 1 :].strip()


def _pip_entries(text: str) -> tuple[list[PipEntry], list[tuple[int, str]]]:
    """Return live pip list entries and comment-only file:// docs in the pip block."""
    lines = text.splitlines()
    entries: list[PipEntry] = []
    file_comments: list[tuple[int, str]] = []

    in_pip = False
    pip_indent = 0
    for idx, line in enumerate(lines, start=1):
        if not in_pip:
            m = _PIP_START_RE.match(line)
            if m:
                in_pip = True
                pip_indent = len(m.group("indent"))
            continue

        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        if stripped and indent <= pip_indent and not stripped.startswith("#"):
            break
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "file://" in stripped:
                file_comments.append((idx, stripped))
            continue

        m = _LIST_ITEM_RE.match(line)
        if not m or len(m.group("indent")) <= pip_indent:
            continue
        requirement, comment = _split_inline_comment(m.group("value"))
        entries.append(PipEntry(idx, requirement, line.rstrip(), comment))

    return entries, file_comments


def _conda_entries(text: str) -> list[PipEntry]:
    """Return the recipe's conda dependency items (the pip block excluded).

    Text-scanned rather than taken from ``yaml.safe_load`` so every offender
    can be reported at its own line number, exactly like the pip half. Items
    nested deeper than the dependency indent belong to ``- pip:`` and are
    skipped here — ``_pip_entries`` owns those.
    """
    lines = text.splitlines()
    entries: list[PipEntry] = []

    in_deps = False
    deps_indent = 0
    item_indent: int | None = None
    for idx, line in enumerate(lines, start=1):
        if not in_deps:
            m = _CONDA_DEPS_START_RE.match(line)
            if m:
                in_deps = True
                deps_indent = len(m.group("indent"))
            continue

        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= deps_indent:
            break  # dedented out of dependencies:

        m = _LIST_ITEM_RE.match(line)
        if not m:
            continue
        if item_indent is None:
            item_indent = len(m.group("indent"))
        if len(m.group("indent")) > item_indent:
            continue  # nested under `- pip:`
        value, comment = _split_inline_comment(m.group("value"))
        if value.rstrip().endswith(":"):
            continue  # the `- pip:` mapping key itself
        entries.append(PipEntry(idx, value, line.rstrip(), comment))

    return entries


def _classify_conda(entry: PipEntry) -> Offender | None:
    dep = entry.requirement.strip()
    if not dep:
        return Offender(entry.lineno, entry.requirement, "empty conda dependency")

    if _CONDA_EXACT_BUILD_RE.match(dep) or _CONDA_FULL_VERSION_RE.match(dep):
        return None

    series = _CONDA_SERIES_RE.match(dep)
    if series:
        if series.group("pkg") in SERIES_PIN_OK:
            return None
        return Offender(
            entry.lineno, dep,
            f"series pin floats the patch level; pin a full version or add "
            f"{series.group('pkg')!r} to SERIES_PIN_OK with a reason",
        )

    if _CONDA_RANGE_RE.match(dep):
        return Offender(entry.lineno, dep, "range/version constraint is not deterministic")
    if _CONDA_BARE_RE.match(dep):
        if dep in BARE_OK:
            return None
        return Offender(entry.lineno, dep, "bare conda package is not deterministic")
    return Offender(entry.lineno, dep, "unrecognized nondeterministic conda dependency")


def _is_documented_private_file(
    *,
    build_status: str,
    candidate_reason: str,
    entry: PipEntry | None = None,
    comment_text: str = "",
) -> bool:
    if build_status != "candidate":
        return False
    parts = [candidate_reason, comment_text]
    if entry is not None:
        parts.extend([entry.raw, entry.comment])
    text = " ".join(parts).lower()
    return "file://" in text and ("private" in text or "owner" in text)


def _classify_requirement(
    entry: PipEntry,
    build_status: str,
    candidate_reason: str,
) -> Offender | None:
    req = entry.requirement.strip()
    if not req:
        return Offender(entry.lineno, entry.requirement, "empty pip requirement")

    if "file://" in req:
        if _is_documented_private_file(
            build_status=build_status,
            candidate_reason=candidate_reason,
            entry=entry,
        ):
            return None
        return Offender(
            entry.lineno,
            req,
            "file:// source is allowed only when documented in a candidate recipe",
        )

    if _FLAG_RE.match(req):
        return None
    if _WHEEL_URL_RE.match(req):
        return None
    if _EDITABLE_GIT_RE.match(req):
        return None
    if req.startswith("-e git+") or "git+" in req:
        if _DIRECT_GIT_RE.match(req):
            return None
        return Offender(entry.lineno, req, "git requirement lacks an exact commit SHA")

    exact = _EXACT_RE.match(req)
    if exact:
        version = exact.group("version")
        if "*" in version:
            return Offender(entry.lineno, req, "wildcard version is not deterministic")
        return None

    if re.match(rf"^{_PKG}\s*(?:>=|<=|~=|!=|>|<)", req):
        return Offender(entry.lineno, req, "range/version constraint is not deterministic")
    if re.match(rf"^{_PKG}$", req):
        return Offender(entry.lineno, req, "bare package is not deterministic")
    return Offender(entry.lineno, req, "unrecognized nondeterministic pip requirement")


def check_recipe(path: Path) -> RecipeReport:
    text = path.read_text(encoding="utf-8")
    try:
        doc = yaml.safe_load(text) or {}
    except Exception as exc:  # noqa: BLE001
        report = RecipeReport(path.stem, path, "unknown", "")
        report.offenders.append(Offender(1, path.name, f"YAML parse error: {exc!r}"))
        return report

    name = str(doc.get("name") or path.stem)
    build_status = _read_build_status(text)
    candidate_reason = _read_candidate_reason(text)
    entries, file_comments = _pip_entries(text)
    report = RecipeReport(name, path, build_status, candidate_reason)

    for lineno, comment in file_comments:
        if _is_documented_private_file(
            build_status=build_status,
            candidate_reason=candidate_reason,
            comment_text=comment,
        ):
            report.private_file_docs.append(lineno)
        else:
            report.offenders.append(
                Offender(
                    lineno,
                    comment,
                    "file:// documentation is allowed only in a private-source candidate",
                )
            )

    for entry in entries:
        offender = _classify_requirement(entry, build_status, candidate_reason)
        if offender is not None:
            report.offenders.append(offender)
        elif "file://" in entry.requirement:
            report.private_file_docs.append(entry.lineno)

    for entry in _conda_entries(text):
        offender = _classify_conda(entry)
        if offender is not None:
            report.offenders.append(offender)

    report.offenders.sort(key=lambda o: o.lineno)
    return report


def check_all(envs_dir: Path = ENVS_DIR) -> list[RecipeReport]:
    return [
        check_recipe(path)
        for path in sorted(envs_dir.glob("*.yml"))
        if not path.name.startswith("_")
    ]


# ── beyond the YAML: sidecars and locks ──────────────────────────────────────
# The YAML check alone passed 20/20 while envs/equflash.build.sh installed a
# dozen packages with no version, and while envs/mace.yml pinned an ase that
# the validated build never ended up with. Both undercut the claim that a
# recipe reproduces the env it was validated with.

def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _base_version(version: str) -> str:
    return version.split("+", 1)[0].strip()


def check_build_sidecars(envs_dir: Path = ENVS_DIR) -> list[tuple[Path, int, str]]:
    """(sidecar, line, requirement) for every package a *.build.sh installs with
    pip WITHOUT an exact version. A git URL pinned to a commit counts as pinned."""
    import shlex

    offenders: list[tuple[Path, int, str]] = []
    for path in sorted(envs_dir.glob("*.build.sh")):
        raw = path.read_text().splitlines()
        i = 0
        while i < len(raw):
            start, line = i + 1, raw[i]
            while line.rstrip().endswith("\\") and i + 1 < len(raw):   # join continuations
                i += 1
                line = line.rstrip()[:-1] + " " + raw[i]
            i += 1
            if "install" not in line or "pip" not in line.lower():
                continue
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            if "install" not in tokens:
                continue
            args = tokens[tokens.index("install") + 1:]
            for cut, tok in enumerate(args):                     # the pip command ends here
                if tok in ("||", "&&", "|", ";", "{", "}") or tok.endswith(";"):
                    args = args[:cut]
                    break
            text_all = path.read_text()
            skip_next = False
            for tok in args:
                if skip_next:
                    skip_next = False
                    continue
                if tok in ("-r", "--requirement", "-c", "--constraint", "-e", "--editable",
                           "-f", "--find-links", "-i", "--index-url", "--extra-index-url"):
                    skip_next = True
                    continue
                if tok.startswith("-") or tok in (";", "&&", "||", "|") or tok.startswith("$"):
                    continue
                if "git+" in tok:
                    # pinned to a commit, directly or through a variable the
                    # sidecar assigns a full hex SHA (SHA="65f8ea93...")
                    var = re.search(r"@\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", tok)
                    pinned = bool(re.search(r"@[0-9a-f]{7,40}(#|$)", tok)) or bool(
                        var and re.search(rf'^{var.group(1)}="?[0-9a-f]{{40}}"?\s*$', text_all, re.M))
                    if not pinned:
                        offenders.append((path, start, tok))
                    continue
                if "==" not in tok and not tok.endswith((".whl", ".tar.gz")):
                    offenders.append((path, start, tok))
    return offenders


def _recipe_pins(text: str) -> dict[str, tuple[str, str]]:
    """{name: (version, "conda"|"pip")} for every exact pin a recipe declares."""
    pins: dict[str, tuple[str, str]] = {}
    in_pip = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if stripped.startswith("- pip:"):
            in_pip = True
            continue
        if not stripped.startswith("- "):
            continue
        spec = stripped[2:].strip()
        indent = len(line) - len(line.lstrip())
        if in_pip and indent < 6:          # left the nested pip list
            in_pip = False
        match = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|=)\s*([^\s,;]+)$", spec)
        if match:
            pins[_norm(match.group(1))] = (match.group(3), "pip" if in_pip else "conda")
    return pins


def _lock_versions(locks_dir: Path, env: str) -> dict[str, str]:
    """The FINAL version of each package in the validated build: conda layer
    first, then the pip layer on top (pip reinstalls over conda, as the replay
    does: conda create --file, then pip install -r)."""
    final: dict[str, str] = {}
    conda_lock = locks_dir / f"{env}.conda.txt"
    if conda_lock.exists():
        for url in conda_lock.read_text().splitlines():
            if not url.startswith("http"):
                continue
            stem = url.rsplit("/", 1)[-1]
            stem = re.sub(r"\.(conda|tar\.bz2)$", "", stem)
            parts = stem.rsplit("-", 2)
            if len(parts) == 3:
                final[_norm(parts[0])] = parts[1]
    pip_lock = locks_dir / f"{env}.pip.txt"
    if pip_lock.exists():
        for line in pip_lock.read_text().splitlines():
            match = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)", line.strip())
            if match:
                final[_norm(match.group(1))] = match.group(2)
    return final


def check_recipe_vs_lock(envs_dir: Path = ENVS_DIR) -> list[tuple[str, str, str, str, str]]:
    """(env, package, recipe_version, layer, locked_final_version) wherever a
    recipe pins a version the validated build did not end up with."""
    locks_dir = envs_dir / "locks"
    out = []
    for path in sorted(envs_dir.glob("*.yml")):
        if path.name.startswith("_"):
            continue
        final = _lock_versions(locks_dir, path.stem)
        if not final:
            continue
        for name, (version, layer) in sorted(_recipe_pins(path.read_text()).items()):
            locked = final.get(name)
            if not locked:
                continue
            if version.endswith(".*"):                            # a deliberate series pin
                if _base_version(locked).startswith(version[:-1]):
                    continue
            elif _base_version(locked) == _base_version(version):
                continue
            out.append((path.stem, name, version, layer, locked))
    return out


def _print_report(reports: list[RecipeReport], root: Path) -> None:
    for report in reports:
        rel = report.path.relative_to(root) if report.path.is_relative_to(root) else report.path
        if report.deterministic:
            suffix = " (candidate private source)" if report.candidate_with_private_source else ""
            print(f"{rel}: deterministic{suffix}")
            continue
        print(f"{rel}: NON-DETERMINISTIC ({len(report.offenders)} offender(s))")
        for offender in report.offenders:
            print(
                f"  - {rel}:{offender.lineno}: {offender.requirement} "
                f"({offender.reason})"
            )

    deterministic = sum(1 for r in reports if r.deterministic)
    nondeterministic = len(reports) - deterministic
    private_candidates = sum(1 for r in reports if r.candidate_with_private_source)
    print()
    print(
        "verify_determinism: "
        f"{deterministic} deterministic / {nondeterministic} non-deterministic / "
        f"{private_candidates} candidate-with-private-source "
        f"({len(reports)} total)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--envs-dir",
        default=str(ENVS_DIR),
        help="directory of *.yml recipes (default: <repo>/envs)",
    )
    args = parser.parse_args(argv)

    envs_dir = Path(args.envs_dir)
    reports = check_all(envs_dir)
    _print_report(reports, REPO_ROOT)

    sidecar = check_build_sidecars(envs_dir)
    print()
    if sidecar:
        print(f"build sidecars: {len(sidecar)} unpinned pip install(s)")
        for path, lineno, req in sidecar:
            print(f"  - {path.name}:{lineno}: {req} (no exact version)")
    else:
        print("build sidecars: every pip install is pinned")

    drift = check_recipe_vs_lock(envs_dir)
    if drift:
        # Informational, not a failure: a recipe pin is a BUILD input, the lock is
        # the validated env's FINAL state, and a later install step may upgrade a
        # package the recipe pinned for the build (setuptools < 81 is pinned so
        # source builds still find pkg_resources; a later step then lifts it). The
        # lock is what OMM_USE_LOCK=1 replays exactly.
        print(f"recipe vs lock (info): {len(drift)} recipe pin(s) differ from the validated final env")
        for env, name, version, layer, locked in drift:
            print(f"  - {env}: recipe pins {name} {version} ({layer}); the validated env ends with {locked}")
        print("  a recipe pin governs the build; OMM_USE_LOCK=1 replays the exact validated env")
    else:
        print("recipe vs lock: every recipe pin matches the validated build")

    ok = all(r.deterministic for r in reports) and not sidecar
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
