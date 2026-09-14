"""Drift guards for the shared end-to-end recipes (recipes/*.md).

The recipes are the workflow home (ask → plan → approve → execute → verify
→ out of scope) that both host adapters point at: Claude Code through
skills/*/SKILL.md, Codex through AGENTS.md §10. Five things must stay true
without anyone remembering to check:

  * discovery — every skill points at its recipe, and AGENTS.md points at
    every recipe (a recipe nobody points at is dead text);
  * structure — every recipe opens with an executable-chain table and
    carries the six stages in order, so a host that reads one recipe can
    rely on the same protocol in all of them;
  * executable source of truth — every recipe names the owner files that
    actually install / emit / run / verify its workflow, every such path
    exists, and no recipe carries a package-management command of its own
    (installation knowledge lives in install.sh + envs/, never in prose);
  * implemented vs planned — every scripts/ or run_examples/ path advertised
    ABOVE the "Planned helpers" heading exists; every helper listed BELOW it
    does not (the day one lands, this test forces its promotion); a flag is
    "parsed" only when the script's argparse really adds it (AST evidence,
    not a substring of the source) — planned flags are not parsed, promoted
    flags are;
  * readiness is read from the working tree — a helper that exists but has
    no dedicated tests/test_<stem>*.py is listed by the recipe as landed
    without a unit test, and that list must match the tree exactly in both
    directions (a helper gaining a test drops off it);
  * no duplicated bodies / public framing — recipes share no 10-gram with
    the skills (and, via test_onramp_contract_no_dup.py, none with
    AGENTS.md); no hardware product names, measured-result patterns,
    personal paths, or ad hoc clone/copy commands.

GPU-free, stdlib-only.
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPE_DIR = REPO_ROOT / "recipes"
AGENTS = REPO_ROOT / "AGENTS.md"

# One recipe per plugin skill; the skill directory name IS the recipe name.
SKILL_NAMES = sorted(p.name for p in (REPO_ROOT / "skills").iterdir() if (p / "SKILL.md").is_file())
RECIPES = {name: RECIPE_DIR / f"{name}.md" for name in SKILL_NAMES}

CHAIN_HEADING = "## Executable chain"
STAGE_HEADINGS = ["## 1. Ask", "## 2. Plan", "## 3. Approve", "## 4. Execute", "## 5. Verify", "## 6. Out of scope"]
PLANNED_HEADING = "## Planned helpers (not implemented)"
MISSING_MARKER = "Missing executable parts"
UNTESTED_MARKER = "Landed without a unit test"
CANDIDATE_FLAG = "--plugin-dir"

# Tests that check documentation contracts rather than a helper's behaviour;
# their mention of a script name is not unit-test evidence for it.
CONTRACT_TESTS = {"test_recipe_contract.py", "test_onramp_contract_no_dup.py", "test_skill_contract_paths.py"}

# The owner files each recipe MUST name above the planned heading: the
# executables that install / emit / run / verify that workflow. A recipe
# that stops naming one of these has drifted back into prose.
REQUIRED_OWNER_FILES = {
    "setup": [
        "install.sh", "env.sh", "envs/_expected.json", "scripts/lint_recipes.py",
        "scripts/setup_survey.py", "scripts/setup_verify.py", "scripts/setup_sweep.py",
        "scripts/setup_guardrail.py", "scripts/fresh_root.py", "scripts/evidence_report.py",
        "run_examples/single_point.py",
    ],
    "run": ["run_examples/single_point.py", "run_examples/relax.py"],
    "catbench": [
        "run_examples/catbench_quickstart.py", "scripts/catbench_jobgen.py",
        "scripts/catbench_report.py", "scripts/catbench_version.py",
        "scripts/catbench_datasets.py", "scripts/catbench_vasp_stage.py",
        "scripts/catbench_leaderboard.py", "install.sh",
    ],
    "finetune": [
        "scripts/upstream_finetune.py", "scripts/gen_finetune.py", "scripts/ft_dataset.py",
        "scripts/ft_run.py", "scripts/ft_verify.py", "scripts/ft_sweep.py", "docs/finetune.md",
    ],
    "distill": ["scripts/distill_bootstrap.py", "scripts/build_lammps_nnmtp.sh", "scripts/distill_verify.py"],
}

# Flags a recipe advertises as implemented on an existing script: the script
# must really parse them (mirror image of PLANNED_FLAGS).
PROMOTED_FLAGS = {
    "run": {
        "run_examples/relax.py": ["--structure", "--fmax", "--steps"],
        "run_examples/single_point.py": ["--structure"],
    },
    "setup": {
        "scripts/setup_verify.py": ["--all-variants", "--no-local-record"],
        "scripts/setup_sweep.py": [
            "--fresh-root", "--ft-dataset", "--ft-audit", "--seed-spec", "--preflight",
            "--min-free-gib", "--start-reserve-gib", "--no-cleanup",
            "--peak-gib", "--cold-cache-gib", "--ft-download-gib",
        ],
        "scripts/fresh_root.py": ["--owned-registry", "--allowed-parent", "--owned-group", "--require"],
        "scripts/evidence_report.py": ["--strict", "--bundle", "--allow-degraded", "--audit"],
    },
    "catbench": {
        "scripts/catbench_version.py": ["--mode", "--record", "--offline"],
        "scripts/catbench_datasets.py": ["--target", "--confirm", "--check", "--fetch", "--workdir", "--record", "--catbench-version"],
        "run_examples/catbench_quickstart.py": ["--only", "--emit-only", "--catbench-version", "--regenerate"],
        "scripts/catbench_vasp_stage.py": ["--source", "--dest", "--dataset-name", "--coeff", "--all-files"],
        "scripts/catbench_leaderboard.py": ["--snapshot", "--models", "--catbench-version", "--calc-num",
                                            "--official-id", "--official-conditions"],
        "scripts/catbench_jobgen.py": ["--calc-file", "--python", "--catbench-version", "--structure", "--tag", "--regenerate"],
    },
    "finetune": {
        "scripts/ft_sweep.py": ["--env", "--ledger", "--audit"],
        "scripts/ft_verify.py": ["--list-loaders", "--json", "--model", "--device", "--version"],
        "scripts/ft_run.py": ["--seed", "--split", "--allow-partial-seed", "--emit-only", "--slurm"],
        "scripts/ft_dataset.py": ["--split", "--seed", "--energy-key", "--force-key"],
    },
    "distill": {
        "scripts/distill_bootstrap.py": [
            "--acceptance", "--mode", "--energy-mae-max", "--force-mae-max", "--target-ps",
            "--max-iter", "--no-progress-limit", "--heldout-file", "--heldout-seed", "--wallclock-max-h",
        ],
        "scripts/distill_verify.py": ["--work", "--json", "--no-lmp-witness"],
    },
}

# Package-management verbs that must never appear in a recipe: the recipe
# invokes install.sh / the env recipes, it never lists or repairs packages.
PACKAGE_COMMANDS = re.compile(
    r"\b(pip3?|python3? -m pip|conda|mamba|micromamba)\s+(install|uninstall|update|upgrade|env create|create)\b"
)

# Consensus-plan Part 3.1 design targets. A name may appear in a recipe ONLY
# below the planned heading, and only while it does not exist on disk.
# Every Part 3.1 helper has landed and been promoted (fresh_root.py,
# ft_sweep.py, catbench_version.py, catbench_datasets.py,
# catbench_vasp_stage.py, catbench_leaderboard.py, evidence_report.py,
# distill_verify.py — all in REQUIRED_OWNER_FILES); the set stays so the
# bidirectional guard is ready for the next design target.
PLANNED_SCRIPTS: set[str] = set()
# Planned flags on scripts that exist today: the script's argparse must not
# add the flag until it is implemented (then the recipe must move the flag
# above the planned heading and this map must drop it).
# Landed and promoted: setup_verify.py --all-variants/--no-local-record,
# setup_sweep.py --fresh-root, distill_bootstrap.py --acceptance,
# catbench_jobgen.py --catbench-version/--calc-file (now in PROMOTED_FLAGS).
PLANNED_FLAGS = {
    "scripts/ft_run.py": ["--ledger"],
    "scripts/gen_finetune.py": [],
}

# Repo-relative references a recipe may advertise. A placeholder segment
# (`scripts/prestage_<env>_weights.py`) is deliberately NOT matched: the
# lookahead refuses a match that runs into `<`.
_REF = re.compile(
    r"\b(?:scripts|run_examples|envs|docs|tests)/[A-Za-z0-9_\-./]+(?![<A-Za-z0-9_\-./])"
    r"|(?<![\w/.])(?:install|env)\.sh\b"
)


def _refs(text: str) -> set[str]:
    out = set()
    for m in _REF.findall(text):
        ref = re.sub(r":[\d\-]+$", "", m.rstrip("."))
        if not ref.endswith("/"):
            out.add(ref)
    return out


def _split_planned(text: str) -> tuple[str, str]:
    idx = text.find(PLANNED_HEADING)
    assert idx >= 0, f"missing heading {PLANNED_HEADING!r}"
    return text[:idx], text[idx:]


def _parsed_flags(script: str) -> set[str]:
    """Flags the script's argparse really adds: string constants handed to any
    `.add_argument(...)` call in the module AST (sub-parsers included). A flag
    that only appears inside a string template or a comment is not parsed."""
    return _parsed_flags_from_source((REPO_ROOT / script).read_text(encoding="utf-8"))


def _parsed_flags_from_source(source: str) -> set[str]:
    tree = ast.parse(source)
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    flags.add(arg.value)
    return flags


def _has_unit_test(script: str) -> bool:
    """Readiness from the working tree: a dedicated tests/test_<stem>*.py."""
    stem = Path(script).stem
    return any(
        p.name not in CONTRACT_TESTS for p in (REPO_ROOT / "tests").glob(f"test_{stem}*.py")
    )


def _untested_list(planned: str) -> set[str]:
    """The scripts a recipe declares as landed without a unit test: the refs in
    the paragraph that follows UNTESTED_MARKER (ends at the first blank line)."""
    idx = planned.find(UNTESTED_MARKER)
    if idx < 0:
        return set()
    para = planned[idx:].split("\n\n", 1)[0]
    return {r for r in _refs(para) if r.startswith("scripts/") and r.endswith(".py")}


def _ngrams(text: str, n: int = 10) -> set[tuple[str, ...]]:
    words = re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


# --- discovery ---------------------------------------------------------------

def test_recipe_files_exist():
    assert SKILL_NAMES, "no skills/*/SKILL.md found"
    assert (RECIPE_DIR / "README.md").is_file(), "recipes/README.md (index + shared protocol) missing"
    missing = [str(p.relative_to(REPO_ROOT)) for p in RECIPES.values() if not p.is_file()]
    assert not missing, f"skills without a recipe: {missing}"


def test_each_skill_points_at_its_recipe():
    for name in SKILL_NAMES:
        skill = (REPO_ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8")
        assert f"recipes/{name}.md" in skill, f"skills/{name}/SKILL.md does not point at recipes/{name}.md"


def test_agents_points_at_every_recipe():
    text = AGENTS.read_text(encoding="utf-8")
    assert "recipes/README.md" in text, "AGENTS.md does not point at recipes/README.md"
    for name in SKILL_NAMES:
        assert f"recipes/{name}.md" in text, f"AGENTS.md does not point at recipes/{name}.md"


def test_readme_index_lists_every_recipe():
    text = (RECIPE_DIR / "README.md").read_text(encoding="utf-8")
    for name in SKILL_NAMES:
        assert f"recipes/{name}.md" in text, f"recipes/README.md index lacks recipes/{name}.md"


# --- structure ---------------------------------------------------------------

def test_recipe_stage_headings_in_order():
    for name, path in RECIPES.items():
        text = path.read_text(encoding="utf-8")
        chain = text.find(CHAIN_HEADING)
        assert chain >= 0, f"{name}: missing {CHAIN_HEADING!r} table"
        positions = []
        for h in STAGE_HEADINGS:
            idx = text.find(h)
            assert idx >= 0, f"{name}: missing stage heading {h!r}"
            positions.append(idx)
        assert chain < positions[0], f"{name}: the executable chain must precede the stages"
        assert positions == sorted(positions), f"{name}: stage headings out of order"
        assert text.find(PLANNED_HEADING) > positions[-1], f"{name}: planned section must come last"


def test_readme_states_the_executable_source_of_truth_rule():
    text = (RECIPE_DIR / "README.md").read_text(encoding="utf-8")
    assert "## Executable source of truth" in text
    assert MISSING_MARKER.lower() in text.lower()


# --- executable source of truth ----------------------------------------------

def test_each_recipe_names_its_owner_files():
    for name, path in RECIPES.items():
        body, _ = _split_planned(path.read_text(encoding="utf-8"))
        absent = [f for f in REQUIRED_OWNER_FILES[name] if f not in body]
        assert not absent, f"{name}: owner files no longer named above the planned heading: {absent}"
        for f in REQUIRED_OWNER_FILES[name]:
            assert (REPO_ROOT / f).exists(), f"{name}: required owner file missing on disk: {f}"


def test_recipes_carry_no_package_management_commands():
    bad = []
    for path in sorted(RECIPE_DIR.glob("*.md")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = PACKAGE_COMMANDS.search(line)
            if m:
                bad.append(f"{path.name}:{i}: {m.group(0)!r}")
    assert not bad, (
        "recipes must invoke install.sh / envs/*.yml, never carry package commands of their own:\n"
        + "\n".join(bad)
    )


def test_each_recipe_lists_its_missing_executable_parts():
    for name, path in RECIPES.items():
        _, planned = _split_planned(path.read_text(encoding="utf-8"))
        assert MISSING_MARKER in planned, f"{name}: planned section lacks a '{MISSING_MARKER}' list"


# --- implemented vs planned ---------------------------------------------------

def test_advertised_paths_exist_and_are_not_planned():
    missing, leaked = [], []
    files = {**RECIPES, "README": RECIPE_DIR / "README.md"}
    for name, path in files.items():
        text = path.read_text(encoding="utf-8")
        body = _split_planned(text)[0] if PLANNED_HEADING in text else text
        for ref in sorted(_refs(body)):
            if ref in PLANNED_SCRIPTS:
                leaked.append(f"{name} -> {ref}")
            elif not (REPO_ROOT / ref).exists():
                missing.append(f"{name} -> {ref}")
    assert not missing, "recipes advertise paths that do not exist (rename drift):\n" + "\n".join(missing)
    assert not leaked, "planned helpers advertised as implemented:\n" + "\n".join(leaked)


def test_planned_scripts_do_not_exist_yet():
    """Bidirectional: the day a planned helper lands, this fails so the recipe
    promotes it above the planned heading and PLANNED_SCRIPTS drops it."""
    landed = [s for s in sorted(PLANNED_SCRIPTS) if (REPO_ROOT / s).exists()]
    assert not landed, f"planned helpers now exist — promote them in the recipes: {landed}"


def test_planned_section_only_names_known_targets():
    unknown = []
    for name, path in RECIPES.items():
        _, planned = _split_planned(path.read_text(encoding="utf-8"))
        for ref in sorted(_refs(planned)):
            if ref in PLANNED_SCRIPTS:
                continue
            if ref in PLANNED_FLAGS and (REPO_ROOT / ref).exists():
                continue
            if (REPO_ROOT / ref).exists():
                # an existing file may be named when describing a gap
                # ("no test fixture for scripts/fresh_root.py in tests/")
                continue
            unknown.append(f"{name} -> {ref}")
    assert not unknown, (
        "planned section names a helper outside the approved Part 3.1 list "
        "(scope approval needed):\n" + "\n".join(unknown)
    )


def test_parsed_flags_ignore_docstrings_comments_and_templates():
    """Regression for the planned-flag check: a flag named only in a docstring,
    a comment, a help string, an f-string template or a plain string constant
    is not "parsed"; only an `.add_argument("--x")` string is."""
    src = (
        '"""usage: tool --ledger FILE   (planned, not implemented)"""\n'
        'import argparse\n'
        '# TODO: add --ledger once the ledger writer lands\n'
        'TEMPLATE = "python3 tool.py --ledger {out}"\n'
        'MSG = f"rerun with --offline"\n'
        'def main():\n'
        '    ap = argparse.ArgumentParser()\n'
        '    ap.add_argument("--real", help="see --ledger for the planned twin")\n'
        '    sub = ap.add_subparsers()\n'
        '    sub.add_parser("x").add_argument("--nested", action="store_true")\n'
    )
    parsed = _parsed_flags_from_source(src)
    assert parsed == {"--real", "--nested"}, parsed
    assert "--ledger" not in parsed and "--offline" not in parsed


def test_planned_flags_not_parsed_yet():
    implemented = []
    for script, flags in PLANNED_FLAGS.items():
        parsed = _parsed_flags(script)
        for flag in flags:
            if flag in parsed:
                implemented.append(f"{script} {flag}")
    assert not implemented, f"planned flags are now parsed — promote them in the recipes: {implemented}"


def test_planned_flags_are_described_as_planned():
    """Every planned flag appears in some recipe, and only below the planned heading."""
    bodies = {n: _split_planned(p.read_text(encoding="utf-8")) for n, p in RECIPES.items()}
    for script, flags in PLANNED_FLAGS.items():
        for flag in flags:
            token = f"{script} {flag}"
            above = [n for n, (b, _) in bodies.items() if token in b]
            below = [n for n, (_, p) in bodies.items() if token in p or (script in p and flag in p)]
            assert not above, f"{token} advertised above the planned heading in {above}"
            assert below, f"{token} is in PLANNED_FLAGS but no recipe lists it as planned"


def test_promoted_flags_are_really_parsed():
    """Every flag a recipe advertises on an existing script (PROMOTED_FLAGS)
    is named above the planned heading AND parsed by that script — the
    mirror image of test_planned_flags_not_parsed_yet."""
    for name, scripts in PROMOTED_FLAGS.items():
        body, _ = _split_planned(RECIPES[name].read_text(encoding="utf-8"))
        for script, flags in scripts.items():
            parsed = _parsed_flags(script)
            for flag in flags:
                assert flag in body, f"{name} recipe no longer advertises {script} {flag}"
                assert flag in parsed, f"{name} recipe advertises {script} {flag} but its argparse does not add it"


# --- readiness (working tree, both directions) --------------------------------

def test_untested_helpers_are_declared_and_nothing_else():
    """Every scripts/*.py a recipe invokes above the planned heading that has
    no dedicated tests/test_<stem>*.py must be named in the recipe's "Landed
    without a unit test" paragraph — and nothing that HAS a test may stay
    there. File presence is not readiness; the recipe says which is which."""
    for name, path in RECIPES.items():
        body, planned = _split_planned(path.read_text(encoding="utf-8"))
        invoked = {r for r in _refs(body) if r.startswith("scripts/") and r.endswith(".py") and (REPO_ROOT / r).exists()}
        untested = {r for r in invoked if not _has_unit_test(r)}
        declared = _untested_list(planned)
        assert declared == untested, (
            f"{name}: '{UNTESTED_MARKER}' list drifted from the tree — "
            f"undeclared: {sorted(untested - declared)}; stale (now tested): {sorted(declared - untested)}"
        )


def test_candidate_local_discovery_is_documented():
    """Both hosts can be pointed at a candidate checkout (a snapshot or fresh
    runtime root) instead of the installed plugin: Claude Code through
    `--plugin-dir`, Codex through the AGENTS.md of its cwd. The docs must say
    so and must not claim the default-installed plugin exercises a candidate."""
    readme = (RECIPE_DIR / "README.md").read_text(encoding="utf-8")
    agents = AGENTS.read_text(encoding="utf-8")
    assert CANDIDATE_FLAG in readme, f"recipes/README.md lacks the {CANDIDATE_FLAG} candidate invocation"
    assert CANDIDATE_FLAG in agents, f"AGENTS.md §10 lacks the {CANDIDATE_FLAG} candidate invocation"


# --- no duplicated bodies / public framing -----------------------------------

def test_no_shared_10gram_between_recipes_and_skills():
    skill_grams: set[tuple[str, ...]] = set()
    for name in SKILL_NAMES:
        skill_grams |= _ngrams((REPO_ROOT / "skills" / name / "SKILL.md").read_text(encoding="utf-8"))
    offenders = {}
    for path in sorted(RECIPE_DIR.glob("*.md")):
        shared = _ngrams(path.read_text(encoding="utf-8")) & skill_grams
        if shared:
            offenders[path.name] = sorted(" ".join(g) for g in shared)[:5]
    assert not offenders, f"skills re-encode recipe bodies (shared 10-grams): {offenders}"


def _brand_patterns():
    spec = importlib.util.spec_from_file_location(
        "verify_readme_brand", REPO_ROOT / "scripts" / "verify_readme_brand.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HARDWARE, mod.MEASURED


def test_recipes_and_skills_pass_the_brand_gates():
    """The README brand regexes (no hardware product names, no measured-result
    patterns) applied to every recipe and skill — the same public-framing rule."""
    hardware, measured = _brand_patterns()
    files = sorted(RECIPE_DIR.glob("*.md")) + [REPO_ROOT / "skills" / n / "SKILL.md" for n in SKILL_NAMES]
    findings = []
    for path in files:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            rel = path.relative_to(REPO_ROOT)
            m = hardware.search(line)
            if m:
                findings.append(f"{rel}:{i}: hardware product name {m.group(0)!r}")
            for pat in measured:
                m = pat.search(line)
                if m:
                    findings.append(f"{rel}:{i}: measured-result pattern {m.group(0)!r}")
    assert not findings, "\n".join(findings)


def test_recipes_carry_no_personal_paths_or_validated_only_claims():
    bad = []
    for path in sorted(RECIPE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for needle in ("/home/", "validated only on"):
            if needle in text:
                bad.append(f"{path.name}: {needle!r}")
    assert not bad, f"public recipes must not carry personal paths or hardware-scoped claims: {bad}"


def test_recipes_carry_no_ad_hoc_clone_or_copy_commands():
    """A candidate is validated from a content-addressed working-tree snapshot
    (scripts/fresh_root.py), never from `git clone` of HEAD; a VASP tree is
    staged by scripts/catbench_vasp_stage.py, never by an rsync line whose
    symlink handling the recipe would have to get right."""
    bad = []
    for path in sorted(RECIPE_DIR.glob("*.md")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for needle in ("git clone", "rsync "):
                if needle in line:
                    bad.append(f"{path.name}:{i}: {needle!r}")
    assert not bad, "\n".join(bad)
