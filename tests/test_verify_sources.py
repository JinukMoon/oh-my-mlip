"""Offline tests for scripts/verify_sources.py.

Every network call is mocked, so this suite needs no network, GPU, conda env,
torch, or ase. It pins the status mapping:

  - torch wheel index 200 + version present  -> line ok
  - PyPI 200 with a wheel                     -> line ok
  - PyPI 200 sdist-only                       -> flagged sdist-only
  - PyPI 404                                  -> recipe fail (clean recipe)
  - git repo HEAD 200                         -> line ok
  - file:// in a clean recipe install line    -> recipe fail
  - file:// noted only in a candidate comment -> NOT fail (candidate)
  - file:// in a candidate install line       -> fail regardless
  - exit code: 0 when no fail, 1 when any fail
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_sources as vs  # noqa: E402


# ── fetcher fixtures ─────────────────────────────────────────────────────────

def make_fetcher(*, pypi_status=200, pypi_has_wheel=True, torch_status=200,
                 torch_body=None, git_status=200, index_status=200):
    """Build a fetcher with per-host configurable responses."""
    import json as _json

    def fetcher(method, url):
        if "pypi.org/pypi/" in url and url.endswith("/json"):
            if pypi_status != 200:
                return pypi_status, ""
            urls = (
                [{"packagetype": "bdist_wheel", "filename": "x.whl"}]
                if pypi_has_wheel
                else [{"packagetype": "sdist", "filename": "x.tar.gz"}]
            )
            return 200, _json.dumps({"urls": urls, "releases": {}})
        if "download.pytorch.org/whl/" in url:
            body = "" if torch_body is None else torch_body
            return torch_status, body
        if "github.com" in url or url.startswith("https://") and "huggingface" not in url:
            # treat anything else (git repos, index urls) by status knobs below
            pass
        if method == "HEAD":
            return git_status, ""
        return index_status, ""

    return fetcher


# ── single-line classification ───────────────────────────────────────────────

def test_torch_line_ok_when_version_present():
    f = make_fetcher(torch_status=200, torch_body="torch-2.7.1+cu126-...whl")
    lr = vs._verify_line("torch==2.7.1+cu126", f)
    assert lr.kind == "torch" and lr.status == "ok"


def test_torch_line_ok_empty_body_cannot_disprove():
    f = make_fetcher(torch_status=200, torch_body="")
    lr = vs._verify_line("torch==2.7.1+cu126", f)
    assert lr.status == "ok"


def test_torch_line_fail_when_version_absent():
    f = make_fetcher(torch_status=200, torch_body="torch-9.9.9+cu126.whl")
    lr = vs._verify_line("torch==2.7.1+cu126", f)
    assert lr.status == "fail"


def test_torch_line_fail_on_index_404():
    f = make_fetcher(torch_status=404)
    lr = vs._verify_line("torch==2.7.1+cu126", f)
    assert lr.status == "fail"


def test_pypi_wheel_ok():
    f = make_fetcher(pypi_status=200, pypi_has_wheel=True)
    lr = vs._verify_line("catbench==1.1.2", f)
    assert lr.kind == "pypi" and lr.status == "ok"


def test_pypi_sdist_only_flagged():
    f = make_fetcher(pypi_status=200, pypi_has_wheel=False)
    lr = vs._verify_line("openequivariance==0.6.4", f)
    assert lr.status == "sdist-only"


def test_pypi_404_fails():
    f = make_fetcher(pypi_status=404)
    lr = vs._verify_line("does-not-exist==1.0", f)
    assert lr.status == "fail" and "404" in lr.detail


def test_local_version_is_extra_index_not_pypi():
    f = make_fetcher()
    lr = vs._verify_line("torch_scatter==2.1.2+pt29cu126", f)
    assert lr.status == "extra-index"


def test_git_line_ok():
    f = make_fetcher(git_status=200)
    lr = vs._verify_line(
        "sevenn @ git+https://github.com/MDIL-SNU/SevenNet.git@deadbeef", f
    )
    assert lr.kind == "git" and lr.status == "ok"


def test_git_line_fail_on_404():
    f = make_fetcher(git_status=404)
    lr = vs._verify_line(
        "pkg @ git+https://github.com/nope/nope.git@abc123", f
    )
    assert lr.status == "fail"


def test_extra_index_url_ok():
    f = make_fetcher(index_status=200)
    lr = vs._verify_line(
        "--extra-index-url https://download.pytorch.org/whl/cu126", f
    )
    assert lr.kind == "index" and lr.status == "ok"


def test_file_url_line_fails():
    f = make_fetcher()
    lr = vs._verify_line("tensorpotential @ file:///home/x/grace", f)
    assert lr.kind == "local" and lr.status == "fail"


def test_local_path_line_fails():
    f = make_fetcher()
    lr = vs._verify_line("/home/jumoon/pretrained_models/tace", f)
    assert lr.kind == "local" and lr.status == "fail"


# ── recipe-level status mapping (write tiny recipes to tmp) ───────────────────

def _write_recipe(tmp_path: Path, name: str, build_status: str, pip_lines: list[str]) -> Path:
    lines = "\n".join(f"      - {l}" for l in pip_lines)
    text = textwrap.dedent(f"""\
        # build_status: {build_status}
        name: {name}
        channels:
          - conda-forge
        dependencies:
          - python=3.11
          - pip
          - pip:
        """) + lines + "\n"
    p = tmp_path / f"{name}.yml"
    p.write_text(text, encoding="utf-8")
    return p


def test_clean_recipe_ok(tmp_path):
    p = _write_recipe(
        tmp_path, "good", "clean",
        ["--extra-index-url https://download.pytorch.org/whl/cu126",
         "torch==2.7.1+cu126", "catbench==1.1.2"],
    )
    f = make_fetcher(torch_body="", pypi_has_wheel=True)
    r = vs.verify_recipe(p, f)
    assert r.status == "ok"


def test_clean_recipe_fails_on_404(tmp_path):
    p = _write_recipe(tmp_path, "bad", "clean", ["ghost-pkg==1.0"])
    f = make_fetcher(pypi_status=404)
    r = vs.verify_recipe(p, f)
    assert r.status == "fail"


def test_clean_recipe_with_file_url_install_line_fails(tmp_path):
    p = _write_recipe(
        tmp_path, "dirty", "clean",
        ["torch==2.7.1+cu126", "tensorpotential @ file:///home/x/tp"],
    )
    f = make_fetcher(torch_body="")
    r = vs.verify_recipe(p, f)
    assert r.status == "fail"


def test_candidate_with_404_is_not_fail(tmp_path):
    """A candidate's unresolved (404) pip line is expected, not a failure."""
    p = _write_recipe(tmp_path, "cand", "candidate", ["ghost-pkg==1.0"])
    f = make_fetcher(pypi_status=404)
    r = vs.verify_recipe(p, f)
    assert r.status == "candidate"
    assert any(l.status == "expected" for l in r.lines)


def test_candidate_with_file_url_comment_only_not_fail(tmp_path):
    """file:// appearing ONLY in a header comment must not fail a candidate."""
    text = textwrap.dedent("""\
        # build_status: candidate
        # candidate-reason: tensorpotential is a local file:// build, owner needed
        name: cand2
        dependencies:
          - pip
          - pip:
              - torch==2.5.1+cu121
              - catbench==1.1.2
        """)
    p = tmp_path / "cand2.yml"
    p.write_text(text, encoding="utf-8")
    f = make_fetcher(torch_body="", pypi_has_wheel=True)
    r = vs.verify_recipe(p, f)
    assert r.status == "candidate"


def test_candidate_with_file_url_install_line_fails_regardless(tmp_path):
    """file:// in an actual install line fails even for a candidate recipe."""
    p = _write_recipe(
        tmp_path, "cand3", "candidate",
        ["torch==2.5.1+cu121", "tp @ file:///home/x/tp"],
    )
    f = make_fetcher(torch_body="")
    r = vs.verify_recipe(p, f)
    assert r.status == "fail"


# ── exit codes via verify_all over a tmp dir ─────────────────────────────────

def test_exit_zero_when_no_fail(tmp_path):
    _write_recipe(tmp_path, "a", "clean",
                  ["torch==2.7.1+cu126", "catbench==1.1.2"])
    _write_recipe(tmp_path, "b", "candidate", ["ghost==1.0"])
    f = make_fetcher(torch_body="", pypi_status=200, pypi_has_wheel=True)
    # candidate 'b' has a resolvable pkg here (200), still candidate -> no fail
    results = vs.verify_all(tmp_path, f)
    assert all(r.status != "fail" for r in results)


def test_real_envs_dir_all_pass_with_permissive_fetcher():
    """The committed envs/ recipes must produce zero 'fail' under a permissive
    (all-200, wheel-backed) fetcher — proving no install line is an unresolved
    local path / file:// in a clean recipe."""
    f = make_fetcher(torch_body="", pypi_status=200, pypi_has_wheel=True,
                     git_status=200, index_status=200)
    results = vs.verify_all(vs.ENVS_DIR, f)
    failed = [r.name for r in results if r.status == "fail"]
    assert not failed, f"unexpected fail recipes: {failed}"


def test_main_returns_zero_in_mock_mode():
    assert vs.main(["--mock"]) == 0


def test_main_returns_one_when_fail(tmp_path, capsys):
    _write_recipe(tmp_path, "broken", "clean",
                  ["pkg @ file:///home/x/pkg"])
    # Use a real-ish run by pointing --envs-dir at tmp and forcing mock fetcher
    # via monkeypatching is overkill; mock mode never produces file:// fails on
    # the real envs, so we exercise main() with a custom envs dir + mock.
    rc = vs.main(["--mock", "--envs-dir", str(tmp_path)])
    assert rc == 1
