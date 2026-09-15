"""The public library, docs and README are English-only: no Hangul in tracked text files."""

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HANGUL = re.compile("[\\u1100-\\u11ff\\u3131-\\u318e\\uac00-\\ud7a3]")


def _tracked_files():
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout
    return [REPO / p for p in out.decode().split("\0") if p]


def test_no_hangul_in_tracked_text_files():
    offenders = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
            continue  # binary (images) or removed in the working tree
        for lineno, line in enumerate(text.splitlines(), 1):
            if HANGUL.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}")
    assert not offenders, "Korean text found:\n" + "\n".join(offenders)
