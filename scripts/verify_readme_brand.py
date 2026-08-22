#!/usr/bin/env python3
"""verify_readme_brand.py — README brand gates (GPU-free, CI).

The README speaks about value and purpose, assuming ideal operation. Three
gates keep it that way:

  1. no hardware PRODUCT names (vendor cards). GPU *architecture codes*
     (sm86/sm89) are part of the arch-pinning contract and stay allowed.
  2. no measured results: numbers glued to physical units, benchmark fractions
     like ``27/31``, or host-count claims — a README that quotes yesterday's
     measurements is stale tomorrow; verified state lives in the generated
     status docs instead.
  3. the generated status-table markers survive every rewrite (the
     gen_status_table --check contract depends on them).

Exit 0 iff all gates pass. Findings are printed one per line.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

HARDWARE = re.compile(
    r"\b(RTX\s?\w*|GTX\s?\w*|A[456]000|A4500|L40S?|H100|A100|V100|T4|4060\s?Ti|"
    r"GeForce|Quadro|Xeon|EPYC|Threadripper)\b"
)
MEASURED = [
    # number + physical/benchmark unit
    re.compile(r"\b\d+(?:\.\d+)?\s*(meV|eV(?:/[ÅA])?|ms|µs|GB\s+VRAM|ps/day|x\s+faster)\b"),
    # benchmark fractions: 27/31, 17/20 ... (calendar-looking 1/2 etc. still match — keep them out of the README)
    re.compile(r"\b\d{1,3}\s*/\s*\d{1,3}\b(?!\d)"),
    # host-count result claims
    re.compile(r"\b(two|three|four|\d+)\s+(independent\s+)?hosts?\b", re.I),
    # MAE/percentage results
    re.compile(r"\b(MAE|ADwT|AMDwT)\s*[:=]?\s*\d"),
]
MARKERS = ["<!-- STATUS_TABLE_START -->", "<!-- STATUS_TABLE_END -->"]


def main() -> int:
    text = README.read_text(encoding="utf-8")
    findings: list[str] = []

    for i, line in enumerate(text.splitlines(), 1):
        m = HARDWARE.search(line)
        if m:
            findings.append(f"README.md:{i}: hardware product name {m.group(0)!r}")
        for pat in MEASURED:
            m = pat.search(line)
            if m:
                findings.append(f"README.md:{i}: measured-result pattern {m.group(0)!r}")

    for marker in MARKERS:
        if marker not in text:
            findings.append(f"README.md: missing required marker {marker}")

    for f in findings:
        print(f)
    print(f"verify_readme_brand: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
