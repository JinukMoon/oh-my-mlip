#!/usr/bin/env python3
"""Compile the per-GPU-arch Allegro .pt2 (same flow as prepare_nequip_weights.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_nequip_weights import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
