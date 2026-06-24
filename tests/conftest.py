"""Put the repo root on sys.path so `import oh_my_mlip` works without pip
(the package is path-importable by design — no setup.py)."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
