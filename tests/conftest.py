import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
for p in (RAIZ, RAIZ / "model"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
