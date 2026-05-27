from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omnicoder.eval.decode_diagnostics_2026 import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
