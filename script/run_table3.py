"""Train Table III controls and immediately run the shared sky evaluator."""
from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from src.util.ablation_tables import main
    raise SystemExit(main(3))
