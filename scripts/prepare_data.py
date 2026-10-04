from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import load_config
from moons_lab.data import prepare_data


def main() -> None:
    cfg = load_config(1)
    manifest = prepare_data(cfg)
    print(manifest)


if __name__ == "__main__":
    main()
