from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import load_config
from moons_lab.data import prepare_data


def main() -> None:
    cfg = load_config(1)
    manifest_path = (
        os.environ.get("MOONS_DEMO_MANIFEST_PATH")
        or "reports/LAB1/hash_manifest.json"
    )
    manifest = prepare_data(cfg, manifest_path=manifest_path)
    print(manifest)


if __name__ == "__main__":
    main()
