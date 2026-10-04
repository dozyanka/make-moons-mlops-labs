from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_lab8 import build_artifact_manifest

if __name__ == "__main__":
    import json
    path = ROOT / "reports/LAB8/artifacts_manifest.json"
    path.write_text(json.dumps(build_artifact_manifest(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
