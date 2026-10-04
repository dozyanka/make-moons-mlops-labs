from pathlib import Path
import hashlib
import json
import shutil

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression


root = Path.cwd()
demo = root / "defense_demo"
models = demo / "models"
models.mkdir(parents=True, exist_ok=True)

raw = pd.read_csv(root / "data" / "raw" / "moons.csv")
x = raw[["x1", "x2"]].to_numpy(float)
y = raw["y"].to_numpy(int)

lab1 = LogisticRegression(max_iter=2000)
lab1.fit(x, y)
joblib.dump(lab1, models / "lab1_logistic.joblib")

copies = {
    root / "reports/LAB2/ml_model.joblib":
        models / "lab2_full.joblib",
    root / "reports/LAB2/ml_model_chunk.joblib":
        models / "lab2_chunks.joblib",
    root / "reports/LAB3/dl_model.pt":
        models / "lab3_full.pt",
    root / "reports/LAB3/dl_model_chunk.pt":
        models / "lab3_streaming.pt",
}

for source, target in copies.items():
    if not source.exists():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)

manifest = {}
for path in sorted(models.iterdir()):
    if not path.is_file():
        continue
    manifest[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }

(models / "manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("Live model files prepared:")
for name, info in manifest.items():
    print(name, info["bytes"], info["sha256"])
