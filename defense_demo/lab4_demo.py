"""Standalone LAB4 demo: load working model and NumPy validator, no retraining."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from moons_lab.config import load_config
from moons_lab.data import load_xy
from moons_lab.validator import load_validator, make_ood_samples


def main() -> None:
    cfg = load_config(4, ROOT / "configs/lab4.json")
    m = joblib.load(ROOT / cfg.working_model_path)
    v = load_validator(ROOT / "reports/LAB4/validator_params.npz")
    xt, yt = load_xy(ROOT / cfg.test_path)
    xood = make_ood_samples(cfg.ood_samples, cfg.ood_shift, cfg.seed)
    points = [("normal", xt[0:1], None),
              ("shifted", xood[0:1], None),
              ("outlier", xood[cfg.ood_samples//2:cfg.ood_samples//2+1], None),
              ("label_flip_with_claim", xt[0:1], 1-yt[0:1])]
    print("=== LAB4 / Make Moons validator demo ===")
    for kind, x, claimed in points:
        p = np.asarray(m.predict_proba(x))
        p = p[:, 1] if p.ndim == 2 else p
        d = v.explain(x, p, claimed)[0]
        print(json.dumps({"kind": kind, "x": x.tolist(), "p1": float(p[0]),
                          "claimed_label": None if claimed is None else int(claimed[0]),
                          **d}, ensure_ascii=False))
    print(json.dumps({"kind": "unknown_category", "result": v.validate_record({"x1": "alien", "x2": 0.5})},
                     ensure_ascii=False))
    print("Reports: reports/LAB4/lab4_report.md, validator_metrics.csv, validator_negative.log")


if __name__ == "__main__":
    main()
