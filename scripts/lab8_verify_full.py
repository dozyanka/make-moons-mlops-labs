"""Independent, read-only LAB8 full-evidence audit. No fake-duration shortcuts.

Checks saved HTTP request rows, clock span, CPU/RAM cadence, drift/alerts,
incidents, installation provenance, E2E repeats and SHA-256 manifest.
The --seal flag is reserved for the local submission installer after the run
has FINISHED and its stdout transcript has been closed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

STAGES = (
    "warmup", "clean_ml_sequential", "clean_dl_parallel", "clean_dl_variable_rps",
    "drift_dl_sequential", "rollback_under_load",
)


class EvidenceError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def read_json(path: Path):
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    require(path.is_file(), f"missing {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"missing {path}")
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(root: Path, report: Path) -> int:
    manifest = read_json(report / "artifacts_manifest.json")
    entries = list(manifest.get("items", [])) + list(manifest.get("runtime_evidence", []))
    require(manifest.get("count") == len(entries) and len(entries) > 20,
            "manifest count mismatch/too few source files")
    paths = [item["path"] for item in entries]
    require(len(paths) == len(set(paths)), "manifest contains duplicate paths")
    require(".gitattributes" in paths, "missing Linguist/source provenance .gitattributes")
    require("reports/LAB8/summary.json" in paths, "missing LAB8 summary in manifest")
    for item in entries:
        rel = Path(item["path"])
        path = rel if rel.is_absolute() else root / rel
        require(path.is_file(), f"manifest file missing: {item['path']}")
        require(path.stat().st_size == item["bytes"], f"manifest size mismatch: {item['path']}")
        require(sha(path) == item["sha256"], f"manifest SHA mismatch: {item['path']}")
    return len(entries)


def verify_full(root: Path, report: Path, *, check_manifest=True) -> dict:
    root = root.resolve()
    report = report.resolve()
    s = read_json(report / "summary.json")
    require(s.get("evidence_mode") == "full", "SMOKE_ONLY is not final evidence")
    require(s.get("tracking_backend") == "mlflow" and s.get("mlflow_run_id"),
            "real MLflow experiment not recorded")
    require(s.get("all_pass") is True, "full-run acceptance is FAIL")
    require(s.get("false_alerts_on_clean", 999) <= 1, "too many clean drift alerts")
    require(s.get("drift_alert_count") == 1, "drift phase must have exactly one active drift alert")
    require(s.get("drift_delay_windows") is not None and
            0 <= s["drift_delay_windows"] <= 2, "drift detected outside 2-window deadline")
    stages = s.get("stage_results", [])
    require([v.get("stage") for v in stages] == list(STAGES), "required six operational stages missing")
    require(all(v.get("pass") is True for v in stages), "at least one stage FAIL")

    checks = {}
    for item in stages:
        name = item["stage"]
        reqs = read_csv(report / f"{name}_requests.csv")
        minimum_sec = 300 if name == "warmup" else 600
        minimum_req = 5000 if name == "warmup" else 10000
        require(item["elapsed_s"] >= minimum_sec, f"{name}: duration below {minimum_sec}s")
        require(len(reqs) >= minimum_req and len(reqs) == item["total"],
                f"{name}: insufficient/mismatched real request rows")
        success = [r for r in reqs if r["success"].lower() == "true"]
        require(len(success) == len(reqs) == item["successful"] and item["lost_or_failed"] == 0,
                f"{name}: lost or unsuccessful HTTP requests")
        require(item["success_fraction"] >= .999, f"{name}: success ratio")
        require(item["http_transport"] == "persistent_http_1_1_loopback", f"{name}: HTTP transport")
        t0 = datetime.fromisoformat(reqs[0]["time_utc"])
        t1 = datetime.fromisoformat(reqs[-1]["time_utc"])
        span = (t1 - t0).total_seconds()
        require(span >= minimum_sec - 5, f"{name}: requests do not span actual duration ({span}s)")
        require(span <= item["elapsed_s"] + 3, f"{name}: request timestamp span invalid")
        lat = np.asarray([float(r["latency_ms"]) for r in success], dtype=float)
        require(np.isfinite(lat).all() and (lat >= 0).all(), f"{name}: non-finite latency")
        calculated_p95 = float(np.quantile(lat, .95))
        require(abs(calculated_p95 - item["p95_ms"]) < .02,
                f"{name}: p95 mismatch: CSV={calculated_p95:.4f}, JSON={item['p95_ms']:.4f}")
        if name != "warmup":
            require(calculated_p95 <= item["latency_limit_ms"] and
                    math.isclose(item["latency_limit_ms"],
                                 1.5 * item["baseline_p95_ms"], rel_tol=1e-7),
                    f"{name}: p95 exceeds original LAB6 ratio 1.5")
        resources = read_csv(report / f"{name}_resources.csv")
        points = [float(r["elapsed_s"]) for r in resources]
        require(len(points) >= 100 and points[0] <= 2.1 and
                points[-1] >= item["elapsed_s"] - 2.1, f"{name}: CPU/RAM samples incomplete")
        gaps = np.diff(points)
        require((gaps > 0).all() and float(max(gaps)) <= 2.1,
                f"{name}: telemetry gap >2.1 seconds")
        for row in resources:
            require(float(row["rss_mb"]) >= 0 and math.isfinite(float(row["process_cpu_pct"])),
                    f"{name}: invalid CPU/RAM sample")
        for suffix in ("cpu", "memory"):
            require((report / f"{name}_resources_{suffix}.png").stat().st_size > 1000,
                    f"{name}: {suffix} plot missing")
        if name in ("clean_dl_parallel", "clean_dl_variable_rps"):
            p = item.get("parallel_evidence", {})
            require(p.get("requested_workers") == p.get("available_logical_cores") and
                    p.get("requested_workers", 0) >= 2 and
                    p.get("simultaneous_executor_threads") == p["requested_workers"] and
                    p.get("parallel_batches_completed", 0) >= minimum_req,
                    f"{name}: all-core worker proof missing")
        if name == "clean_dl_variable_rps":
            minutes = {int(r["minute"]) for r in reqs}
            rates = {round(float(r["target_rps"]), 4) for r in reqs}
            require(len(minutes) >= 10 and len(rates) >= 2,
                    f"{name}: per-minute RPS variation not documented")
        if name == "drift_dl_sequential":
            rejected = sum(int(r["rejected_count"]) for r in reqs)
            require(item["validator_total_samples"] == sum(int(r["batch_size"]) for r in reqs)
                    and rejected == item["validator_rejected_samples"],
                    "drift: mismatched validator reject counters")
            require(item["validator_reject_rate"] >= .90 and item["silent_accepts"] == 0,
                    "drift: validator reject/quiet acceptance threshold")
        if name == "rollback_under_load":
            require(item["rollback_status"] == 200 and
                    set(item["versions_observed"]) == {"v2", "v3_dl"}, "rollback not observed")
        checks[name] = {"requests": len(reqs), "duration_s": round(item["elapsed_s"], 3),
                        "p95_ms": round(calculated_p95, 3), "telemetry": len(points)}

    windows = read_csv(report / "stream_windows.csv")
    require(len(windows) >= 10, "missing streamed drift windows")
    drift = [r for r in windows if r["phase"] == "drift_dl_sequential"]
    require(sum(r["alert"].lower() == "true" for r in drift) == 1,
            "drift has duplicate active alerts")
    alerts = read_jsonl(report / "alerts_sample.jsonl")
    fields = {"time", "metric", "value", "threshold", "level", "action"}
    require(all(fields.issubset(event) for event in alerts), "alert lacks required fields")
    require(sum(e["metric"] == "max_psi_csi" for e in alerts) >= 1,
            "no PSI/CSI drift alert recorded")
    require(any(e["metric"] == "validator_reject_rate" for e in alerts),
            "no validator degradation alert recorded")
    require(any(e["metric"] == "service_up" for e in alerts),
            "no service failure alert recorded")

    faults = s.get("faults", [])
    recorded_faults = read_json(report / "faults.json")
    require([i.get("case") for i in faults] == [f"C{i}" for i in range(1, 6)] and
            recorded_faults == faults and all(i.get("pass") is True for i in faults),
            "C1-C5 incidents incomplete/failed")
    incidents = read_jsonl(report / "incidents_log.jsonl")
    require([v.get("case") for v in incidents] == [f"C{i}" for i in range(1, 6)],
            "incident log missing C1-C5")
    require(faults[0]["alert_detected_s"] <= 30 and faults[0]["recovery_s"] <= 120 and
            all(x == 200 for x in faults[0]["smoke_codes"]), "C1 recovery timeout")
    require(faults[1]["health_code"] == 200 and faults[1]["batch_code"] == 200,
            "C2 process worker recovery failed")
    require(faults[2]["promote_status"] == 409 and faults[2]["prediction_status"] == 200,
            "C3 model fingerprint gate failed")
    require(faults[3]["gap_events"] and faults[3]["recovered_within_windows"] is not None,
            "C4 stream checkpoint/gap evidence missing")
    require(faults[4]["invalid_returncode"] != 0 and faults[4]["valid_returncode"] == 0,
            "C5 invalid configuration not rejected")

    e2e = read_csv(report / "end_to_end.csv")
    require({(r["source"], r["repeat"]) for r in e2e} ==
            {(src, str(rep)) for src in ("real", "synthetic") for rep in (1, 2)},
            "two real/synthetic E2E repeats missing")
    for src in ("real", "synthetic"):
        a, b = (r for r in e2e if r["source"] == src)
        for metric in ("ml_accuracy", "dl_accuracy", "ml_auc", "dl_auc", "validator_accept_rate"):
            require(abs(float(a[metric]) - float(b[metric])) <= .005,
                    f"E2E reproducibility mismatch: {src}/{metric}")
    log = (report / "install.log").read_text(encoding="utf-8")
    require("RESULT: PASS" in log and "seven CLI entrypoints present" in log and
            "pip freeze --all" in log, "fresh venv, seven CLI commands or pip freeze missing")
    require((report / "full_run.log").is_file() and (report / "full_run.log").stat().st_size > 100,
            "complete full-run console transcript missing")
    require((report / "lab8_report.md").is_file() and
            "all_pass=True" in (report / "lab8_report.md").read_text(encoding="utf-8"),
            "report missing PASS conclusion")
    report_text = (report / "lab8_report.md").read_text(encoding="utf-8")
    code_match = re.search(r"Коммит кода:\*\*\s*`([a-f0-9]{40})`", report_text)
    require(code_match is not None, "no auditable code SHA in lab8_report.md")
    ancestry = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor",
                               code_match.group(1), "HEAD"], capture_output=True, text=True)
    require(ancestry.returncode == 0, "recorded code commit is not an ancestor of the evidence")

    checked_files = verify_manifest(root, report) if check_manifest else None
    return {"verdict": "PASS", "evidence_mode": "full", "stages": checks,
            "alerts": len(alerts), "incident_cases": 5, "manifest_files": checked_files,
            "mlflow_run_id": s["mlflow_run_id"]}


def seal(root: Path, report: Path) -> None:
    from moons_lab.lab8_observability import hash_manifest
    # Regenerate only after all run logs have been CLOSED. No success is implied.
    target = report / "artifacts_manifest.json"
    target.write_text(json.dumps(hash_manifest(root, target), indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports/LAB8")
    p.add_argument("--seal", action="store_true", help="seal closed full-run logs before read-only audit")
    args = p.parse_args()
    try:
        if args.seal:
            seal(ROOT, args.report_dir.resolve())
        result = verify_full(ROOT, args.report_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"LAB8 FULL EVIDENCE FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
