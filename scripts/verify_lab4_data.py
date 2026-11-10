"""LAB4 publication verifier: byte-exact inputs, Git index and report provenance.

No third-party dependencies. No data, model, or MLflow store changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "LAB4" / "data_integrity_baseline.json"
REPORTS = ROOT / "reports" / "LAB4"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def verify_data() -> dict[str, object]:
    manifest = json.loads(BASELINE.read_text(encoding="utf-8"))
    expectations = dict(manifest["files"])
    model = manifest["working_model"]
    expectations[model["path"]] = model
    for rel, expected in expectations.items():
        file = ROOT / rel
        if not file.is_file():
            raise AssertionError(f"MISSING input: {rel}")
        contents = file.read_bytes()
        if len(contents) != expected["bytes"] or sha256(contents) != expected["sha256"]:
            raise AssertionError(
                f"BYTE HASH MISMATCH: {rel}: actual={sha256(contents)} expected={expected['sha256']}; "
                "do not rewrite datasets; restore known-good files before publication"
            )
    print(f"PASS: {len(manifest['files'])} original CSV + LAB2 model SHA-256 and sizes")
    return manifest


def verify_git_index(manifest: dict[str, object]) -> None:
    for rel in list(manifest["files"]) + [manifest["working_model"]["path"]]:
        actual = git("show", f":{rel}")
        expectation = manifest["files"].get(rel, manifest["working_model"])
        if sha256(actual) != expectation["sha256"]:
            raise AssertionError(
                f"GIT INDEX CHANGED FILE BYTES: {rel}; check .gitattributes (CSV must be -text)"
            )
    print("PASS: Git index stores original input bytes (no LF/CRLF conversion)")


def verify_published(expected_code_commit: str | None = None) -> None:
    report = REPORTS / "lab4_report.md"
    source = json.loads((REPORTS / "source_manifest.json").read_text(encoding="utf-8"))
    actual_commit = str(source["git_head_at_run"])
    if expected_code_commit and actual_commit != expected_code_commit:
        raise AssertionError(
            f"report was produced from {actual_commit}, expected code commit {expected_code_commit}"
        )
    if len(actual_commit) != 40 or not all(c in "0123456789abcdef" for c in actual_commit.lower()):
        raise AssertionError(f"report contains an invalid git code commit: {actual_commit}")
    # A report commit is a descendant of the preceding code commit.
    subprocess.run(["git", "merge-base", "--is-ancestor", actual_commit, "HEAD"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    code_bytes = json.dumps(source["files_sha256"], sort_keys=True).encode("utf-8")
    if sha256(code_bytes) != source["combined_sha256"]:
        raise AssertionError("source_manifest combined SHA does not match per-file SHA values")
    for rel, recorded_sha in source["files_sha256"].items():
        # Compare against the audited Git snapshot used to write the report.
        # Later changes to the runner do not invalidate the historic experiment.
        committed = git("show", f"{actual_commit}:{rel}")
        if sha256(committed) != recorded_sha:
            raise AssertionError(
                f"Git code commit differs from recorded source bytes: {rel}. "
                "Check Git text normalization or modify .gitattributes and retry."
            )
    body = report.read_text(encoding="utf-8")
    if actual_commit not in body:
        raise AssertionError("lab4_report.md does not reference the committed code SHA")
    repeat = json.loads((REPORTS / "repeat_verification.json").read_text(encoding="utf-8"))
    comparison = json.loads((REPORTS / "comparison.json").read_text(encoding="utf-8"))
    if not repeat.get("all_pass") or not repeat.get("source_hashes_unchanged"):
        raise AssertionError("repeat verification has not passed")
    if not all(comparison.get(k) for k in
               ("stats_within_tolerance", "metrics_within_tolerance", "quality_pass")):
        raise AssertionError("full-vs-chunks comparison or quality did not pass")
    with (REPORTS / "validator_metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("mode") for row in rows} != {"full", "chunks"}:
        raise AssertionError("validator_metrics.csv must contain full and chunks")
    if any(row.get("tracking_backend") != "mlflow" or
           len(row.get("run_id", "")) != 32 for row in rows):
        raise AssertionError("validator_metrics.csv contains invalid or offline MLflow Run IDs")
    for mode in ("full", "chunks"):
        r = repeat.get(mode, {})
        if (not r.get("pass") or r.get("tracking_backend") != "mlflow" or
                len(r.get("repeat_run_id", "")) != 32):
            raise AssertionError(f"independent MLflow repeat for {mode} has not passed")
    for rel in ("validator_params.npz", "validator_params_chunk.npz", "validator_negative.log",
                "pytest_coverage.log", "coverage_details.json", "negative_summary.json"):
        if not (REPORTS / rel).is_file():
            raise AssertionError(f"missing report artifact: {rel}")
    coverage = (REPORTS / "pytest_coverage.log").read_text(encoding="utf-8-sig")
    if "passed" not in coverage or "TOTAL" not in coverage:
        raise AssertionError("pytest coverage log is missing a successful test summary")
    print(f"PASS: committed code {actual_commit}, source SHA-256, 4 MLflow Run IDs, reports, repeat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-git-index", action="store_true")
    parser.add_argument("--check-published", action="store_true")
    parser.add_argument("--expected-code-commit")
    args = parser.parse_args()
    inputs = verify_data()
    if args.check_git_index:
        verify_git_index(inputs)
    if args.check_published:
        verify_published(args.expected_code_commit)


if __name__ == "__main__":
    main()
