from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + ENV["PYTHONPATH"] if ENV.get("PYTHONPATH") else "")


def run(*parts: str, log: Path | None = None) -> None:
    cmd = [sys.executable, *parts]
    print("+", " ".join(cmd), flush=True)
    if log is None:
        subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)
    else:
        proc = subprocess.run(cmd, cwd=ROOT, env=ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log.write_text(proc.stdout, encoding="utf-8")
        if proc.returncode:
            print(proc.stdout)
            raise SystemExit(proc.returncode)


def main() -> None:
    for lab in range(1, 9):
        run(f"scripts/run_lab{lab}.py")
    run("-m", "pytest", "--cov=moons_lab", "--cov-report=term-missing", log=ROOT / "reports/LAB8/all_tests.log")
    run("scripts/build_coverage_logs.py")
    run("scripts/build_reports.py")
    run("scripts/update_manifest.py")
    run("scripts/precommit_checks.py", "--all")
    print("All labs completed successfully.")


if __name__ == "__main__":
    main()
