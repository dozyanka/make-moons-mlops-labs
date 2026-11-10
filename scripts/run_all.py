"""Safe reproducibility launcher for LAB1-7 and optional LAB8.

It delegates to the same isolated DEMO routines as RUN_DEMO.cmd. Published
reports and baseline training data are restored after LAB1-4 reruns.
Historical LAB8 SHA-256 evidence is NEVER rebuilt from the current tree.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DEMO import menu

ORIGINAL_PR36_SHA = "f0dbbc537270ab036156c65a9ed14ad6d6b83ba6"


def verify_historical_lab8() -> None:
    """Audit sealed evidence in the *original* PR #36 checkout, read-only."""
    origin = ["git", "-C", str(ROOT)]
    subprocess.run([*origin, "rev-parse", "--is-inside-work-tree"],
                   check=True, capture_output=True)
    known = subprocess.run([*origin, "cat-file", "-e", f"{ORIGINAL_PR36_SHA}^{{commit}}"],
                           capture_output=True)
    if known.returncode != 0:
        subprocess.run([*origin, "fetch", "--no-tags", "origin", "+refs/pull/36/head:refs/remotes/origin/lab8-original-pr36"],
                       check=True)
    subprocess.run([*origin, "cat-file", "-e", f"{ORIGINAL_PR36_SHA}^{{commit}}"], check=True)
    with tempfile.TemporaryDirectory(prefix="moons-lab8-historical-") as tmp:
        checkout = Path(tmp) / "original-pr36"
        subprocess.run([*origin, "worktree", "add", "--detach", str(checkout), ORIGINAL_PR36_SHA], check=True)
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(checkout / "src")
            subprocess.run([sys.executable, str(checkout / "scripts/lab8_verify_full.py"),
                            "--report-dir", str(checkout / "reports/LAB8")],
                           cwd=checkout, env=env, check=True)
        finally:
            subprocess.run([*origin, "worktree", "remove", "--force", str(checkout)], check=True)
    print("[OK] Историческая ЛР8: проверка оригинального PR #36 прошла")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce LAB1-7 via isolated DEMO")
    parser.add_argument("--lab8", choices=("skip", "quick", "full"), default="skip",
                        help="LAB8: skip (default), short smoke or >=55min real full run")
    parser.add_argument("--audit-original-lab8", action="store_true",
                        help="read-only audit of original PR #36 sealed evidence")
    args = parser.parse_args(argv)
    if args.audit_original_lab8:
        verify_historical_lab8()
        return

    for number in range(1, 8):
        print(f"\n=== LAB{number} ===", flush=True)
        getattr(menu, f"run_lab{number}")()

    if args.lab8 != "skip":
        print(f"\n=== LAB8 ({args.lab8}) ===", flush=True)
        menu.run_lab8(quick=args.lab8 == "quick")
    print("[OK] Запрошенные лабораторные воспроизведены; официальные отчёты сохранены")


if __name__ == "__main__":
    main()
