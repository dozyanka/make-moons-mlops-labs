"""Real fresh-venv installation probe (no --no-deps shortcut in full mode)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=ROOT / "demo_output/LAB8/install.log")
    p.add_argument("--quick", action="store_true", help="Smoke only, not proof of full dependency resolution")
    args = p.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    transcript = []

    def run(cmd, cwd=None):
        transcript.append("$ " + " ".join(str(x) for x in cmd))
        print(transcript[-1], flush=True)
        res = subprocess.run(cmd, cwd=cwd or ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, errors="replace")
        transcript.append(res.stdout)
        transcript.append("exit=" + str(res.returncode))
        if res.returncode:
            raise RuntimeError("installation command failed: " + transcript[-2][-1200:])

    try:
        with tempfile.TemporaryDirectory(prefix="moons_lab8_clean_") as tmp:
            folder = Path(tmp) / "venv"
            venv.EnvBuilder(with_pip=True).create(folder)
            py = folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            if args.quick:
                run([str(py), "-m", "pip", "install", "--no-deps", "-e", str(ROOT)])
                run([str(py), "-c", "import moons_lab; print(moons_lab.__version__)"])
            else:
                run([str(py), "-m", "pip", "install", "-e", f"{ROOT}[all,report]"])
                run([str(py), "-c", "import moons_lab, torch, fastapi, evidently, mlflow, sklearn, pandas; print('FRESH_ENV_OK')"])
                # Ensure every advertised entry point exists in isolated venv.
                bin_dir = py.parent
                names = ("train_ml", "train_dl", "train_validator", "generate", "serve", "evaluate", "monitor")
                for name in names:
                    exe = bin_dir / (name + (".exe" if os.name == "nt" else ""))
                    if not exe.is_file():
                        raise RuntimeError(f"missing CLI entry point: {exe}")
                transcript.append("seven CLI entrypoints present")
        transcript.append("RESULT: PASS" if not args.quick else "RESULT: SMOKE_ONLY")
    except Exception as exc:
        transcript.append("RESULT: FAIL: " + repr(exc))
        args.log.write_text("\n".join(transcript), encoding="utf-8")
        raise
    args.log.write_text("\n".join(transcript), encoding="utf-8")
    print(transcript[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
