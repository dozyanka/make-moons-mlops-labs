from __future__ import annotations

import subprocess
import sys
from pathlib import Path


from common import RESULTS

HERE = Path(__file__).resolve().parent


def run(*args):
    command = [sys.executable, *args]
    print()
    print(">", " ".join(map(str, command)))
    subprocess.run(command, cwd=HERE.parent, check=True)


def main():
    run("defense_demo/lab1_demo.py")
    run("defense_demo/lab2_demo.py", "--mode", "full")
    run("defense_demo/lab2_demo.py", "--mode", "chunks")
    run("defense_demo/lab3_demo.py", "--mode", "full")
    run("defense_demo/lab3_demo.py", "--mode", "streaming")
    run("defense_demo/stream_memory_demo.py")
    run("defense_demo/lab4_demo.py")

    print()
    print("=" * 72)
    print("ALL LAB1–LAB4 LIVE DEFENSE DEMOS PASSED")
    print("=" * 72)
    print("Results:", RESULTS)


if __name__ == "__main__":
    main()
