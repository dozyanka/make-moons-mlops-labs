from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key)\s*[:=]\s*['\"][^'\"]{4,}['\"]"),
    re.compile(r"-----BEGIN (RSA|OPENSSH|EC) PRIVATE KEY-----"),
]
SKIP_DIRS = {".git", ".venv", "mlruns", "runs_fallback", "__pycache__", ".pytest_cache"}
MAX_FILE_BYTES = 5 * 1024 * 1024


def display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)



def files_under_root() -> list[Path]:
    result: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            result.append(path)
    return result


def check_style(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.suffix != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
            ast.parse(text, filename=str(path))
        except (UnicodeDecodeError, SyntaxError) as exc:
            errors.append(f"syntax: {display_path(path)}: {exc}")
            continue
        for no, line in enumerate(text.splitlines(), 1):
            if line.rstrip() != line:
                errors.append(f"trailing-space: {display_path(path)}:{no}")
            if "\t" in line:
                errors.append(f"tab: {display_path(path)}:{no}")
    return errors


def check_secrets(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".joblib", ".pt", ".npz", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret-like token: {display_path(path)}")
                break
    return errors


def check_large_files(paths: list[Path]) -> list[str]:
    errors = []
    allowed = {"data/raw/moons.csv", "data/processed/train.csv", "data/processed/validation.csv", "data/processed/test.csv"}
    for path in paths:
        rel = display_path(path)
        if path.stat().st_size > MAX_FILE_BYTES and rel not in allowed:
            errors.append(f"large file >5 MiB: {rel} ({path.stat().st_size} bytes)")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", action="store_true")
    parser.add_argument("--secrets", action="store_true")
    parser.add_argument("--large-files", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--path", action="append", default=[])
    args = parser.parse_args()
    paths = [Path(p) for p in args.path] if args.path else files_under_root()
    errors: list[str] = []
    if args.all or args.style:
        errors += check_style(paths)
    if args.all or args.secrets:
        errors += check_secrets(paths)
    if args.all or args.large_files:
        errors += check_large_files(paths)
    if errors:
        print("BLOCKED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
