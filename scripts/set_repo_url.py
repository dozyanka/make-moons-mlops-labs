from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if len(sys.argv) != 2 or not re.match(r"^https://github\.com/[^/]+/[^/]+/?$", sys.argv[1]):
        raise SystemExit("Usage: python scripts/set_repo_url.py https://github.com/USER/REPO")
    url = sys.argv[1].rstrip("/")
    changed = 0
    for path in sorted((ROOT / "reports").rglob("lab*_report.md")):
        text = path.read_text(encoding="utf-8")
        new = re.sub(r"\*\*Репозиторий:\*\*.*", f"**Репозиторий:** {url}", text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"Updated repository URL in {changed} reports: {url}")


if __name__ == "__main__":
    main()
