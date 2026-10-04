from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "src"),
)

from moons_lab.config import load_config
from moons_lab.data import prepare_data
from moons_lab.eda import (
    compute_eda,
    plot_class_balance,
    plot_confusion,
    plot_correlation,
    plot_feature_distributions,
    plot_problem,
)
from moons_lab.utils import sha256_file


def _run_negative_controls() -> None:
    report = (
        ROOT
        / "reports/LAB1/precommit_block.log"
    )

    lines: list[str] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)

        secret = tmp / "secret.txt"
        secret_value = (
            "AK"
            + "IA"
            + "ABCDEFGHIJKLMNOP"
        )
        key_name = "api" + "_key"

        secret.write_text(
            f'{key_name}: "{secret_value}"\n',
            encoding="utf-8",
        )

        large = tmp / "large.bin"
        large.write_bytes(
            b"0"
            * (
                5 * 1024 * 1024
                + 100
            )
        )

        bad_py = tmp / "bad.py"
        bad_py.write_text(
            "def broken(:\n"
            "    pass\n",
            encoding="utf-8",
        )

        cases = [
            (
                "STYLE_INVALID_PYTHON",
                [
                    "--style",
                    "--path",
                    str(bad_py),
                ],
            ),
            (
                "SECRET",
                [
                    "--secrets",
                    "--path",
                    str(secret),
                ],
            ),
            (
                "LARGE_FILE",
                [
                    "--large-files",
                    "--path",
                    str(large),
                ],
            ),
        ]

        for name, args in cases:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(
                        ROOT
                        / "scripts/precommit_checks.py"
                    ),
                    *args,
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            lines.append(
                f"[{name}] "
                f"exit={proc.returncode}"
            )
            lines.append(
                proc.stdout.strip()
            )

    report.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _hash_negative_control() -> None:
    source = (
        ROOT
        / "data/raw/moons.csv"
    )

    expected = sha256_file(source)

    with tempfile.TemporaryDirectory() as tmp_dir:
        altered = (
            Path(tmp_dir)
            / "moons.csv"
        )

        altered.write_bytes(
            source.read_bytes()
        )

        with altered.open("ab") as file:
            file.write(
                b"\n0.0,0.0,0"
            )

        actual = sha256_file(altered)

    payload = {
        "expected_sha256": expected,
        "altered_sha256": actual,
        "match": expected == actual,
        "verdict": (
            "tampering_detected"
            if expected != actual
            else "ERROR"
        ),
    }

    output = (
        ROOT
        / "reports/LAB1/hash_negative_test.log"
    )

    output.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    cfg = load_config(1)

    manifest = prepare_data(cfg)

    raw = (
        ROOT
        / "data/raw/moons.csv"
    )

    figures = (
        ROOT
        / "reports/LAB1/figures"
    )

    figures.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats, summary = compute_eda(raw)

    stats.to_csv(
        ROOT
        / "reports/LAB1/eda_stats.csv",
        index=False,
    )

    plot_problem(
        raw,
        figures
        / "nonlinearity_scatter.png",
        cfg.seed,
    )

    plot_confusion(
        raw,
        figures
        / "linear_confusion.png",
    )

    plot_class_balance(
        raw,
        figures
        / "class_balance.png",
    )

    plot_feature_distributions(
        raw,
        figures
        / "feature_distributions.png",
    )

    plot_correlation(
        raw,
        figures
        / "correlation_matrix.png",
    )

    _run_negative_controls()
    _hash_negative_control()

    print(
        json.dumps(
            {
                "manifest": manifest,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
