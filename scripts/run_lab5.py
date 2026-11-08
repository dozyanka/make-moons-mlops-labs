"""LAB5: deterministic synthetic make_moons generator and verified evidence package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.config import Lab5Config, load_config
from moons_lab.drift import ks_stat, psi
from moons_lab.generator import (
    GeneratorSettings,
    dataframe_bytes,
    dataframe_hash,
    generate_dataframe,
    write_dataframe_csv,
)

FEATURES = ("x1", "x2")
SIMILAR_LIMIT = 0.10
RANDOM_PSI_FLOOR = 0.20
MIN_SIMILAR_FRACTION = 0.80
MIN_RANDOM_FRACTION = 0.50


def settings_from_config(cfg: Lab5Config) -> dict[str, GeneratorSettings]:
    base = dict(n_samples=cfg.n_samples, seed=cfg.seed)
    return {
        "similar": GeneratorSettings(mode="similar", noise=cfg.similar_noise, **base),
        "random": GeneratorSettings(mode="random", random_range=cfg.random_range, **base),
        "drifted": GeneratorSettings(
            mode="similar", noise=cfg.similar_noise,
            shift_x=cfg.drift_shift_x,
            shift_y=cfg.drift_shift_y,
            rotation_deg=cfg.drift_rotation_deg,
            label_flip_rate=cfg.drift_label_flip_rate,
            class_1_ratio=0.70,
            seasonality_amplitude=0.15,
            **base,
        ),
    }


def read_reference(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f"The LAB1 reference dataset is missing: {path}")
    file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = ROOT / "reports" / "LAB1" / "hash_manifest.json"
    expected = json.loads(baseline.read_text(encoding="utf-8"))["raw"]["sha256"]
    if file_hash != expected:
        raise RuntimeError("LAB1 reference CSV changed: refusing to change LAB4 input")
    data = pd.read_csv(path)
    if list(data.columns) != ["x1", "x2", "y"]:
        raise ValueError("LAB1 input must contain x1,x2,y in this order")
    if data.empty or not np.isfinite(data[["x1", "x2"]].to_numpy(dtype=float)).all():
        raise ValueError("LAB1 input has invalid numerical features")
    if not data["y"].isin([0, 1]).all():
        raise ValueError("LAB1 input must have binary classes")
    return data, file_hash


def class_psi(ref: pd.Series, generated: pd.Series) -> float:
    """PSI with two *fixed* categories, unlike continuous-feature quantile bins."""
    a = np.maximum(np.bincount(ref.to_numpy(dtype=int), minlength=2) / len(ref), 1e-9)
    b = np.maximum(
        np.bincount(generated.to_numpy(dtype=int), minlength=2) / len(generated), 1e-9
    )
    return float(np.sum((b - a) * np.log(b / a)))


def compare(reference: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode, frame in frames.items():
        for feature in FEATURES:
            r = reference[feature]
            g = frame[feature]
            rows.append({
                "mode": mode,
                "feature": feature,
                "reference_mean": float(r.mean()),
                "generated_mean": float(g.mean()),
                "reference_std": float(r.std()),
                "generated_std": float(g.std()),
                "reference_class_1_share": float(reference["y"].mean()),
                "generated_class_1_share": float(frame["y"].mean()),
                "psi": psi(r.to_numpy(), g.to_numpy()),
                "ks": ks_stat(r.to_numpy(), g.to_numpy()),
            })
        rows.append({
            "mode": mode,
            "feature": "class_1_share",
            "reference_mean": float(reference["y"].mean()),
            "generated_mean": float(frame["y"].mean()),
            "reference_std": float(reference["y"].std()),
            "generated_std": float(frame["y"].std()),
            "reference_class_1_share": float(reference["y"].mean()),
            "generated_class_1_share": float(frame["y"].mean()),
            "psi": class_psi(reference["y"], frame["y"]),
            "ks": ks_stat(reference["y"].to_numpy(), frame["y"].to_numpy()),
        })
    return pd.DataFrame.from_records(rows)


def thresholds(metrics: pd.DataFrame) -> dict[str, object]:
    if not {"mode", "feature", "psi", "ks"}.issubset(metrics.columns):
        raise ValueError("LAB5 metrics lack mandatory mode/feature/psi/ks columns")
    similar = metrics[(metrics["mode"] == "similar") & metrics["feature"].isin(FEATURES)]
    random = metrics[(metrics["mode"] == "random") & metrics["feature"].isin(FEATURES)]
    if len(similar) != len(FEATURES) or len(random) != len(FEATURES):
        raise ValueError("Need both numeric features in both generation modes")
    similar_pass = (similar["psi"] <= SIMILAR_LIMIT) | (similar["ks"] <= SIMILAR_LIMIT)
    random_pass = random["psi"] >= RANDOM_PSI_FLOOR
    score = {
        "similar_fraction": float(similar_pass.mean()),
        "random_fraction": float(random_pass.mean()),
        "similar_pass": bool(similar_pass.mean() >= MIN_SIMILAR_FRACTION),
        "random_pass": bool(random_pass.mean() >= MIN_RANDOM_FRACTION),
    }
    return score


def run_pytest(output: Path) -> int:
    workspace = ROOT / "demo_output" / "pytest_runtime" / f"LAB5_{uuid.uuid4().hex}"
    workspace.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "tests/test_generator.py", "tests/test_lab5_workflow.py",
        "--cov=moons_lab.generator", "--cov-report=term-missing",
        f"--basetemp={workspace}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command, cwd=ROOT, env=env, check=False, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = "> " + " ".join(command) + "\n\n" + completed.stdout
    (output / "generator_tests.log").write_text(log, encoding="utf-8", newline="\n")
    print(completed.stdout, flush=True)
    if completed.returncode:
        raise RuntimeError("LAB5 pytest failed; see generator_tests.log")
    lines = [line.strip() for line in completed.stdout.splitlines()]
    matches = [line for line in lines if "moons_lab/generator.py" in line.replace("\\", "/")]
    if len(matches) != 1:
        raise RuntimeError("Cannot find generator.py coverage row")
    percentages = re.findall(r"(\d{1,3})%", matches[0])
    if not percentages:
        raise RuntimeError("Cannot parse generator.py coverage")
    coverage = int(percentages[-1])
    if coverage < 80:
        raise RuntimeError(f"LAB5 generator coverage {coverage}% is below 80%")
    return coverage


def make_report(
    metrics: pd.DataFrame, scores: dict[str, object],
    settings: dict[str, GeneratorSettings], hashes: dict[str, str],
    reference_hash: str, coverage: int,
) -> str:
    lines = [
        "# Лабораторная работа № 5. Управляемый генератор синтетических данных",
        "",
        "**Выполнил:** Владимир Шаталов, ЕТ-122",
        "**Вариант:** 09 — `sklearn.datasets.make_moons` (нелинейная классификация), профиль P1.",
        "",
        "## 1. Цель и исходные данные",
        "",
        "Создать генератор с управляемыми режимами и дрейфом, получить воспроизводимые "
        "выборки и проверить статистическое сходство с исходными данными ЛР1.",
        "Оригинальный набор `data/raw/moons.csv` **только читается**, SHA-256: "
        f"`{reference_hash}`.",
        "",
        "## 2. Схема настроек и алгоритм",
        "",
        "`GeneratorSettings` (Pydantic) проверяет `mode` (`similar`/`random`), `n_samples`, "
        "`seed`, `noise`, `shift_x`, `shift_y`, `rotation_deg`, `scale`, "
        "`class_1_ratio`, `label_flip_rate`, `random_range`, "
        "`seasonality_amplitude`, `seasonality_period`, `output_columns`.",
        "Выходная схема строго фиксирована: **x1, x2 (float64), y (0 или 1)**; "
        "некорректные режимы, лишние поля, NaN и выходящие за диапазон значения отклоняются.",
        "",
        "- `similar`: исходная нелинейная геометрия двух полумесяцев (`make_moons`), "
        "с шумом 0.22 и управляемым бутстрэпом по классам.",
        "- `random`: независимый равномерный шум признаков в заданных границах, "
        "бинарная метка формируется отдельно.",
        "- `drifted`: исходная геометрия с поворотом и сдвигами, "
        "изменённой долей класса 1, инверсией части меток и периодической добавкой к x2.",
        "",
        "Периодическая добавка является *искусственным контролируемым возмущением "
        "табличных данных*, а не моделью реального временного ряда.",
        "Конкретные значения всех ручек сохранены в `generator_config.json`.",
        "",
        "## 3. Сравнение с исходным распределением",
        "",
        "PSI рассчитан по 10 квантильным корзинам **эталона**; KS — "
        "двухвыборочная статистика Колмогорова–Смирнова. Для бинарного `y` "
        "PSI вычисляется по фиксированным классам 0/1.",
        "",
        "| Режим | Признак | Среднее эталона | Среднее генератора | PSI | KS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics.itertuples():
        lines.append(
            f"| {row.mode} | {row.feature} | {row.reference_mean:.4f} | "
            f"{row.generated_mean:.4f} | {row.psi:.4f} | {row.ks:.4f} |"
        )
    lines.extend([
        "",
        "**Критерий похожести:** для минимум 80% числовых признаков "
        "PSI ≤ 0.10 **или** KS ≤ 0.10. Поскольку признаков два, должны пройти оба.",
        f"**Результат:** {scores['similar_fraction']:.0%} (" +
        ("PASS" if scores["similar_pass"] else "FAIL") + ").",
        "",
        "**Критерий случайного режима:** минимум у 50% признаков PSI ≥ 0.20 "
        "(достаточно одного из двух).",
        f"**Результат:** {scores['random_fraction']:.0%} (" +
        ("PASS" if scores["random_pass"] else "FAIL") + ").",
        "",
        "## 4. Воспроизводимость и негативные контроли",
        "",
        "Генерация с одним seed выполняется дважды; сравниваются **байты CSV**, "
        "а не только числовые значения. Сохранение фиксирует UTF-8, LF и десять "
        "знаков после запятой. Смена seed должна изменять SHA-256.",
        "",
        "| Режим | SHA-256 CSV | Повтор |",
        "|---|---|---|",
    ])
    for name in ("similar", "random", "drifted"):
        lines.append(f"| {name} | `{hashes[name]}` | побайтово совпал |")
    lines.extend([
        "",
        "Тесты отдельно проверяют поворот, сдвиг, масштаб, изменение долей классов, "
        "переворот меток, шум, периодическую добавку, два режима, "
        "некорректные конфигурации и переключение настроек без перезапуска.",
        "",
        "## 5. Тесты и покрытие",
        "",
        "| Проверка | Результат |",
        "|---|---:|",
        "| pytest: схема, режимы, дрейф, seed, API и метрики | PASS |",
        f"| Покрытие `src/moons_lab/generator.py` | **{coverage}%** |",
        "| Требуемый минимум покрытия | 80% |",
        "",
        "Полный лог выполнения и пропущенные строки: `generator_tests.log`. "
        "Режимы и числовые сравнения: `synthetic_metrics.csv`. "
        "Все конфигурации: `generator_config.json`. "
        "Свидетельство повтора: `reproducibility.json`. "
        "Три CSV находятся в `data/synthetic/`.",
        "",
        "## 6. Выводы",
        "",
        "1. Реализован типизированный генератор для варианта 09 и двух режимов выборки.",
        "2. Режим `similar` проходит установленный критерий статистической близости.",
        "3. Режим `random` создаёт заметно отличающееся распределение.",
        "4. Все объявленные ручки дрейфа проверены тестами; результаты по seed воспроизводимы побайтово.",
        f"5. Покрытие генератора составляет {coverage}% и удовлетворяет требованию 80%; "
        "оригиналы ЛР1–4 не изменялись.",
        "",
    ])
    return "\n".join(lines)


def execute(report_dir: Path, data_dir: Path, *, skip_tests: bool = False) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(5)
    settings = settings_from_config(cfg)
    reference, source_sha = read_reference(ROOT / "data/raw/moons.csv")
    frames = {name: generate_dataframe(setting) for name, setting in settings.items()}
    hashes: dict[str, str] = {}
    for name, frame in frames.items():
        target = data_dir / f"{name}.csv"
        hashes[name] = write_dataframe_csv(frame, target)
        repeat = generate_dataframe(settings[name])
        if dataframe_bytes(repeat) != target.read_bytes():
            raise RuntimeError(f"Reproducibility failed in {name}: CSV differs by bytes")
        changed_seed = settings[name].model_copy(update={"seed": settings[name].seed + 1})
        if dataframe_hash(generate_dataframe(changed_seed)) == hashes[name]:
            raise RuntimeError(f"Negative control failed: seed does not affect {name}")
    metrics = compare(reference, frames)
    scores = thresholds(metrics)
    if not (scores["similar_pass"] and scores["random_pass"]):
        raise RuntimeError(f"LAB5 distribution criteria failed: {scores}")
    metrics.to_csv(
        report_dir / "synthetic_metrics.csv", index=False,
        float_format="%.8f", lineterminator="\n",
    )
    reproducibility = {
        "byte_identical_on_repeat": True,
        "different_seed_produces_different_sha256": True,
        "sha256": hashes,
        "thresholds": scores,
    }
    (report_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    config_payload = {
        "variant": 9,
        "source": "sklearn.datasets.make_moons",
        "reference_sha256": source_sha,
        "presets": {k: v.model_dump(mode="json") for k, v in settings.items()},
        "outputs_sha256": hashes,
    }
    (report_dir / "generator_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    coverage = run_pytest(report_dir) if not skip_tests else -1
    if coverage >= 80:
        (report_dir / "lab5_report.md").write_text(
            make_report(metrics, scores, settings, hashes, source_sha, coverage),
            encoding="utf-8", newline="\n",
        )
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("LAB5 distribution thresholds:", scores)
    print("LAB5 data SHA-256:", hashes)
    if coverage >= 80:
        print(f"LAB5 PASSED; generator coverage {coverage}%; report: {report_dir / 'lab5_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default="reports/LAB5")
    parser.add_argument("--data-dir", default="data/synthetic")
    # Used by isolated integration tests; regular LAB5 runs always execute pytest.
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    data_dir = Path(args.data_dir)
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if report_dir.resolve().is_relative_to((ROOT / "reports/LAB4").resolve()):
        raise ValueError("LAB5 output must never overwrite LAB4 report files")
    if data_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()) or data_dir.resolve().is_relative_to((ROOT / "data/processed").resolve()):
        raise ValueError("LAB5 output must never overwrite LAB1-4 input CSV files")
    execute(report_dir, data_dir, skip_tests=args.skip_tests)


if __name__ == "__main__":
    main()
