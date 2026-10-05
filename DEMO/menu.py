from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DEMO_OUTPUT = ROOT / "demo_output"


def run(args: list[str], *, log_name: str | None = None) -> None:
    cmd = [str(PYTHON), *args]
    print()
    print(">", " ".join(cmd))

    if log_name is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return

    log_dir = DEMO_OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{stamp}.log"
    print("Лог:", log_path)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def run_exe(args: list[str], *, log_name: str | None = None) -> None:
    print()
    print(">", " ".join(args))

    if log_name is None:
        subprocess.run(args, cwd=ROOT, check=True)
        return

    log_dir = DEMO_OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{stamp}.log"
    print("Лог:", log_path)

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            args,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, args)


@contextlib.contextmanager
def reproduced_report(lab: str):
    official = ROOT / "reports" / lab
    destination = DEMO_OUTPUT / lab
    temp_root = Path(tempfile.mkdtemp(prefix=f"make-moons-{lab.lower()}-"))
    backup = temp_root / lab

    had_official = official.exists()

    if had_official:
        shutil.copytree(official, backup)
    else:
        official.mkdir(parents=True, exist_ok=True)

    if lab == "LAB3":
        (official / "figures").mkdir(parents=True, exist_ok=True)

    try:
        yield

        if not official.exists():
            raise RuntimeError(f"{lab}: reports/{lab} was not created")

        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(official, destination)

        print()
        print("[OK] Новый прогон сохранён отдельно:")
        print("    ", destination)
    finally:
        if official.exists():
            shutil.rmtree(official)

        if had_official and backup.exists():
            shutil.copytree(backup, official)

        shutil.rmtree(temp_root, ignore_errors=True)


def prepare_data() -> None:
    run(["scripts/prepare_data.py"])


def run_lab1() -> None:
    print("\n" + "=" * 72)
    print("ЛР1 — ВОСПРОИЗВЕДЕНИЕ")
    print("=" * 72)

    with reproduced_report("LAB1"):
        prepare_data()
        run(["scripts/run_lab1.py"], log_name="LAB1")

        if (ROOT / ".git").exists() and (ROOT / "scripts/finalize_lab1.py").exists():
            run(["scripts/finalize_lab1.py"], log_name="LAB1_FINALIZE")

        run(
            [
                "-m",
                "pytest",
                "tests/test_data.py",
                "tests/test_eda.py",
                "-q",
            ],
            log_name="LAB1_TESTS",
        )

        precommit = ROOT / ".venv" / "Scripts" / "pre-commit.exe"
        if (ROOT / ".git").exists() and precommit.exists():
            run_exe([str(precommit), "run", "--all-files"], log_name="LAB1_PRECOMMIT")
        else:
            run(["scripts/precommit_checks.py", "--all"], log_name="LAB1_CHECKS")

    print("Оригинал: reports/LAB1")
    print("Новый прогон: demo_output/LAB1")


def run_lab2() -> None:
    print("\n" + "=" * 72)
    print("ЛР2 — ВОСПРОИЗВЕДЕНИЕ")
    print("=" * 72)

    with reproduced_report("LAB2"):
        prepare_data()
        run(["scripts/run_lab2.py"], log_name="LAB2")

        if (ROOT / ".git").exists() and (ROOT / "scripts/finalize_lab2.py").exists():
            run(["scripts/finalize_lab2.py"], log_name="LAB2_FINALIZE")

        run(
            [
                "-m",
                "pytest",
                "tests/test_data.py",
                "tests/test_metrics.py",
                "tests/test_ml.py",
                "tests/test_tracking_utils.py",
                "-q",
            ],
            log_name="LAB2_TESTS",
        )

    print("Оригинал: reports/LAB2")
    print("Новый прогон: demo_output/LAB2")


def run_lab3() -> None:
    print("\n" + "=" * 72)
    print("ЛР3 — ВОСПРОИЗВЕДЕНИЕ")
    print("=" * 72)

    with reproduced_report("LAB3"):
        prepare_data()
        run(["scripts/run_lab3.py"], log_name="LAB3")

        if (ROOT / ".git").exists() and (ROOT / "scripts/finalize_lab3.py").exists():
            run(["scripts/finalize_lab3.py"], log_name="LAB3_FINALIZE")

        run(
            [
                "-m",
                "pytest",
                "tests/test_data.py",
                "tests/test_metrics.py",
                "tests/test_dl.py",
                "-q",
            ],
            log_name="LAB3_TESTS",
        )

    print("Оригинал: reports/LAB3")
    print("Новый прогон: demo_output/LAB3")
    print("Если CUDA недоступна, основной запуск ЛР3 выполняется на CPU.")
    print("Сохранённая CPU/GPU parity: reports/LAB3/device_parity.csv")


def quick_demo() -> None:
    print("\n" + "=" * 72)
    print("БЫСТРАЯ ДЕМОНСТРАЦИЯ ГОТОВЫХ МОДЕЛЕЙ")
    print("=" * 72)
    prepare_data()
    run(["defense_demo/run_all.py"], log_name="QUICK_DEMO")


def all_tests() -> None:
    print("\n" + "=" * 72)
    print("ТЕСТЫ ЛР1–ЛР3")
    print("=" * 72)

    run(
        [
            "-m",
            "pytest",
            "tests/test_data.py",
            "tests/test_eda.py",
            "tests/test_metrics.py",
            "tests/test_ml.py",
            "tests/test_tracking_utils.py",
            "tests/test_dl.py",
            "-q",
        ],
        log_name="TESTS",
    )


def start_mlflow() -> None:
    mlflow_exe = ROOT / ".venv" / "Scripts" / "mlflow.exe"
    if not mlflow_exe.exists():
        raise FileNotFoundError(mlflow_exe)

    subprocess.Popen(
        [
            str(mlflow_exe),
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--port",
            "5000",
        ],
        cwd=ROOT,
    )
    webbrowser.open("http://127.0.0.1:5000")
    print("MLflow UI: http://127.0.0.1:5000")


def open_results() -> None:
    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        os.startfile(ROOT / "reports")
        os.startfile(DEMO_OUTPUT)
    else:
        print("reports:", ROOT / "reports")
        print("demo_output:", DEMO_OUTPUT)


def smoke_test() -> None:
    required = [
        ROOT / "pyproject.toml",
        ROOT / "scripts" / "prepare_data.py",
        ROOT / "scripts" / "run_lab1.py",
        ROOT / "scripts" / "run_lab2.py",
        ROOT / "scripts" / "run_lab3.py",
        ROOT / "defense_demo" / "run_all.py",
        ROOT / "reports" / "LAB1",
        ROOT / "reports" / "LAB2",
        ROOT / "reports" / "LAB3",
    ]

    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing:")
        for path in missing:
            print(" -", path)
        raise SystemExit(1)

    run(["DEMO/verify_environment.py"])
    print()
    print("[OK] DEMO smoke test passed.")


def show_menu() -> None:
    print("\n" + "=" * 72)
    print(" MAKE-MOONS MLOPS LABS — ЗАПУСК ДЛЯ ПРОВЕРКИ")
    print("=" * 72)
    print()
    print(" 1. Проверить окружение")
    print(" 2. Запустить ЛР1")
    print(" 3. Запустить ЛР2")
    print(" 4. Запустить ЛР3")
    print(" 5. Запустить ЛР1–ЛР3 с нуля")
    print(" 6. Быстрая демонстрация сохранённых моделей")
    print(" 7. Запустить тесты ЛР1–ЛР3")
    print(" 8. Запустить MLflow UI")
    print(" 9. Открыть оригинальные и новые результаты")
    print()
    print(" 0. Выход")
    print()


def interactive() -> None:
    while True:
        show_menu()
        choice = input("Выберите пункт: ").strip()

        try:
            if choice == "1":
                run(["DEMO/verify_environment.py"])
            elif choice == "2":
                run_lab1()
            elif choice == "3":
                run_lab2()
            elif choice == "4":
                run_lab3()
            elif choice == "5":
                run_lab1()
                run_lab2()
                run_lab3()
                print("\nЛР1: PASS\nЛР2: PASS\nЛР3: PASS")
            elif choice == "6":
                quick_demo()
            elif choice == "7":
                all_tests()
            elif choice == "8":
                start_mlflow()
            elif choice == "9":
                open_results()
            elif choice == "0":
                return
            else:
                print("Неизвестный пункт.")
        except Exception as exc:
            print()
            print("ОШИБКА:", exc)

        if choice != "0":
            input("\nНажмите Enter, чтобы вернуться в меню...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
    else:
        interactive()


if __name__ == "__main__":
    main()