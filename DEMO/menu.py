from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
DEMO_OUTPUT = ROOT / "demo_output"

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def run(args: list[str], *, log_name: str | None = None) -> None:
    cmd = [str(PYTHON), *args]
    print()
    print(">", " ".join(cmd), flush=True)

    if log_name is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return

    log_dir = DEMO_OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{stamp}.log"
    print("Лог:", log_path, flush=True)

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
            print(line, end="", flush=True)
            log.write(line)

        code = process.wait()

    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def run_exe(args: list[str], *, log_name: str | None = None) -> None:
    print()
    print(">", " ".join(args), flush=True)

    if log_name is None:
        subprocess.run(args, cwd=ROOT, check=True)
        return

    log_dir = DEMO_OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{log_name}_{stamp}.log"
    print("Лог:", log_path, flush=True)

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
            print(line, end="", flush=True)
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
            run_exe(
                [str(precommit), "run", "--all-files"],
                log_name="LAB1_PRECOMMIT",
            )
        else:
            run(
                ["scripts/precommit_checks.py", "--all"],
                log_name="LAB1_CHECKS",
            )

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


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def start_mlflow(
    port: int = 5000,
    *,
    open_browser: bool = True,
) -> subprocess.Popen[str] | None:
    mlflow_exe = ROOT / ".venv" / "Scripts" / "mlflow.exe"
    if not mlflow_exe.exists():
        raise FileNotFoundError(mlflow_exe)

    if port_is_open(port):
        url = f"http://127.0.0.1:{port}"
        print(f"MLflow UI уже запущен: {url}")
        if open_browser:
            webbrowser.open(url)
        return None

    log_dir = DEMO_OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"MLFLOW_UI_{port}.log"

    command = [
        str(mlflow_exe),
        "server",
        "--backend-store-uri",
        "sqlite:///mlflow.db",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
    ]

    creationflags = 0
    popen_kwargs: dict[str, object] = {}

    if os.name == "nt":
        creationflags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        popen_kwargs["start_new_session"] = True

    log_file = log_path.open("a", encoding="utf-8")

    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            **popen_kwargs,
        )
    finally:
        log_file.close()

    deadline = time.time() + 30.0

    while time.time() < deadline:
        if port_is_open(port):
            url = f"http://127.0.0.1:{port}"
            print(f"MLflow UI: {url}")
            print(f"Лог MLflow: {log_path}")
            if open_browser:
                webbrowser.open(url)
            return process

        if process.poll() is not None:
            tail = ""
            if log_path.exists():
                lines = log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
                tail = "\n".join(lines[-20:])

            raise RuntimeError(
                "MLflow завершился до запуска сервера.\n"
                f"Лог: {log_path}\n{tail}"
            )

        time.sleep(0.5)

    process.terminate()
    raise RuntimeError(
        f"MLflow не открыл порт {port} за 30 секунд. "
        f"Смотрите лог: {log_path}"
    )


def get_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def stop_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        result = subprocess.run(
            [
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            return

    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def mlflow_smoke_test() -> None:
    port = get_free_local_port()
    process: subprocess.Popen[str] | None = None

    try:
        process = start_mlflow(port, open_browser=False)

        if process is None:
            raise RuntimeError("MLflow smoke-test did not start its own process.")

        print(f"[OK] MLflow one-worker smoke test passed on port {port}.")
    finally:
        if process is not None:
            stop_process_tree(process)

    deadline = time.time() + 10.0
    while time.time() < deadline:
        if not port_is_open(port):
            print(f"[OK] MLflow smoke-test process tree stopped; port {port} is free.")
            return
        time.sleep(0.25)

    raise RuntimeError(
        f"MLflow smoke-test left port {port} occupied after cleanup."
    )

def open_results() -> None:
    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        os.startfile(ROOT / "reports")
        os.startfile(DEMO_OUTPUT)
        print("Открыты папки reports и demo_output.")
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
    parser.add_argument("--mlflow-smoke-test", action="store_true")
    args = parser.parse_args()

    if args.mlflow_smoke_test:
        mlflow_smoke_test()
    elif args.smoke_test:
        smoke_test()
    else:
        interactive()


if __name__ == "__main__":
    main()