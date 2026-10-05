from __future__ import annotations

import importlib
import platform
import sys

MODULES = [
    ("numpy", "NumPy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("torch", "PyTorch"),
    ("mlflow", "MLflow"),
    ("pytest", "pytest"),
    ("docx", "python-docx"),
    ("pre_commit", "pre-commit"),
]

print("=" * 72)
print("ПРОВЕРКА ОКРУЖЕНИЯ")
print("=" * 72)
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform())

failed: list[str] = []

for module_name, display_name in MODULES:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "OK")
        print(f"[OK] {display_name}: {version}")
    except Exception as exc:
        failed.append(display_name)
        print(f"[FAIL] {display_name}: {exc}")

try:
    import torch

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
    else:
        print("GPU: не требуется для основного запуска ЛР1-3; ЛР3 работает на CPU.")
except Exception:
    pass

if failed:
    print()
    print("Не удалось импортировать:", ", ".join(failed))
    raise SystemExit(1)

print()
print("Окружение готово.")
