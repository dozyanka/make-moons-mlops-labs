"""LAB8 reproducible E2E + full live HTTP operations + C1-C5 drills.

--quick is a short engineering smoke, NOT 20-point official evidence.
--full takes at least 55 minutes of wall-clock load stages + installation.
All generated evidence stays under --report-dir; old labs are read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.alerts import AlertManager, AlertRule
from moons_lab.data import load_xy
from moons_lab.dl_pipeline import load_checkpoint
from moons_lab.generator import GeneratorSettings, generate_dataframe
from moons_lab.lab8_config import Lab8FullConfig, load_lab8_config
from moons_lab.lab8_live import (fault_suite, http, plot_usage, setup_registry,
                                  stage_summary, start_service, stop_service, utc, wait_healthy,
                                  write_csv)
from moons_lab.lab8_observability import StreamGuard, hash_manifest
from moons_lab.validator import load_validator
from moons_lab.tracking import tracked_run


def positive(model, x):
    raw = np.asarray(model.predict_proba(x))
    return raw[:, 1] if raw.ndim == 2 else raw.ravel()


def e2e(report: Path, seed: int) -> pd.DataFrame:
    ml = joblib.load(ROOT / "reports/LAB2/ml_model.joblib")
    dl = load_checkpoint(ROOT / "reports/LAB3/dl_model.pt")
    validator = load_validator(ROOT / "reports/LAB4/validator_params.npz")
    records = []
    for name in ("real", "synthetic"):
        for repeat in (1, 2):
            if name == "real":
                x, y = load_xy(ROOT / "data/processed/test.csv")
            else:
                df = generate_dataframe(GeneratorSettings(mode="similar", n_samples=3600, seed=seed, noise=.22))
                x = df[["x1", "x2"]].to_numpy()
                y = df["y"].to_numpy()
            t0 = time.perf_counter()
            mlp = positive(ml, x)
            dlp = dl.predict_proba(x)
            accepted = validator.accept(x, mlp)
            from sklearn.metrics import accuracy_score, roc_auc_score
            records.append({"source": name, "repeat": repeat, "rows": len(x),
                            "ml_accuracy": float(accuracy_score(y, mlp >= .5)),
                            "dl_accuracy": float(accuracy_score(y, dlp >= .5)),
                            "ml_auc": float(roc_auc_score(y, mlp)),
                            "dl_auc": float(roc_auc_score(y, dlp)),
                            "validator_accept_rate": float(np.mean(accepted)),
                            "elapsed_s": time.perf_counter() - t0})
    out = pd.DataFrame(records)
    for name in ("real", "synthetic"):
        pair = out[out.source == name]
        for metric in ("ml_accuracy", "dl_accuracy", "ml_auc", "dl_auc", "validator_accept_rate"):
            if abs(float(pair.iloc[0][metric]) - float(pair.iloc[1][metric])) > 1e-12:
                raise AssertionError(f"E2E nonreproducible: {name}.{metric}")
    out.to_csv(report / "end_to_end.csv", index=False)
    return out


def measure_stage(name: str, base: str, monitor: str, cfg: Lab8FullConfig,
                  frame: pd.DataFrame, stream: StreamGuard, *, seconds: int, requests: int,
                  n_jobs: int, batch_size: int, report: Path, proc, quick: bool,
                  baseline: float, model_version: str, switch_to: str | None = None,
                  switched_reference: np.ndarray | None = None,
                  validated: bool = False) -> tuple[dict, list[dict]]:
    from moons_lab.lab8_live import write_csv
    starting = time.monotonic()
    timestamps = []
    resource_rows = []
    raw = frame[["x1", "x2"]].to_numpy(dtype=float)
    rowidx = 0
    switch_status = None
    stop = threading.Event()
    measured = psutil.Process(proc.pid)

    def sample():
        measured.cpu_percent(None)
        psutil.cpu_percent(None)
        while not stop.is_set():
            try:
                rss = measured.memory_info().rss / 1048576
                cpu = measured.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss, cpu = 0., 0.
            resource_rows.append({"t_utc": utc(), "elapsed_s": time.monotonic() - starting,
                                  "rss_mb": rss, "process_cpu_pct": cpu,
                                  "system_cpu_pct": psutil.cpu_percent(None)})
            stop.wait(cfg.sample_interval_s)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    target_rate = requests / max(seconds, 1)
    next_due = starting
    try:
        while len(timestamps) < requests or time.monotonic() - starting < seconds:
            if switch_to and switch_status is None and len(timestamps) >= requests // 2:
                switch_status, state = http(base, "/rollback", "POST", timeout=cfg.request_timeout_s)
                if switch_status != 200 or state.get("active") != switch_to:
                    raise RuntimeError(f"rollback under load failed: {switch_status} {state}")
                model_version = switch_to
                n_jobs = 1
                if switched_reference is not None:
                    stream.rebase(switched_reference, "rollback_after")
            minute = int((time.monotonic() - starting) // 60)
            factor = (0.5 if minute % 2 == 0 else 1.5) if name == "clean_dl_variable_rps" else 1.0
            dynamic_rate = max(0.01, target_rate * factor)
            now = time.monotonic()
            if next_due > now:
                time.sleep(min(next_due - now, 0.2))
                continue
            batch = raw[np.arange(rowidx, rowidx + batch_size) % len(raw)]
            rowidx += len(batch)
            items = [{"x1": float(a), "x2": float(b)} for a, b in batch]
            payload = {"items": items, "n_jobs": n_jobs}
            next_due += 1 / dynamic_rate
            t0 = time.perf_counter()
            endpoint = "/lab8/predict/validated" if validated else "/predict/batch"
            status, answer = http(base, endpoint, "POST", payload, timeout=cfg.request_timeout_s)
            latency = (time.perf_counter() - t0) * 1000
            valid = (status == 200 and answer.get("count") == batch_size and
                     len(answer.get("results", [])) == batch_size and
                     all(r.get("model_version") == model_version for r in answer["results"]))
            if validated:
                valid = valid and answer.get("silent_accepts") == 0 and (
                    answer.get("accepted_count", -1) + answer.get("rejected_count", -1) == batch_size)
            row = {"request_id": len(timestamps), "stage": name, "time_utc": utc(),
                   "status": 200 if valid else (status or 0),
                   "http_status": status, "latency_ms": latency, "batch_size": batch_size,
                   "n_jobs": n_jobs, "target_rps": dynamic_rate, "minute": minute, "expected_version": model_version,
                   "success": valid,
                   "rejected_count": answer.get("rejected_count", 0) if validated else 0,
                   "silent_accepts": answer.get("silent_accepts", 0) if validated else 0}
            timestamps.append(row)
            if valid:
                probabilities = (np.asarray(answer["monitor_probabilities"], dtype=float) if validated else
                                 np.asarray([r["probability"] for r in answer["results"]], dtype=float))
                events = stream.add(batch, probabilities)
            if len(timestamps) % 1000 == 0:
                print(f"[{name}] {len(timestamps)} requests, elapsed={time.monotonic()-starting:.1f}s", flush=True)
            # Real monitor HTTP observations; batch sampling prevents gratuitous 10x HTTP overhead.
            if valid and len(timestamps) % 20 == 0:
                http(monitor, "/ingest", "POST", {"x1": items[0]["x1"], "x2": items[0]["x2"],
                                                       "probability": float(probabilities[0])})
        elapsed = time.monotonic() - starting
    finally:
        stop.set()
        sampler.join(timeout=3)
    write_csv(report / f"{name}_requests.csv", timestamps)
    usage_path = report / f"{name}_resources.csv"
    write_csv(usage_path, resource_rows)
    plot_usage(resource_rows, usage_path)
    summary = stage_summary(name, timestamps, elapsed, seconds, requests, baseline,
                            cfg.max_latency_ratio)
    summary["versions_observed"] = sorted({r["expected_version"] for r in timestamps if r["success"]})
    if validated:
        summary["validator_rejected_samples"] = sum(r["rejected_count"] for r in timestamps)
        summary["validator_total_samples"] = len(timestamps) * batch_size
        summary["silent_accepts"] = sum(r["silent_accepts"] for r in timestamps)
        summary["pass"] &= summary["silent_accepts"] == 0
    if switch_to:
        summary["rollback_status"] = switch_status
        summary["pass"] &= switch_status == 200 and set(summary["versions_observed"]) == {"v3_dl", switch_to}
    if resource_rows:
        delta = [b["elapsed_s"] - a["elapsed_s"] for a, b in zip(resource_rows, resource_rows[1:])]
        summary["sample_interval_max_s"] = max(delta, default=0)
        summary["pass"] &= (max(delta, default=0) <= 2.1)
    return summary, timestamps


def full_drills(report: Path, runtime: Path, cfg: Lab8FullConfig, quick: bool) -> dict:
    report.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    registry = setup_registry(ROOT, runtime / "registry")
    base = f"http://127.0.0.1:{cfg.model_port}"
    monitor = f"http://127.0.0.1:{cfg.monitor_port}"
    model_proc = None
    monitor_proc = None
    frame_ref = generate_dataframe(GeneratorSettings(mode="similar", n_samples=5000, seed=cfg.seed))
    ml = joblib.load(ROOT / "reports/LAB2/ml_model.joblib")
    xref = frame_ref[["x1", "x2"]].to_numpy()
    reference_probability = positive(ml, xref)
    dl_ref_p = load_checkpoint(ROOT / "reports/LAB3/dl_model.pt").predict_proba(xref)
    alerts_path = report / "alerts_sample.jsonl"
    alerts_path.write_text("", encoding="utf-8")
    stream = StreamGuard(xref, reference_probability, window=cfg.window_size, step=cfg.window_step,
                         threshold=cfg.drift_psi_threshold, alert_path=alerts_path,
                         checkpoint=runtime / "stream_checkpoint.json", cooldown=cfg.alert_cooldown_seconds)
    alert = AlertManager(alerts_path, cooldown_seconds=cfg.alert_cooldown_seconds)
    reports = []
    try:
        monitor_proc = start_service(ROOT, "monitor", cfg.monitor_port, log=report / "monitor_log.jsonl")
        model_proc = start_service(ROOT, "model", cfg.model_port, registry=registry)
        wait_healthy(base)
        wait_healthy(monitor)
        baseline = cfg.baseline_p95_ms
        if baseline <= 0:
            # LAB6 published service metrics used as explicitly labeled baseline,
            # not represented as a new LAB6 network measurement.
            lab6 = json.loads((ROOT / "reports/LAB6/service_metrics.json").read_text(encoding="utf-8"))
            baseline = float(lab6["p95_request_latency_ms"])
        if baseline <= 0:
            raise ValueError("non-positive LAB6 baseline; supply --baseline-ms")
        stages = [
            ("warmup", "v2", 1, False),
            ("clean_ml_sequential", "v2", 1, False),
            ("clean_dl_parallel", "v3_dl", max(1, os.cpu_count() or 1), False),
            ("clean_dl_variable_rps", "v3_dl", max(1, os.cpu_count() or 1), False),
            ("drift_dl_sequential", "v3_dl", 1, True),
            ("rollback_under_load", "v3_dl", 1, False),
        ]
        first_drift_window = None
        first_alert_window = None
        for name, version, jobs, shifted in stages:
            stream.rebase(reference_probability if shifted else
                          (dl_ref_p if version == "v3_dl" else reference_probability), name)
            stage_first_window = stream.window_index + 1
            current = http(base, "/models")[1]["active"]
            if current != version:
                code, msg = http(base, "/promote", "POST", {"version": version, "reason": f"LAB8 stage {name}"})
                if code != 200:
                    raise RuntimeError(f"promote {version}: {code} {msg}")
            if cfg.source == "real":
                data = pd.read_csv(ROOT / "data/processed/test.csv")
                if shifted:
                    data = data.copy()
                    data["x1"] += cfg.drift_shift_x
            else:
                data = generate_dataframe(GeneratorSettings(mode="similar", n_samples=6000,
                        seed=cfg.seed if not shifted else cfg.seed + 122,
                        shift_x=cfg.drift_shift_x if shifted else 0.0))
            sec = 3 if quick else (cfg.warmup_seconds if name == "warmup" else cfg.stage_seconds)
            reqs = 30 if quick else (max(100, cfg.min_stage_requests // 2) if name == "warmup" else cfg.min_stage_requests)
            if name == "clean_dl_variable_rps" and not quick:
                # Per-minute pacing varies between 0.5x and 1.5x target; see requests CSV.
                pass
            summary, rows = measure_stage(name, base, monitor, cfg, data, stream, seconds=sec,
                                          requests=reqs, n_jobs=jobs, batch_size=max(cfg.batch_size, jobs),
                                          report=report, proc=model_proc, quick=quick,
                                          baseline=baseline, model_version=version,
                                          switch_to="v2" if name == "rollback_under_load" else None,
                                          switched_reference=reference_probability if name == "rollback_under_load" else None,
                                          validated=shifted)
            if name == "warmup":
                summary["pass"] = summary["elapsed_s"] >= sec and summary["total"] >= reqs and summary["lost_or_failed"] == 0
            # Every stage evaluates applicability and service health; alerts are real and deduplicated.
            current_features = data[["x1", "x2"]].to_numpy()
            current_mlp = positive(ml, current_features)
            validator_all = load_validator(ROOT / "reports/LAB4/validator_params.npz")
            rejected_fraction = float(np.mean(~validator_all.accept(current_features, current_mlp)))
            service_error = http(base, "/metrics")[1].get("error_rate", 0.0)
            if name != "warmup":
                alert.evaluate(AlertRule("validator_reject_rate", cfg.validator_reject_rate_threshold,
                                         "above", "critical", "inspect validator and data source"), rejected_fraction)
                alert.evaluate(AlertRule("service_error_rate", cfg.service_error_rate_threshold,
                                         "above", "critical", "restart/inspect model API"), float(service_error))
            summary["observed_validator_reject_rate"] = rejected_fraction
            summary["observed_service_error_rate"] = float(service_error)
            if shifted:
                first_drift_window = stage_first_window
                first_alert_window = next((r["window"] for r in stream.rows
                    if r["phase"] == name and r["window"] >= stage_first_window and r["alert"]), None)
                validator = load_validator(ROOT / "reports/LAB4/validator_params.npz")
                x = data[["x1", "x2"]].to_numpy()
                probs = positive(ml, x)
                rate = float(np.mean(~validator.accept(x, probs)))
                observed_gate_rate = summary.get("validator_rejected_samples", 0) / max(1, summary.get("validator_total_samples", 1))
                summary["validator_reject_rate"] = observed_gate_rate
                summary["validator_offline_reject_rate"] = rate
                summary["validator_threshold_pass"] = observed_gate_rate >= cfg.validator_drift_reject_min and abs(observed_gate_rate - rate) <= .05
                summary["pass"] &= summary["validator_threshold_pass"]
            reports.append(summary)
            print(json.dumps(summary, indent=2, default=str), flush=True)
        stream.export_csv(report / "stream_windows.csv")
        stream.save()
        model_proc, faults = fault_suite(ROOT, base, monitor, registry, model_proc,
                                        report, cfg, alert, stream)
        (report / "faults.json").write_text(json.dumps(faults, indent=2), encoding="utf-8")
        clean = [r for r in stream.rows if r["phase"] in {"warmup", "clean_ml_sequential", "clean_dl_parallel", "clean_dl_variable_rps", "rollback_under_load", "rollback_after"}]
        false_alerts = sum(bool(r["alert"]) for r in clean)
        drill = {"stage_results": reports, "faults": faults,
                 "false_alerts_on_clean": false_alerts,
                 "first_drift_window": first_drift_window,
                 "first_alert_window": first_alert_window,
                 "drift_delay_windows": (first_alert_window - first_drift_window)
                         if first_alert_window is not None and first_drift_window is not None else None,
                 "valid_duration": all(r["pass"] for r in reports),
                 "faults_pass": all(item["pass"] for item in faults)}
        drill["all_pass"] = (drill["valid_duration"] and drill["faults_pass"] and
                              false_alerts <= cfg.max_clean_alerts and
                              drill["drift_delay_windows"] is not None and
                              drill["drift_delay_windows"] <= cfg.detection_deadline_windows)
        return drill
    finally:
        stop_service(model_proc)
        stop_service(monitor_proc)


def write_report(report: Path, cfg: Lab8FullConfig, e2e_data: pd.DataFrame,
                 result: dict, mode: str, run_id: str) -> None:
    from moons_lab.lab8_observability import hash_manifest
    import hashlib
    code = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    input_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [ROOT / "reports/LAB2/ml_model.joblib", ROOT / "reports/LAB3/dl_model.pt",
                              ROOT / "reports/LAB4/validator_params.npz", ROOT / "configs/lab8_full.json"]}
    lines = ["# ЛР8. Сборка, алерты, эксплуатация под нагрузкой и отказоустойчивость", "",
             f"**Коммит кода:** `{code}`. **Хеши исходных артефактов:** " +
             "; ".join(f"`{k}` = `{v}`" for k,v in input_hashes.items()) + ".",
             f"**Режим:** `{mode}`; **дата UTC:** {utc()}; **вариант:** 09 make_moons.",
             f"**MLflow Run ID:** `{run_id}`.",
             "**Исходные модели:** опубликованные LAB2, LAB3 и валидатор LAB4; данные LAB5.",
             "", "## Сквозная проверка real / synthetic", "",
             "Дважды выполнены детерминированные прогоны; источник: `end_to_end.csv`.", "",
             "## Семь CLI-команд", "",
             "`train_ml`, `train_dl`, `train_validator`, `generate`, `serve`, `evaluate`, `monitor` — `pyproject.toml`.",
             "", "## Нагрузочные этапы", "",
             "Числа ниже получены по реальному HTTP на двух Uvicorn-процессах, не через TestClient.",
             "Порог p95 из `reports/LAB6/service_metrics.json` либо явно заданный baseline; источник и допуск записаны в JSON.",
             "", "| Этап | Запросы | Время, с | Успешно | p95, мс | RPS | Выполнено |",
             "|---|---:|---:|---:|---:|---:|---|",]
    for item in result["stage_results"]:
        lines.append(f"| {item['stage']} | {item['total']} | {item['elapsed_s']:.2f} | "
                     f"{item['successful']} | {item['p95_ms']:.3f} | {item['rps']:.2f} | {item['pass']} |")
    lines.extend(["", "## Отказоустойчивость C1–C5", "",
                  "| Сценарий | Вердикт |", "|---|---|", *[
                      f"| {item['case']} | {item['pass']} |" for item in result["faults"]],
                  "", "## Отчёт о выявленных ограничениях", "",
                  "- Сервис ЛР6 исходно обслуживает joblib-модели, адаптер LAB8 подгружает PyTorch MLР из неизменённого LAB3 checkpoint.",
                  "- В C2 использован отдельный процессный режим обслуживания пакета; штатная ЛР7 использует потоки joblib.",
                  "- Проверки времени и p95 используют фактические часы и результаты HTTP. Превышение допусков сохраняется как FAIL, а не исправляется вручную.",
                  "- При обрыве потока C4 записывается checkpoint; буфер ограничен размером окна.",
                  "- Дрейфовый этап использует отдельный `/lab8/predict/validated`: отказы LAB4 возвращаются явно, "
                  "а отклонённые записи не получают модельный прогноз. CSI для этого участка вычисляется "
                  "по вероятностям контрольной ML-модели валидатора.",
                  "- Внешний наблюдатель отсутствует: аварийная остановка model-server фиксируется процессом испытания.",
                  "", "## 5 групп обнаруживаемых отказов", "",
                  "1. Невалидный Pydantic — C5 завершает скрипт без запуска сервиса.",
                  "2. Хеш модели нарушен — C3 возвращает 409, рабочая модель продолжает отвечать.",
                  "3. Потеря Uvicorn — C1 фиксирует здоровье и восстановление.",
                  "4. Остановка исполнителя — C2 проверяет настоящий процесс и fallback.",
                  "5. Остановка потока — C4 возобновляет буфер с контрольной точки.",
                  "", "## Методика измерений", "",
                  "Все запросы отправляются через urllib к реальному TCP-порту; задержка — от отправки до получения ответа.",
                  "Система CPU/RSS измеряется независимым потоком через psutil каждые <=2 с; CSV и графики по этапам.",
                  "Число запросов и время оцениваются по монотонным часам; в quick доказательство времени отсутствует.",
                  "", "## Итоговый вердикт", "", f"`all_pass={result['all_pass']}`. Для сдачи полный прогон обязателен; quick — не доказательство."])
    (report / "lab8_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--config", default="configs/lab8_full.json")
    parser.add_argument("--report-dir", type=Path, default=Path("demo_output/LAB8"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("demo_output/LAB8_runtime"))
    parser.add_argument("--baseline-ms", type=float, default=None)
    parser.add_argument("--install-check", action="store_true")
    args = parser.parse_args()
    cfg = load_lab8_config(args.config)
    if args.baseline_ms is not None:
        cfg = cfg.model_copy(update={"baseline_p95_ms": args.baseline_ms})
        if cfg.baseline_p95_ms <= 0:
            parser.error("--baseline-ms must be positive")
    report = args.report_dir.resolve()
    runtime = args.runtime_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    e = e2e(report, cfg.seed)
    if args.install_check or args.full:
        cmd = [sys.executable, "scripts/lab8_install_check.py", "--log", str(report / "install.log")]
        if args.quick:
            cmd.append("--quick")
        subprocess.run(cmd, cwd=ROOT, check=True)
    if args.quick and not args.install_check:
        (report / "install.log").write_text("QUICK: clean dependency install not performed\n", encoding="utf-8")
    with tracked_run("lab8_full" if args.full else "lab8_quick", experiment="lab8_operations") as run:
        run.log_params({"mode": "full" if args.full else "quick", "config": str(args.config),
                        "git_sha": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()})
        result = full_drills(report, runtime, cfg, quick=args.quick)
        result["mlflow_run_id"] = run.run_id
        result["tracking_backend"] = run.backend
        result["evidence_mode"] = "full" if args.full else "SMOKE_ONLY"
        (report / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_report(report, cfg, e, result, result["evidence_mode"], run.run_id)
        for k, v in (("all_pass", float(result["all_pass"])),
                     ("false_alerts", float(result["false_alerts_on_clean"]))):
            run.log_metrics({k: v})
        for stage in result["stage_results"]:
            slug = stage["stage"]
            run.log_metrics({f"{slug}_p95_ms": float(stage["p95_ms"]),
                             f"{slug}_rps": float(stage["rps"]),
                             f"{slug}_successful": float(stage["successful"])})
        for filename in ("summary.json", "lab8_report.md", "end_to_end.csv"):
            run.log_artifact(report / filename)
    from moons_lab.lab8_observability import hash_manifest
    (report / "defense_checklist.md").write_text(
        "# Защитный чек-лист (проверить и подписать вручную)\n\n"
        "- [ ] ЛР1–3: ранее сданы; файлы не менялись\n"
        "- [ ] ЛР4–7: CI и зафиксированные хеши проверены\n"
        "- [ ] ЛР8: полный прогон, >= 10000 запросов и >=600с на каждый основной этап\n"
        "- [ ] C1–C5: журналы инцидентов и скринкасты\n"
        "- [ ] Отчёты, критическая рецензия БЯМ, раскрытие использования ИИ\n"
        "- [ ] Преподаватель: поле приёмки __________________\n", encoding="utf-8")
    (report / "artifacts_manifest.json").write_text(
        json.dumps(hash_manifest(ROOT, report / "artifacts_manifest.json"), indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(report / "summary.json"), "mode": result["evidence_mode"],
                      "all_pass": result["all_pass"], "mlflow": result["mlflow_run_id"]}, indent=2))
    if args.full and not result["all_pass"]:
        raise SystemExit("FULL LAB8 RUN COMPLETED WITH FAILED ACCEPTANCE CRITERIA (see report); no silent pass")


if __name__ == "__main__":
    main()
