"""Start real FastAPI model or independent monitor using Uvicorn.

Examples:
  python scripts/serve_lab6.py --service monitor --port 8001
  python scripts/serve_lab6.py --service model --port 8000 --monitor-url http://127.0.0.1:8001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moons_lab.lab6_bootstrap import build_registry
from moons_lab.monitor_service import MonitorState, create_monitor_app
from moons_lab.service import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", choices=["model", "monitor"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--monitor-url", default=None)
    parser.add_argument("--runtime-dir", type=Path, default=ROOT / "demo_output" / "LAB6_runtime")
    parser.add_argument("--reset-registry", action="store_true", help="explicitly reset LAB6 runtime state")
    args = parser.parse_args()
    if args.service == "monitor":
        app = create_monitor_app(MonitorState(log_path=args.runtime_dir / "monitor_observations.jsonl"))
        port = args.port or 8001
    else:
        registry = build_registry(ROOT, args.runtime_dir / "registry", reset=args.reset_registry)

        def sink(x1: float, x2: float, probability: float) -> None:
            if not args.monitor_url:
                return
            url = args.monitor_url.rstrip("/") + "/ingest"
            data = json.dumps({"x1": x1, "x2": x2, "probability": probability}).encode("utf-8")
            req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=.35) as response:
                if response.status != 200:
                    raise RuntimeError(f"monitor HTTP {response.status}")

        def request_sink(path: str, status_code: int, latency_ms: float,
                         model_version: str, model_uptime_s: float) -> None:
            if not args.monitor_url:
                return
            url = args.monitor_url.rstrip("/") + "/ingest/service"
            body = {"path": path, "status_code": status_code,
                    "latency_ms": latency_ms, "model_version": model_version,
                    "model_uptime_s": model_uptime_s}
            req = Request(url, data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=.35) as response:
                if response.status != 200:
                    raise RuntimeError(f"monitor HTTP {response.status}")

        app = create_app(registry, monitor_sink=sink if args.monitor_url else None,
                         monitor_request_sink=request_sink if args.monitor_url else None)
        port = args.port or 8000
    print(f"LAB6 {args.service} online: http://{args.host}:{port}/docs")
    uvicorn.run(app, host=args.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
