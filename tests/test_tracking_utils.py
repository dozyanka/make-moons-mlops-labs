from pathlib import Path

from moons_lab.tracking import tracked_run
from moons_lab.utils import json_dump, measured_call, seed_everything, sha256_bytes, sha256_file


def test_utils_and_fallback_tracking(tmp_path, monkeypatch):
    seed_everything(123)
    p = tmp_path / "x.txt"
    p.write_text("abc", encoding="utf-8")
    assert sha256_file(p) == sha256_bytes(b"abc")
    out = tmp_path / "a.json"
    json_dump(out, {"b": 1})
    assert out.exists()
    measured = measured_call(lambda x: x + 1, 2)
    assert measured.value == 3 and measured.elapsed_s >= 0 and measured.peak_memory_mb >= 0
    with tracked_run("test_run", "tests") as run:
        run.log_params({"a": 1})
        run.log_metrics({"m": 0.5})
        run.log_artifact(p)
        assert len(run.run_id) > 5
