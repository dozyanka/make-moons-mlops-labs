from fastapi.testclient import TestClient

from moons_lab.monitor_service import MonitorState, Observation, create_monitor_app


def test_monitor_state_and_api(tmp_path):
    state = MonitorState()
    state.add(Observation(x1=1.0, x2=2.0, probability=0.7))
    assert state.metrics()["count"] == 1
    state.save_plot(tmp_path / "plot.png")
    assert (tmp_path / "plot.png").exists()
    client = TestClient(create_monitor_app(state))
    assert client.get("/health").status_code == 200
    assert client.post("/ingest", json={"x1": 0, "x2": 0, "probability": 0.5}).status_code == 200
    assert client.get("/metrics").json()["count"] == 2
