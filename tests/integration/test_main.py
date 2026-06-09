from fastapi.testclient import TestClient

import backend.main as main


client = TestClient(main.app)


def test_home():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"message": "Backend is running!"}


def test_state_endpoint():
    resp = client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert "temperature" in data
    assert "workload" in data
    assert "cooling" in data


def test_optimize_endpoint_updates_state_and_logs_metrics(monkeypatch):
    monkeypatch.setattr(main, "predict_action", lambda state: 2)

    initial_state_resp = client.get("/state")
    assert initial_state_resp.status_code == 200
    initial_state = initial_state_resp.json()

    opt_resp = client.post("/optimize")
    assert opt_resp.status_code == 200
    new_state = opt_resp.json()

    for key in ("temperature", "workload", "cooling"):
        assert key in new_state

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    history = metrics_resp.json()
    assert isinstance(history, list)
    assert len(history) >= 1

    last_entry = history[-1]
    for key in (
        "timestamp",
        "step",
        "episode_id",
        "model_version",
        "temperature",
        "workload",
        "cooling",
        "action",
        "energy",
        "reward",
        "done",
        "temperature_delta",
    ):
        assert key in last_entry

    changed = (
        new_state["temperature"] != initial_state["temperature"]
        or new_state["cooling"] != initial_state["cooling"]
        or new_state["workload"] != initial_state["workload"]
    )
    assert changed