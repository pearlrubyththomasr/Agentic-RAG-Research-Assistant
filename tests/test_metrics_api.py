import sys
import types

from fastapi.testclient import TestClient

import api.main as main


def test_metrics_endpoint_monkeypatched(monkeypatch):
    # Provide a fake psutil module to ensure metrics endpoint works without psutil installed
    fake_psutil = types.SimpleNamespace()
    fake_psutil.virtual_memory = lambda: types.SimpleNamespace(total=1000, used=500, percent=50)
    fake_psutil.cpu_percent = lambda interval=0.1: 1.2
    fake_psutil.Process = lambda: types.SimpleNamespace(memory_info=lambda: types.SimpleNamespace(rss=123, vms=456))
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    client = TestClient(main.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "cpu_percent" in data or "timestamp" in data
