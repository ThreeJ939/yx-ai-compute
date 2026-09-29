from fastapi.testclient import TestClient

from app.api import create_app
from app.config import Settings


def test_health_and_capabilities():
    settings = Settings(service_id="ai-compute-service")
    client = TestClient(create_app(settings))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "UP"
    caps = client.get("/capabilities")
    assert caps.status_code == 200
    body = caps.json()
    codes = {a["code"] for a in body["algorithms"]}
    assert "ENGINEERING_VEHICLE_DETECTION" in codes
    assert "FISHING_DETECTION" in codes


def test_preview_page_and_root_redirect():
    settings = Settings(service_id="ai-compute-service", preview_enabled=True)
    client = TestClient(create_app(settings))
    root = client.get("/", follow_redirects=False)
    assert root.status_code in (302, 307)
    assert root.headers["location"] == "/preview"
    page = client.get("/preview")
    assert page.status_code == 200
    assert "视频流" in page.text
    assert "/preview/stream" in page.text
