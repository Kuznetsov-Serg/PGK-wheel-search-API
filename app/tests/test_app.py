from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_dummy_endpoint():
    response = client.get("/dummy_endpoint")
    data = response.json()
    assert data == "dummy response"
