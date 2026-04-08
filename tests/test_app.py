import pytest
from fastapi import status
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("You disappointed me very much", "negative"),
        ("You are really awersome", "positive"),
        ("It is a ball", "neutral"),
    ],
)
def test_predict(text, expected):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"prediction": expected}


def test_welcome_root():
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Welcome to the Sentiment analysis API"}


def test_predict_validation_for_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "String should have at least 1 character" in detail[0]["msg"]
    assert "text" in detail[0]["loc"]


def test_predict_validation_for_missing_text_attr():
    response = client.post("/predict", json={})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Field required" in detail[0]["msg"]
    assert "text" in detail[0]["loc"]


def test_response_is_valid_json():
    response = client.post("/predict", json={"text": "I feel really great"})
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
