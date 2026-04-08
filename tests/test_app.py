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


def test_predict_validation_error():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    assert "detail" in response.json()
