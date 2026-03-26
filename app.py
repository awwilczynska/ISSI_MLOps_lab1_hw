from fastapi import FastAPI
from api.models.sentiment import PredictRequest, PredictResponse


app = FastAPI()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the Sentiment analysis API"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    return PredictResponse(prediction="positive")
