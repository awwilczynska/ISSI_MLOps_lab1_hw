from fastapi import FastAPI
from api.models.sentiment import PredictRequest, PredictResponse
import inference


app = FastAPI()
transformer = inference.load_transformer()
classifier = inference.load_classifier()


@app.get("/")
def welcome_root():
    return {"message": "Welcome to the Sentiment analysis API"}


@app.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    prediction = inference.sentiment_analysis(request.text, transformer, classifier)
    return PredictResponse(prediction=prediction)
