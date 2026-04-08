import pytest
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from inference import load_transformer, load_classifier, sentiment_analysis


@pytest.fixture(scope="module")
def models() -> tuple[SentenceTransformer, LogisticRegression]:
    transformer = load_transformer()
    classifier = load_classifier()
    return transformer, classifier


def test_models_loading(models):
    transformer, classifier = models
    assert transformer is not None
    assert classifier is not None


@pytest.mark.parametrize(
    "text",
    [
        "This is her house",
        "I need to buy a ticket",
        "I feel really great",
        "I feel sad",
    ],
)
def test_inference_logic(models, text):
    transformer, classifier = models
    result = sentiment_analysis(text, transformer, classifier)
    assert result in ["positive", "neutral", "negative"]


def test_inference_expected_output():
    transformer = load_transformer()
    classifier = load_classifier()
    assert (
        sentiment_analysis("I feel really great", transformer, classifier) == "positive"
    )
    assert sentiment_analysis("I feel sad", transformer, classifier) == "negative"
    assert (
        sentiment_analysis("I need to buy a ticket", transformer, classifier)
        == "neutral"
    )
