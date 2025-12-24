"""Test API"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api.main import app
import src.api.main


# Initializing the test client
client = TestClient(app)


def test_read_root():
    """Test the root endpoint /."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert "available_endpoints" in response.json()


def test_health_check_no_model():
    """Test /health when the model is NOT loaded."""
    src.api.main.loader = None
    response = client.get("/health")
    mock_loader = MagicMock()
    mock_loader.model = None
    src.api.main.loader = mock_loader
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"detail": "Model not loaded"}


def test_health_check_success():
    """Test /health when the model IS loaded."""
    # Create a fake loader with a fake template.
    mock_loader = MagicMock()
    mock_loader.model = "FakeKerasModel"
    # Inject the fake model into the application
    src.api.main.loader = mock_loader
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


@patch("src.api.main.clean_text")  # Mock preprocessing to isolate the API.
def test_predict_positive_sentiment(mock_clean):
    """Test the /predict endpoint with a positive result."""
    # 1. Mock Configuration
    mock_clean.return_value = "cleaned text"  # Preprocessing result
    mock_loader = MagicMock()
    # Simulate the tokenizer
    mock_loader.tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    # Simulate the model prediction (Score > 0.5 = POSITIVE)
    mock_loader.model.predict.return_value = [[0.85]]
    # Inject the mock
    src.api.main.loader = mock_loader
    # 2. API Call
    payload = {"content": "I love this product!"}
    response = client.post("/predict", json=payload)
    # 3. Verifications
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "POSITIVE"
    assert data["confidence"] == 0.85
    # Verify that the model has been called
    mock_loader.model.predict.assert_called_once()


@patch("src.api.main.clean_text")
def test_predict_negative_sentiment(mock_clean):
    """Test the /predict endpoint with a negative result."""
    mock_clean.return_value = "cleaned text"
    mock_loader = MagicMock()
    mock_loader.tokenizer.texts_to_sequences.return_value = [[1, 2, 3]]
    # Score < 0.5 = NEGATIVE
    mock_loader.model.predict.return_value = [[0.15]]
    src.api.main.loader = mock_loader
    payload = {"content": "This is terrible."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert data["confidence"] == 0.15


def test_metrics_endpoint():
    """Test /metrics endpoint."""
    mock_loader = MagicMock()
    # Inject fake metrics
    mock_loader.metrics = {"accuracy": 0.95, "f1_score": 0.92}
    src.api.main.loader = mock_loader
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.json()["model_performance"]["accuracy"] == 0.95


def test_train_endpoint():
    """Test train endpoint (not start the training)."""
    with patch("src.api.main.BackgroundTasks.add_task") as mock_add_task:
        response = client.post("/train")
        assert response.status_code == 200
        assert response.json() == {
            "message": "Training pipeline triggered in background"
        }
        mock_add_task.assert_called_once()
