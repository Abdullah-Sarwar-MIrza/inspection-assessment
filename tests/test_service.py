"""Real trained-model integration and malformed-upload boundary checks."""
import io
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from casting_inspector.api import app
from casting_inspector.model import MAX_BYTES, Predictor, decode_image


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def image_bytes(size=(64, 64), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format=fmt)
    return buffer.getvalue()


def test_health(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["status"] == "ready"


def test_api_cli_parity_and_real_prediction(client):
    examples = json.loads(Path("reports/example_predictions.json").read_text())
    for example in examples:
        raw = Path(example["image"]).read_bytes()
        response = client.post("/predict", files={"file": ("image.jpg", raw, "image/jpeg")})
        assert response.status_code == 200
        result = response.json()
        expected = Predictor().predict(decode_image(raw))
        assert result["predicted_class"] == expected["predicted_class"]
        assert result["defect_probability"] == pytest.approx(expected["defect_probability"], abs=1e-6)
        assert result["confidence"] == pytest.approx(example["prediction"]["confidence"], abs=1e-5)
        assert response.headers["X-Request-ID"] == result["request_id"]
        assert result["latency_ms"] > 0


@pytest.mark.parametrize("raw,mime,status", [
    (b"not-an-image", "image/jpeg", 422),
    (b"", "image/png", 422),
    (b"hello", "text/plain", 415),
    (image_bytes((16, 16)), "image/png", 422),
    (image_bytes(fmt="GIF"), "image/png", 422),
    (image_bytes()[:30], "image/png", 422),
    (b"x" * (MAX_BYTES + 1), "image/jpeg", 413),
], ids=["corrupt", "empty", "unsupported-mime", "too-small", "disguised-gif", "truncated", "too-many-bytes"])
def test_invalid_upload(client, raw, mime, status):
    assert client.post("/predict", files={"file": ("x", raw, mime)}).status_code == status


def test_missing_upload(client):
    assert client.post("/predict").status_code == 422


def test_body_limit_without_content_length(client):
    def chunks():
        for _ in range(9):
            yield b"x" * (1024 * 1024)
    response = client.post("/predict", content=chunks(), headers={"Content-Type": "application/octet-stream"})
    assert response.status_code == 413


def test_pixel_limit(client):
    response = client.post("/predict", files={"file": ("large.png", image_bytes((4001, 4000)), "image/png")})
    assert response.status_code == 422


def test_unexpected_error_is_sanitized(client, monkeypatch):
    def fail(_):
        raise RuntimeError("private internal detail")
    monkeypatch.setattr(app.state.predictor, "predict", fail)
    response = client.post("/predict", files={"file": ("x.png", image_bytes(), "image/png")})
    assert response.status_code == 500
    assert response.json()["detail"] == "Internal inference error"
    assert "private internal detail" not in response.text


def test_png_inference(client):
    response = client.post("/predict", files={"file": ("x.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    assert 0 <= response.json()["confidence"] <= 1


def test_model_integrity(tmp_path):
    (tmp_path / "metadata.json").write_bytes(Path("artifacts/metadata.json").read_bytes())
    (tmp_path / "model.pt").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity"):
        Predictor(tmp_path)


def test_split_isolation():
    import csv
    with open("reports/split_manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hashes = {s: {r["sha256_pixels"] for r in rows if r["split"] == s}
              for s in ("train", "validation", "test")}
    assert not hashes["train"] & hashes["validation"]
    assert not hashes["train"] & hashes["test"]
    assert not hashes["validation"] & hashes["test"]
