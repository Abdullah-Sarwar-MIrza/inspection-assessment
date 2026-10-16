"""Shared training/serving preprocessing and portable inference model."""
import hashlib
import io
import json
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

IMAGE_SIZE = 160
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 16_000_000


def preprocess():
    # Preserve the whole image: small edge defects must not be cropped away.
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def decode_image(raw: bytes) -> Image.Image:
    if not raw or len(raw) > MAX_BYTES:
        raise ValueError("Image must contain 1 byte to 8 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in {"JPEG", "PNG"}:
                    raise ValueError("Only JPEG and PNG are supported")
                if im.width * im.height > MAX_PIXELS or min(im.size) < 32:
                    raise ValueError("Image dimensions must be at least 32x32 and at most 16 megapixels")
                im.load()
                return ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid or oversized image") from exc


class Inspector(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.register_buffer("feature_mean", torch.zeros(512))
        self.register_buffer("feature_scale", torch.ones(512))
        self.head = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head((features - self.feature_mean) / self.feature_scale).squeeze(-1)


class Predictor:
    def __init__(self, model_dir="artifacts"):
        root = Path(model_dir)
        self.metadata = json.loads((root / "metadata.json").read_text())
        if self.metadata.get("image_size") != IMAGE_SIZE:
            raise ValueError("Model preprocessing version mismatch")
        weight_path = root / "model.pt"
        digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
        if digest != self.metadata["sha256"]:
            raise ValueError("Model integrity verification failed")
        self.model = Inspector()
        self.model.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
        self.model.eval()
        self.transform = preprocess()
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE))

    def predict(self, image):
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(self.transform(image).unsqueeze(0))).item()
        defective = probability >= self.metadata["threshold"]
        return {
            "predicted_class": "Defective" if defective else "Normal",
            "confidence": probability if defective else 1 - probability,
            "defect_probability": probability,
            "threshold": self.metadata["threshold"],
            "model_version": self.metadata["model_version"],
        }
