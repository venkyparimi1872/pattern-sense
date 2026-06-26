"""
Run: uvicorn main:app --reload --port 8000
"""

import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms

# CNN Architecture

class PatSeCNN(nn.Module):
    """Custom CNN architecture used for multi-class fabric pattern classification."""

    def __init__(self, num_classes):
        """Initialize convolutional feature extractor and classifier head.

        Args:
            num_classes: Number of output classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3,  32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                          # 112×112
 
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),                          # 56×56
 
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),                           # 28×28

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),                            # 14×14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(512, 256),           nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
 
    def forward(self, x):
        """Run a forward pass and return class logits."""
        return self.classifier(self.features(x))

def build_resnet50(num_classes):
    """Build a ResNet50 model with a custom dropout + linear classification head."""

    net = resnet50(weights=None)
    in_features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return net


# Load Models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {}

def load_model(path: str, model_type: str):
    """Load a model checkpoint and return model plus metadata for inference.

    Args:
        path: Path to checkpoint file.
        model_type: Either "cnn" or "resnet50".

    Returns:
        A dictionary containing model object and inference metadata, or None
        if the checkpoint file does not exist.
    """

    path = Path(path)
    if not path.exists():
        print(f"  [WARN] {path} not found — skipping.")
        return None
    ckpt = torch.load(path, map_location=device)
    num_classes = ckpt["num_classes"]
    classes     = ckpt["classes"]
    label_to_idx = ckpt["label_to_idx"]

    if model_type == "cnn":
        model = PatSeCNN(num_classes)
    else:
        model = build_resnet50(num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    return {
        "model":       model,
        "classes":     classes,
        "label_to_idx": label_to_idx,
        "img_size":    ckpt.get("img_size", 224),
        "mean":        ckpt.get("mean", [0.485, 0.456, 0.406]),
        "std":         ckpt.get("std",  [0.229, 0.224, 0.225]),
        "val_acc":     ckpt.get("val_acc", None),
    }

print("Loading models…")
cnn_data = load_model("cnn_best_model.pth", "cnn")
r50_data = load_model("resnet50_best_model.pth", "resnet50")

if cnn_data:
    MODELS["CNN"]     = cnn_data
    print(f"  ✓ CNN loaded  (val_acc={cnn_data['val_acc']:.4f})")
if r50_data:
    MODELS["ResNet50"] = r50_data
    print(f"  ✓ ResNet50 loaded  (val_acc={r50_data['val_acc']:.4f})")

if not MODELS:
    raise RuntimeError("No model files found. Place cnn_best_model.pth and/or resnet50_best_model.pth next to main.py.")


# Inference helper

def preprocess(image_bytes: bytes, img_size: int, mean: list, std: list) -> torch.Tensor:
    """Decode and normalize raw image bytes into a model-ready tensor."""

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def predict(model_key: str, image_bytes: bytes) -> dict:
    """Run inference for a selected model and return ranked class scores."""

    data   = MODELS[model_key]
    tensor = preprocess(image_bytes, data["img_size"], data["mean"], data["std"])
    with torch.no_grad():
        logits = data["model"](tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_idx   = int(np.argmax(probs))
    top_label = data["classes"][top_idx]
    top_conf  = float(probs[top_idx])

    all_scores = [
        {"label": data["classes"][i], "confidence": float(probs[i])}
        for i in np.argsort(probs)[::-1]
    ]

    return {
        "prediction": top_label,
        "confidence": top_conf,
        "all_scores": all_scores,
        "model_val_acc": data["val_acc"],
    }


# FastAPI app

app = FastAPI(title="Fabric Pattern Classifier")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the home page with the list of available models."""

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models":  list(MODELS.keys()),
    })


@app.post("/predict")
async def predict_endpoint(
    file:  UploadFile = File(...),
    model: str        = "ResNet50",
):
    """Accept an uploaded image and return prediction results as JSON."""

    if model not in MODELS:
        raise HTTPException(400, f"Model '{model}' not available. Choose from: {list(MODELS.keys())}")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Upload must be an image file.")

    image_bytes = await file.read()
    result = predict(model, image_bytes)
    return JSONResponse(result)


@app.get("/models")
async def list_models():
    """Return available models and their metadata."""

    return {
        k: {"val_acc": v["val_acc"], "classes": v["classes"]}
        for k, v in MODELS.items()
    }
