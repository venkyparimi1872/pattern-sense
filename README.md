# Pattern Sense — Fabric Pattern Classifier
### FastAPI inference server for CNN + ResNet50 models

---

## Project Structure

```
patse_app/
├── main.py                  ← FastAPI backend
├── requirements.txt
├── cnn_best_model.pth       ← place your model files here
├── resnet50_best_model.pth  ← place your model files here
└── templates/
    └── index.html           ← frontend UI
```

---

## Setup & Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Place your model files**

Copy `cnn_best_model.pth` and `resnet50_best_model.pth` (from your Kaggle output)
into the `patse_app/` folder next to `main.py`.

**3. Run the server**
```bash
cd patse_app
uvicorn main:app --reload --port 8000
```

**4. Open in browser**
```
http://localhost:8000
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Web UI |
| POST | `/predict` | Classify an image |
| GET | `/models` | List loaded models + classes |

### POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@my_fabric.jpg" \
  -F "model=ResNet50"
```

**Response:**
```json
{
  "prediction": "floral",
  "confidence": 0.8732,
  "all_scores": [
    {"label": "floral",   "confidence": 0.8732},
    {"label": "cartoon",  "confidence": 0.0614},
    ...
  ],
  "model_val_acc": 0.6262
}
```

---

## Supported Pattern Classes
animal · cartoon · floral · geometry · ikat · plain · polka dot · squares · stripes · tribal

---

## Notes
- The app auto-detects GPU if available (falls back to CPU)
- Either model file is optional — the app loads whichever `.pth` files are present
- Model checkpoint must contain keys: `model_state_dict`, `classes`, `label_to_idx`, `img_size`, `mean`, `std`
  (all saved automatically by the training script)
