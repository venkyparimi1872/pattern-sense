# Pattern Sense: Classification Fabric Patterns

Pattern Sense is an end-to-end deep learning project for multi-class fabric pattern recognition.
The project includes model training, evaluation, confusion-matrix analysis, a FastAPI inference service, and a web interface for interactive prediction.

## 1. Project Objective

The main objective is to classify fabric images into pattern categories using deep learning technologies, and compare a custom CNN with a transfer-learning approach (ResNet50 fine-tuning).

This project focuses on:
- practical model performance on real fabric textures
- robust data handling and preprocessing
- deployable inference via API and UIs

## 2. Problem Statement

Fabric and textile patterns — stripes, polka dots, floral prints, and so on, are everywhere in fashion, retail, and manufacturing, but identifying and sorting them is still mostly done by hand. This is slow, inconsistent (different people categorize patterns differently), and doesn't scale when you have thousands of images to sort, such as in an online catalog or a fabric inventory system.

The task is framed as supervised multi-class classification. Given an image, the model predicts which pattern type it belongs to, removing the need for manual tagging

## 3. Dataset and Classes

The training pipeline loads images from class-wise directories and validates files before use.
Corrupt files are skipped automatically.

Classes used in the current setup:
- animal
- floral
- plain
- polka dot
- squares
- stripes

The pipeline performs stratified splitting:
- Train: 70%
- Validation: 15%
- Test: 15%

## 4. Folder Structure

```text
patse_app/
├── app.py                  # FastAPI inference backend
├── main.py                 # Training, evaluation, and checkpoint saving
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI template
├── static/                 # Static assets for the UI
└── images/        # Training/evaluation figures used in the README
```

## 5. System Architecture

The system has three layers:

1. Training and evaluation pipeline
- File: main.py
- Handles data preprocessing, model training, early stopping, checkpoint saving,
  learning-curve plotting, and confusion-matrix evaluation.

2. Inference backend
- File: app.py
- Loads available model checkpoints, preprocesses uploaded images,
  runs prediction, and returns ranked class probabilities.

3. Frontend UI
- File: templates/index.html
- Supports image upload, model selection, and prediction visualization.

## 6. Model Architectures

### 5.1 Custom CNN

Architecture highlights:
- 4 convolution blocks with BatchNorm, ReLU, MaxPool
- Feature progression: 3 -> 32 -> 64 -> 128 -> 256 channels
- Input size: 224 x 224 RGB
- Classifier head:
  - Flatten
  - Linear(256 x 14 x 14 -> 512) + ReLU + Dropout(0.5)
  - Linear(512 -> 256) + ReLU + Dropout(0.3)
  - Linear(256 -> number_of_classes)

### 5.2 ResNet50 (Transfer Learning)

Approach:
- Base model: ResNet50 pretrained on ImageNet
- Replaced final fully connected layer with:
  - Dropout(0.4)
  - Linear(in_features -> number_of_classes)

Two-phase fine-tuning:
- Phase 1: freeze backbone, train classifier head
- Phase 2: unfreeze full model, train end-to-end with lower learning rate

## 7. Methodology

### 6.1 Preprocessing and Augmentation

- Resize images to 224 x 224
- Train augmentations:
  - Random horizontal flip
  - Random rotation (15 degrees)
  - Color jitter
- Normalization with ImageNet mean/std

### 6.2 Class Imbalance Handling

- WeightedRandomSampler for balanced training batches
- Class-weighted CrossEntropyLoss (sqrt-scaled inverse-frequency weights)

### 6.3 Optimization Strategy

Custom CNN:
- Optimizer: Adam
- LR: 1e-3
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR
- Early stopping patience: 10

ResNet50:
- Phase 1 LR: 1e-3
- Phase 2 LR: 1e-4
- Weight decay: 1e-4
- Scheduler: CosineAnnealingLR
- Early stopping patience: 10

## 8. Tech Stack

Core libraries:
- Python
- PyTorch, Torchvision
- NumPy, Pandas, scikit-learn
- Matplotlib, Pillow

## 9. Results and Comparative Analysis

Based on the latest training outputs:

- Custom CNN test accuracy: 68.11%
- ResNet50 (fine-tuned) test accuracy: 85.37%

### Training and evaluation figures

Custom CNN learning curves:

![Custom CNN training loss and accuracy](/images/cnn_training_curves.png)

Custom CNN confusion matrix:

![Custom CNN confusion matrix](/images/cnn_matrix.png)

ResNet50 learning curves:

![ResNet50 training loss and accuracy](/images/resnet_training_curves.png)

ResNet50 confusion matrix:

![ResNet50 confusion matrix](/images/resnet_matrix.png)

### Key observations

1. Custom CNN
- Training accuracy increased strongly, but validation accuracy saturated around high-60%.
- Loss gap between train and validation widened in later epochs.
- This indicates overfitting and limited generalization.

2. ResNet50 fine-tuned
- Clear and stable improvement in validation metrics during training.
- Achieved substantially better generalization than the custom CNN.
- Confusion matrix is much more diagonal, indicating better class separation.

### Confusion-matrix insights

- Custom CNN shows more cross-class confusion, especially among texture-similar classes.
- ResNet50 reduces these confusions and improves consistency across most classes.

## 10. Challenges Faced

Main challenges encountered during development:

- Visual similarity between some pattern categories
- Class imbalance and skewed sample distribution
- Overfitting in the custom CNN at higher epochs
- Need for robust inference preprocessing compatibility between training and deployment

Mitigations applied:

- stronger augmentation pipeline
- weighted sampling and weighted loss
- early stopping
- transfer learning with staged fine-tuning

## 11. Inference Workflow

At runtime, the backend:
- loads available checkpoints
- reads metadata (class list, image size, normalization values)
- preprocesses uploaded image exactly as required by the model
- returns:
  - top prediction
  - top confidence
  - sorted class-confidence scores

## 12. API Summary

- GET / : UI page
- GET /models : available models and classes
- POST /predict : image classification endpoint

Example request:

curl -X POST http://localhost:8000/predict \
  -F "file=@sample.jpg" \
  -F "model=ResNet50"

## 13. How to Run

1. Install dependencies

pip install -r requirements.txt

2. Ensure at least one checkpoint is present in project root
- cnn_best_model.pth
- resnet50_best_model.pth

3. Start server

uvicorn app:app --reload --port 8000

4. Open UI

http://localhost:8000

## 14. Future Improvements

Potential next steps:
- add per-class precision, recall, and F1 reporting
- experiment with efficient modern backbones (EfficientNet, ConvNeXt)
- add test-time augmentation
- add model explainability (Grad-CAM)