import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# %%
DATA_DIR    = "/kaggle/input/datasets/venky2729/6-classes/6-class"  
OUTPUT_DIR  = "/kaggle/working/"
 
SEED        = 42
IMG_SIZE    = 224         
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15
TEST_FRAC   = 0.15
BATCH_SIZE  = 32
 
# CNN-specific
CNN_EPOCHS      = 80
CNN_LR          = 1e-3
CNN_WEIGHT_DECAY = 1e-4
 
# ResNet50-specific  
R50_EPOCHS      = 30
R50_HEAD_LR     = 1e-3    
R50_FINETUNE_LR = 1e-4     
R50_HEAD_EPOCHS = 10
R50_WEIGHT_DECAY = 1e-4
 
EARLY_STOP_PATIENCE = 10   
 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
VALID_EXTS  = {'.jpg', '.jpeg', '.png', '.webp'}

CLASSES     = sorted(d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d)))
NUM_CLASSES = len(CLASSES)

label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
idx_to_label = {i: cls for cls, i in label_to_idx.items()}
 
print(f"Classes ({NUM_CLASSES}): {CLASSES}")
 
records, bad_files = [], []
for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    for fname in sorted(os.listdir(cls_dir)):
        if os.path.splitext(fname)[1].lower() not in VALID_EXTS:
            continue
        fpath = os.path.join(cls_dir, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
            records.append({"path": fpath, "label": cls, "label_idx": label_to_idx[cls]})
        except Exception:
            bad_files.append(fpath)
 
df = pd.DataFrame(records)
print(f"Valid images: {len(df)}   |   Corrupt / skipped: {len(bad_files)}")

df.to_csv(os.path.join(OUTPUT_DIR, "patse_dataset.csv"), index=False)
 
print("\nClass distribution:")

for cls, n in df['label'].value_counts().sort_index().items():
    print(f"  {cls:<15} {n:>4}")

# %%
X = df["path"].values
y = df["label_idx"].values
 
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=SEED
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC), stratify=y_tmp, random_state=SEED
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")


# %%

# Calculated on the complete dataset

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
 
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])
 
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

class PatSeDataset(Dataset):
    """Torch dataset that reads images from paths and returns label indices."""

    def __init__(self, paths: Sequence[str], labels: Sequence[int], transform=None):
        """Initialize dataset.

        Args:
            paths: Image file paths.
            labels: Integer labels aligned with paths.
            transform: Optional torchvision transform pipeline.
        """
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
 
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.paths)
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load one sample by index.

        Args:
            idx: Sample index.

        Returns:
            A tuple of (image_tensor, label_index).
        """
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(self.labels[idx])

# %%
class_counts  = pd.Series(y_train).value_counts().sort_index()
sample_weights = (1.0 / class_counts.loc[y_train].values)
 
sampler = torch.utils.data.WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
 
inv_freq          = (1.0 / class_counts) * len(y_train) / NUM_CLASSES
loss_class_weights = torch.tensor(np.sqrt(inv_freq.values), dtype=torch.float32).to(device)
 
train_dataset = PatSeDataset(X_train, y_train, transform=train_transform)
val_dataset   = PatSeDataset(X_val,   y_val,   transform=eval_transform)
test_dataset  = PatSeDataset(X_test,  y_test,  transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,    num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,       num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,       num_workers=2, pin_memory=True)
 
print(f"Batches — Train: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")


# %%
class PatSeCNN(nn.Module):
    """
    4-block CNN built for 224×224 RGB input.
    After 4× MaxPool2d(2,2):  224 → 112 → 56 → 28 → 14
    Final feature map: 64 channels × 14 × 14 = 12 544 → classifier.
    """
    def __init__(self, num_classes: int):
        """Initialize CNN feature extractor and classification head.

        Args:
            num_classes: Number of target classes.
        """
        super().__init__()
        self.features = nn.Sequential(
            
            nn.Conv2d(3,  32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                          # 112×112
 
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),                          # 56×56
 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.Conv2d(128,128, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),                           # 28×28

            
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
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images."""
        return self.classifier(self.features(x))
 
 
cnn_model = PatSeCNN(NUM_CLASSES).to(device)
n_params  = sum(p.numel() for p in cnn_model.parameters())
print(f"CNN — Total parameters: {n_params:,}")

# %%
def run_epoch(model, loader, loss_fn, optimizer=None, train_mode=True):
    """Run one full epoch.

    Args:
        model: Model to train/evaluate.
        loader: DataLoader providing image-label batches.
        loss_fn: Loss function.
        optimizer: Optimizer used only when train_mode is True.
        train_mode: If True, enables gradients and optimizer steps.

    Returns:
        Tuple of (average_loss, accuracy).
    """

    model.train() if train_mode else model.eval()
    total_loss, total, correct = 0.0, 0, 0
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for bf, bl in loader:
            bf, bl = bf.to(device, non_blocking=True), bl.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad()
            out  = model(bf)
            loss = loss_fn(out, bl)
            if train_mode:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * bf.size(0)
            _, pred     = torch.max(out, 1)
            total      += bl.size(0)
            correct    += (pred == bl).sum().item()
    return total_loss / total, correct / total


def plot_history(history, title_prefix, save_name):
    """Plot and save training history curves.

    Args:
        history: Dict-like structure with train/val loss and accuracy lists.
        title_prefix: Title prefix for plot labels.
        save_name: Output filename for the saved figure.
    """

    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ep, history["train_loss"], label="Train")
    axes[0].plot(ep, history["val_loss"],   label="Val")
    axes[0].set_title(f"{title_prefix} — Loss");     axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["train_acc"], label="Train")
    axes[1].plot(ep, history["val_acc"],   label="Val")
    axes[1].set_title(f"{title_prefix} — Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=120, bbox_inches="tight")
    plt.show()


def evaluate_and_plot_cm(model, loader, title, save_name):
    """Evaluate a model and generate a confusion matrix plot.

    Args:
        model: Trained model.
        loader: DataLoader for evaluation samples.
        title: Plot title suffix.
        save_name: Output filename for the saved confusion matrix.

    Returns:
        Classification accuracy as a float.
    """

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bf, bl in loader:
            bf  = bf.to(device)
            out = model(bf)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(bl.tolist())
 
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"Test Accuracy ({title}): {acc:.4f}")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1
 
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {title}  (acc={acc:.2%})")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=120, bbox_inches="tight")
    plt.show()
    return acc

# %%
print("TRAINING — Custom CNN")
 
loss_fn_cnn   = nn.CrossEntropyLoss(weight=loss_class_weights)
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=CNN_LR, weight_decay=CNN_WEIGHT_DECAY)
scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_cnn, 
    T_max=CNN_EPOCHS,   
    eta_min=1e-6        
)

cnn_history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
cnn_best_val_acc  = 0.0
cnn_no_improve    = 0
cnn_ckpt_path     = os.path.join(OUTPUT_DIR, "cnn_best_model.pth")
 
t_start = time.time()
for epoch in range(1, CNN_EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(cnn_model, train_loader, loss_fn_cnn, optimizer_cnn, train_mode=True)
    val_loss,   val_acc   = run_epoch(cnn_model, val_loader,   loss_fn_cnn, train_mode=False)
    scheduler_cnn.step()
    current_lr = optimizer_cnn.param_groups[0]['lr']
    
    cnn_history["train_loss"].append(train_loss)
    cnn_history["train_acc"].append(train_acc)
    cnn_history["val_loss"].append(val_loss)
    cnn_history["val_acc"].append(val_acc)
 
    improved = val_acc > cnn_best_val_acc
    tag = "best" if improved else ""
    print(f"Epoch {epoch:3d}/{CNN_EPOCHS} | loss {train_loss:.4f} acc {train_acc:.4f} "
          f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f} | lr {current_lr:.2e} | {time.time()-t0:.1f}s{tag}")
    
    if improved:
        cnn_best_val_acc = val_acc
        cnn_no_improve   = 0
        torch.save({
            "model_state_dict": cnn_model.state_dict(),
            "architecture":     "PatSeCNN",
            "epoch":            epoch,
            "val_acc":          val_acc,
            "num_classes":      NUM_CLASSES,
            "classes":          CLASSES,
            "label_to_idx":     label_to_idx,
            "img_size":         IMG_SIZE,
            "mean":             MEAN,
            "std":              STD,
        }, cnn_ckpt_path)
    else:
        cnn_no_improve += 1
 
    if cnn_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping (CNN): no improvement for {EARLY_STOP_PATIENCE} epochs.")
        break
 
print(f"\nCNN training done in {(time.time()-t_start)/60:.1f} min.  Best val acc: {cnn_best_val_acc:.4f}")

# Training curves
plot_history(cnn_history, "Custom CNN", "training_curves_cnn.png")
 

print("EVALUATION — Custom CNN (best checkpoint)")
 
ckpt = torch.load(cnn_ckpt_path, map_location=device)
cnn_model.load_state_dict(ckpt["model_state_dict"])
 
cnn_test_acc = evaluate_and_plot_cm(
    cnn_model, test_loader,
    title="Custom CNN",
    save_name="confusion_matrix_cnn.png"
)

# %%
print("TRAINING — ResNet50 (pretrained, 2-phase fine-tune)")

def build_resnet50(num_classes: int) -> nn.Module:
    """Create ResNet50 with a custom dropout + linear classifier head.

    Args:
        num_classes: Number of target classes.

    Returns:
        Configured ResNet50 model.
    """

    weights = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=weights)
    in_features = net.fc.in_features          
    net.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes),
    )
    return net
 
 
r50_model   = build_resnet50(NUM_CLASSES).to(device)
loss_fn_r50 = nn.CrossEntropyLoss(weight=loss_class_weights)
 
for param in r50_model.parameters():
    param.requires_grad = False
for param in r50_model.fc.parameters():
    param.requires_grad = True

optimizer_r50 = torch.optim.Adam(
    [p for p in r50_model.parameters() if p.requires_grad],
    lr=R50_HEAD_LR, weight_decay=R50_WEIGHT_DECAY
)

scheduler_r50 = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_r50, T_max=(R50_EPOCHS - R50_HEAD_EPOCHS), eta_min=1e-6
)

n_total     = sum(p.numel() for p in r50_model.parameters())
n_trainable = sum(p.numel() for p in r50_model.parameters() if p.requires_grad)
print(f"ResNet50 — Total params: {n_total:,}  |  Phase-1 trainable: {n_trainable:,}")
 
r50_history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "phase": []}
r50_best_val_acc = 0.0
r50_no_improve   = 0
r50_ckpt_path    = os.path.join(OUTPUT_DIR, "resnet50_best_model.pth")
 
t_start = time.time()

for epoch in range(1, R50_EPOCHS + 1):
    t0 = time.time()
 
    if epoch == R50_HEAD_EPOCHS + 1:
        print(f"\n Phase-2 fine-tune LR={R50_FINETUNE_LR:.1e} \n")
        for param in r50_model.parameters():
            param.requires_grad = True
        optimizer_r50 = torch.optim.Adam(
            r50_model.parameters(), lr=R50_FINETUNE_LR, weight_decay=R50_WEIGHT_DECAY
        )
        scheduler_r50 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_r50, T_max=(R50_EPOCHS - R50_HEAD_EPOCHS), eta_min=1e-6
        )
        r50_no_improve = 0   
 
    phase = "head" if epoch <= R50_HEAD_EPOCHS else "finetune"
    train_loss, train_acc = run_epoch(r50_model, train_loader, loss_fn_r50, optimizer_r50, train_mode=True)
    val_loss,   val_acc   = run_epoch(r50_model, val_loader,   loss_fn_r50, train_mode=False)
    scheduler_r50.step()
    current_lr = optimizer_r50.param_groups[0]['lr']

    r50_history["train_loss"].append(train_loss)
    r50_history["train_acc"].append(train_acc)
    r50_history["val_loss"].append(val_loss)
    r50_history["val_acc"].append(val_acc)
    r50_history["phase"].append(phase)
 
    improved = val_acc > r50_best_val_acc
    tag = "  ← best" if improved else ""
    print(f"Epoch {epoch:3d}/{R50_EPOCHS} [{phase:>8}] | loss {train_loss:.4f} acc {train_acc:.4f} "
          f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f} | lr {current_lr:.2e} | {time.time()-t0:.1f}s{tag}")

    if improved:
        r50_best_val_acc = val_acc
        r50_no_improve   = 0
        torch.save({
            "model_state_dict": r50_model.state_dict(),
            "architecture":     "resnet50",
            "epoch":            epoch,
            "val_acc":          val_acc,
            "num_classes":      NUM_CLASSES,
            "classes":          CLASSES,
            "label_to_idx":     label_to_idx,
            "img_size":         IMG_SIZE,
            "mean":             MEAN,
            "std":              STD,
        }, r50_ckpt_path)
    else:
        r50_no_improve += 1
 
    if r50_no_improve >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stopping (ResNet50): no improvement for {EARLY_STOP_PATIENCE} epochs.")
        break
 
print(f"\nResNet50 training done in {(time.time()-t_start)/60:.1f} min.  Best val acc: {r50_best_val_acc:.4f}")

# Training curves
plot_history(r50_history, "ResNet50 (fine-tuned)", "training_curves_resnet50.png")
 
print("EVALUATION — ResNet50 (best checkpoint)")
 
ckpt = torch.load(r50_ckpt_path, map_location=device)
r50_model.load_state_dict(ckpt["model_state_dict"])
 
r50_test_acc = evaluate_and_plot_cm(
    r50_model, test_loader,
    title="ResNet50",
    save_name="confusion_matrix_resnet50.png"
)

# %%
print("FINAL SUMMARY")

print(f"""
Model           |        Best Val Acc          |     Test Acc
  ─────────────────────────────────────────────────────────────────
  Custom CNN    |  {cnn_best_val_acc:.4f}      |  {cnn_test_acc:.4f}
  ResNet50      |  {r50_best_val_acc:.4f}      |  {r50_test_acc:.4f}

"""
)