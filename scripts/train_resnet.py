"""
Train ResNet50 for potato leaf disease classification (10 epochs).
Uses 70/20/10 train/val/test split and saves best checkpoint.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from tqdm import tqdm
from project_paths import CHECKPOINT_DIR, METRICS_DIR, OUTPUT_DIR, ensure_project_dirs, resolve_data_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


DATA_DIR = resolve_data_dir()

NUM_EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


class PotatoLeafDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


def get_loaders(batch_size=BATCH_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base_ds = PotatoLeafDataset(DATA_DIR, transform=None)
    n = len(base_ds)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)

    n_train = int(TRAIN_RATIO * n)
    n_val = int(VAL_RATIO * n)
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    train_ds = PotatoLeafDataset(DATA_DIR, transform=train_tf)
    val_ds = PotatoLeafDataset(DATA_DIR, transform=eval_tf)
    test_ds = PotatoLeafDataset(DATA_DIR, transform=eval_tf)

    train_subset = Subset(train_ds, train_idx.tolist())
    val_subset = Subset(val_ds, val_idx.tolist())
    test_subset = Subset(test_ds, test_idx.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, base_ds.class_names


def build_model(num_classes: int):
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, criterion):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return float(np.mean(losses)), acc, prec, rec, f1


def train():
    ensure_project_dirs()
    print("\n" + "=" * 80)
    print("RESNET50 TRAINING (10 EPOCHS)")
    print("=" * 80)

    train_loader, val_loader, test_loader, class_names = get_loaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    print(f"Classes: {class_names}")

    model = build_model(len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_acc = 0.0
    best_path = CHECKPOINT_DIR / "best_resnet_potato_model.pth"

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_labels.extend(y.detach().cpu().numpy().tolist())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_labels, train_preds)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Val Prec:   {val_prec:.4f} | Val Rec:   {val_rec:.4f} | Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "model_name": "resnet50",
                },
                best_path,
            )
            print(f"  Saved best model -> {best_path}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion)

    summary = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1": test_f1,
        "epochs": NUM_EPOCHS,
        "class_names": class_names,
    }

    with open(METRICS_DIR / "resnet_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best checkpoint: {best_path}")
    print(f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    train()
