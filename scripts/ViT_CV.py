"""
The Following code was executed on kaggle not locally. Use Kaggle's T4 gpu to reproduce the result
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ==================== CONFIG ====================
DATA_DIR = Path("/kaggle/input/datasets/rizwan123456789/potato-disease-leaf-datasetpld/PLD_3_Classes_256")

TRAIN_DIR = DATA_DIR / "Training"
VAL_DIR = DATA_DIR / "Validation"
TEST_DIR = DATA_DIR / "Testing"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 7
LR = 5e-5
NUM_FOLDS = 3
IMAGE_SIZE = 224
MODEL_NAME = "google/vit-base-patch16-224"

# ==================== DATASET ====================
class PotatoDataset(Dataset):
    def __init__(self, root_dirs, processor, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.processor = processor

        self.class_names = sorted([d.name for d in root_dirs[0].iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for root_dir in root_dirs:
            for cls in self.class_names:
                for img_path in (root_dir / cls).glob("*"):
                    if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                        self.images.append(str(img_path))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
        label = self.labels[idx]

        return img, label


# ==================== TRANSFORMS ====================
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.Lambda(lambda x: np.array(x))
])

val_tfms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Lambda(lambda x: np.array(x))
])

# ==================== DATA ====================
full_dataset = PotatoDataset(
    root_dirs=[TRAIN_DIR, VAL_DIR],  # combine for CV
    processor=processor,
    transform=train_tfms
)

labels = np.array(full_dataset.labels)

# Test dataset (kept separate)
test_dataset = PotatoDataset(
    root_dirs=[TEST_DIR],
    processor=processor,
    transform=val_tfms
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== MODEL ====================
def get_model():
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        ignore_mismatched_sizes=True
    )
    return model.to(DEVICE)


# ==================== TRAIN ====================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    preds, targets = [], []

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = model(x).logits
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.argmax(out, 1).cpu().numpy())
        targets.extend(y.cpu().numpy())

    return np.mean(losses), accuracy_score(targets, preds)


def validate(model, loader, criterion):
    model.eval()
    losses = []
    preds, targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x).logits
            loss = criterion(out, y)

            losses.append(loss.item())
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    return np.mean(losses), accuracy_score(targets, preds)


# ==================== CROSS VALIDATION ====================
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n===== FOLD {fold+1} =====")

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    fold_results.append(val_acc)

# ==================== FINAL TEST ====================
print("\n===== FINAL TEST EVALUATION =====")
test_model = get_model()
test_model.eval()

preds, targets = [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = test_model(x).logits

        preds.extend(torch.argmax(out, 1).cpu().numpy())
        targets.extend(y.cpu().numpy())

test_acc = accuracy_score(targets, preds)

print(f"\nCV Mean Accuracy: {np.mean(fold_results):.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
