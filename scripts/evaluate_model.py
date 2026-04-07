"""Comprehensive evaluation for the potato leaf ViT classifier."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, ViTImageProcessor

from project_paths import CHECKPOINT_DIR, METRICS_DIR, OUTPUT_DIR, PLOTS_DIR, ensure_project_dirs, resolve_data_dir

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PotatoLeafViTDataset(Dataset):
    """Dataset wrapper for ViT evaluation."""

    def __init__(self, root_dir: Path, image_processor: ViTImageProcessor):
        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        processed = self.image_processor(image, return_tensors="pt")
        pixel_values = processed["pixel_values"].squeeze(0)
        return pixel_values, self.labels[idx]


class ViTPotatoLeafClassifier(nn.Module):
    """ViT classification wrapper used for loading the trained checkpoint."""

    def __init__(self, num_classes: int, model_name: str):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits


def load_model(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("vit_model_name", "google/vit-base-patch16-224")

    model = ViTPotatoLeafClassifier(num_classes=len(class_names), model_name=model_name)

    state_dict = checkpoint["model_state_dict"]
    normalized_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("vit."):
            new_key = new_key[len("vit."):]
        normalized_state_dict[new_key] = value

    model.load_state_dict(normalized_state_dict, strict=False)
    model = model.to(DEVICE)
    model.eval()
    return model, model_name, class_names


@torch.no_grad()
def evaluate_model(model, loader):
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(all_labels, all_preds, class_names, filename: str):
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted Label", fontweight="bold")
    plt.ylabel("True Label", fontweight="bold")
    plt.title("Confusion Matrix - Test Set", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curves(all_labels, all_probs, class_names):
    from sklearn.preprocessing import label_binarize

    labels_bin = label_binarize(all_labels, classes=range(len(class_names)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax1.set_title("ROC Curves - One-vs-Rest", fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(fontsize=9, loc="lower right")
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), all_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax2.plot(fpr_micro, tpr_micro, color="deeppink", linewidth=2, label=f"Micro-average (AUC={roc_auc_micro:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax2.set_title("Micro-Averaged ROC", fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(all_labels, all_preds, class_names):
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", alpha=0.85)
    ax.bar(x, recall, width, label="Recall", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1-Score", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Per-Class Metrics Comparison", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_report(all_labels, all_preds, class_names, all_probs):
    report = {
        "overall_metrics": {
            "accuracy": float(accuracy_score(all_labels, all_preds)),
            "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
            "precision_weighted": float(precision_score(all_labels, all_preds, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(all_labels, all_preds, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        },
        "per_class_metrics": {},
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0),
    }

    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    for idx, class_name in enumerate(class_names):
        report["per_class_metrics"][class_name] = {
            "precision": float(precision_per_class[idx]),
            "recall": float(recall_per_class[idx]),
            "f1_score": float(f1_per_class[idx]),
        }

    report["confusion_matrix"] = confusion_matrix(all_labels, all_preds).tolist()
    report["roc_auc_macro_ovr"] = float(
        roc_auc_score(np.eye(len(class_names))[all_labels], all_probs, average="macro", multi_class="ovr")
    )

    output_file = METRICS_DIR / "evaluation_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: {output_file}")
    return report


def print_summary(report, class_names):
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY - TEST SET")
    print("=" * 80)
    print(f"\nAccuracy: {report['overall_metrics']['accuracy']:.4f}")
    print(f"Macro Precision: {report['overall_metrics']['precision_macro']:.4f}")
    print(f"Macro Recall: {report['overall_metrics']['recall_macro']:.4f}")
    print(f"Macro F1: {report['overall_metrics']['f1_macro']:.4f}")
    print(f"ROC-AUC Macro OVR: {report['roc_auc_macro_ovr']:.4f}")
    print("\nPer-class metrics:")
    for class_name in class_names:
        metrics = report["per_class_metrics"][class_name]
        print(f"  {class_name}: P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1_score']:.4f}")
    print("=" * 80)


def main():
    ensure_project_dirs()

    checkpoint_path = CHECKPOINT_DIR / "best_vit_potato_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION - POTATO LEAF ViT CLASSIFIER")
    print("=" * 80)
    print(f"\nLoading checkpoint: {checkpoint_path}")

    model, model_name, class_names = load_model(checkpoint_path)
    print(f"Model: {model_name}")
    print(f"Classes: {class_names}")

    image_processor = ViTImageProcessor.from_pretrained(model_name)
    test_dataset = PotatoLeafViTDataset(resolve_data_dir(), image_processor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    all_labels, all_preds, all_probs = evaluate_model(model, test_loader)
    report = generate_report(all_labels, all_preds, class_names, all_probs)
    print_summary(report, class_names)

    plot_confusion_matrix(all_labels, all_preds, class_names, "confusion_matrix_eval.png")
    plot_metrics_comparison(all_labels, all_preds, class_names)
    plot_roc_curves(all_labels, all_probs, class_names)

    print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
