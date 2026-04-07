"""
XAI for ResNet50 potato leaf classifier.
Generates Grad-CAM and CNN rollout maps for one image from each class.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from project_paths import CHECKPOINT_DIR, OUTPUT_DIR, XAI_DIR, ensure_project_dirs, resolve_data_dir


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        self.handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, x, target_class=None):
        self._register_hooks()
        logits = self.model(x)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        self._remove_hooks()
        return cam


class CNNRollout:
    """Simple CNN rollout by aggregating normalized activations from residual stages."""

    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.handles = []

    def _hook(self, module, inp, out):
        self.feature_maps.append(out.detach())

    def _register(self):
        backbone = self.model.model
        for layer in [backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4]:
            self.handles.append(layer.register_forward_hook(self._hook))

    def _clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, x):
        self.feature_maps = []
        self._register()
        with torch.no_grad():
            _ = self.model(x)
        self._clear()

        maps = []
        for fmap in self.feature_maps:
            m = fmap.mean(dim=1).squeeze(0).cpu().numpy()
            m = np.maximum(m, 0)
            m = cv2.resize(m, (224, 224))
            m = (m - m.min()) / (m.max() - m.min() + 1e-8)
            maps.append(m)

        rollout = np.ones((224, 224), dtype=np.float32)
        for m in maps:
            rollout *= m
        rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)
        return rollout


def blend_heatmap(image_np, heatmap, alpha=0.45):
    image_np = image_np.astype(np.uint8)
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    model = ResNetClassifier(num_classes=len(class_names))

    state_dict = checkpoint["model_state_dict"]
    normalized_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]
        normalized_state_dict[new_key] = value

    model.model.load_state_dict(normalized_state_dict, strict=False)
    model = model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image):
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image).unsqueeze(0).to(device)


def pick_one_image_per_class(class_names):
    selected = []
    root = resolve_data_dir()
    for class_name in class_names:
        class_dir = root / class_name
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            files.extend(list(class_dir.glob(ext)))
        if files:
            selected.append(files[0])
    return selected


def run_xai():
    checkpoint_path = CHECKPOINT_DIR / "best_resnet_potato_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, class_names = load_model(checkpoint_path)

    gradcam = GradCAM(model, model.model.layer4[-1].conv3)
    rollout = CNNRollout(model)

    ensure_project_dirs()
    out_dir = XAI_DIR / "resnet"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = pick_one_image_per_class(class_names)
    if not images:
        raise RuntimeError("No images found for class-wise XAI.")

    summary_fig, axes = plt.subplots(len(images), 3, figsize=(13, 4 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB").resize((224, 224))
        image_np = np.array(image)
        x = preprocess_image(image)

        with torch.no_grad():
            logits = model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        cam = gradcam.generate(x, target_class=pred_idx)
        roll = rollout.generate(x)

        cam_overlay = blend_heatmap(image_np, cam)
        roll_overlay = blend_heatmap(image_np, roll)

        class_title = f"True: {img_path.parent.name} | Pred: {class_names[pred_idx]} ({probs[pred_idx]:.1%})"

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].imshow(image_np)
        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[1].imshow(cam_overlay)
        ax[1].set_title("Grad-CAM")
        ax[1].axis("off")
        ax[2].imshow(roll_overlay)
        ax[2].set_title("CNN Rollout")
        ax[2].axis("off")
        fig.suptitle(class_title, fontsize=11, fontweight="bold")
        fig.tight_layout()

        save_path = out_dir / f"xai_{img_path.parent.name}.png"
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        axes[r, 0].imshow(image_np)
        axes[r, 0].set_title(f"Original\n{img_path.parent.name}")
        axes[r, 0].axis("off")
        axes[r, 1].imshow(cam_overlay)
        axes[r, 1].set_title("Grad-CAM")
        axes[r, 1].axis("off")
        axes[r, 2].imshow(roll_overlay)
        axes[r, 2].set_title("CNN Rollout")
        axes[r, 2].axis("off")

        print(f"Saved: {save_path}")

    summary_fig.suptitle("ResNet XAI: One Image Per Class", fontsize=14, fontweight="bold")
    summary_fig.tight_layout()
    summary_path = out_dir / "xai_resnet_summary.png"
    summary_fig.savefig(summary_path, dpi=180, bbox_inches="tight")
    plt.close(summary_fig)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    run_xai()
