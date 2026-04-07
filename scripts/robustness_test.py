"""
Robustness Testing Script
Evaluates model robustness under various augmentations (noise, blur, brightness, etc.)
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification
import warnings
from project_paths import OUTPUT_DIR, ROBUSTNESS_DIR, ensure_project_dirs, resolve_data_dir
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ViTPotatoLeafClassifier(nn.Module):
    def __init__(self, num_classes=3, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )
    
    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values).logits

class RobustnessEvaluator:
    """Evaluates model robustness under various perturbations"""
    
    def __init__(self, model, image_processor, device, class_names=None):
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.class_names = class_names or []
        self.class_to_idx = {name.lower(): idx for idx, name in enumerate(self.class_names)}
        self.model.eval()
    
    @staticmethod
    def add_gaussian_noise(image, std=0.1):
        """Add Gaussian noise"""
        image_np = np.array(image)
        noise = np.random.normal(0, std, image_np.shape)
        noisy_image = np.clip(image_np + noise * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_image)
    
    @staticmethod
    def add_salt_pepper_noise(image, prob=0.01):
        """Add salt and pepper noise"""
        image_np = np.array(image, dtype=np.float32) / 255.0
        mask = np.random.random(image_np.shape)
        
        # Salt
        image_np[mask < prob/2] = 1.0
        # Pepper
        image_np[mask > 1 - prob/2] = 0.0
        
        return Image.fromarray((image_np * 255).astype(np.uint8))
    
    @staticmethod
    def apply_blur(image, kernel_size=5):
        """Apply Gaussian blur"""
        image_np = np.array(image)
        blurred = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), 0)
        return Image.fromarray(blurred)
    
    @staticmethod
    def apply_motion_blur(image, kernel_size=15):
        """Apply motion blur"""
        image_np = np.array(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        kernel = kernel / kernel.sum()
        blurred = cv2.filter2D(image_np, -1, kernel)
        return Image.fromarray(blurred.astype(np.uint8))
    
    @staticmethod
    def adjust_brightness(image, factor=0.5):
        """Adjust brightness"""
        image_np = np.array(image, dtype=np.float32)
        adjusted = np.clip(image_np * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(adjusted)
    
    @staticmethod
    def adjust_contrast(image, factor=0.5):
        """Adjust contrast"""
        image_np = np.array(image, dtype=np.float32)
        mean = image_np.mean()
        adjusted = np.clip((image_np - mean) * factor + mean, 0, 255).astype(np.uint8)
        return Image.fromarray(adjusted)
    
    @staticmethod
    def apply_compression(image, quality=10):
        """Apply JPEG compression"""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert('RGB')
    
    def predict_image(self, image):
        """Get prediction for an image"""
        try:
            processed = self.image_processor(image, return_tensors='pt')
            pixel_values = processed['pixel_values'].to(self.device)
            
            with torch.no_grad():
                logits = self.model(pixel_values)
                pred = torch.argmax(logits, dim=1).item()
                confidence = torch.softmax(logits, dim=1)[0, pred].item()
            
            return pred, confidence
        except Exception as e:
            print(f"Error predicting: {e}")
            return None, None
    
    def evaluate_robustness(self, image_paths, augmentations):
        """Evaluate model robustness"""
        results = {}
        
        for aug_name, aug_func in augmentations.items():
            print(f"\n  Evaluating: {aug_name}...")
            correct = 0
            total = 0
            
            for img_path in image_paths:
                true_label = self._get_label(img_path)
                
                # Apply augmentation
                image = Image.open(img_path).convert('RGB')
                aug_image = aug_func(image)
                
                # Predict
                pred, _ = self.predict_image(aug_image)
                
                if pred == true_label:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            results[aug_name] = accuracy
            print(f"    Accuracy: {accuracy:.4f}")
        
        return results
    
    def _get_label(self, img_path):
        """Extract label from path"""
        parent_name = Path(img_path).parent.name.lower()

        # Prefer exact class-name mapping from training checkpoint.
        if parent_name in self.class_to_idx:
            return self.class_to_idx[parent_name]

        # Fallback matching if naming differs slightly.
        for class_name, idx in self.class_to_idx.items():
            if class_name in parent_name or parent_name in class_name:
                return idx

        # Legacy fallback for common Potato leaf naming variants.
        legacy_map = {
            'potato___early_blight': 0,
            'potato___late_blight': 1,
            'potato___healthy': 2,
            'early_blight': 0,
            'late_blight': 1,
            'healthy': 2,
        }
        if parent_name in legacy_map:
            return legacy_map[parent_name]

        return 0

def collect_test_images():
    """Collect test images from PlantVillage"""
    image_paths = []
    root = resolve_data_dir()
    
    for class_dir in root.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))[:10]  # Get up to 10 per class
            image_paths.extend(images)
    
    return image_paths

def create_augmentation_suite():
    """Create suite of augmentations for robustness testing"""
    return {
        'Original': lambda x: x,
        'Gaussian Noise (0.1)': lambda x: RobustnessEvaluator.add_gaussian_noise(x, std=0.1),
        'Gaussian Noise (0.2)': lambda x: RobustnessEvaluator.add_gaussian_noise(x, std=0.2),
        'Salt & Pepper': lambda x: RobustnessEvaluator.add_salt_pepper_noise(x, prob=0.02),
        'Gaussian Blur (5)': lambda x: RobustnessEvaluator.apply_blur(x, kernel_size=5),
        'Gaussian Blur (9)': lambda x: RobustnessEvaluator.apply_blur(x, kernel_size=9),
        'Motion Blur': lambda x: RobustnessEvaluator.apply_motion_blur(x, kernel_size=15),
        'Brightness (0.5)': lambda x: RobustnessEvaluator.adjust_brightness(x, factor=0.5),
        'Brightness (1.5)': lambda x: RobustnessEvaluator.adjust_brightness(x, factor=1.5),
        'Contrast (0.5)': lambda x: RobustnessEvaluator.adjust_contrast(x, factor=0.5),
        'Contrast (1.5)': lambda x: RobustnessEvaluator.adjust_contrast(x, factor=1.5),
        'JPEG Compression (10)': lambda x: RobustnessEvaluator.apply_compression(x, quality=10),
        'JPEG Compression (30)': lambda x: RobustnessEvaluator.apply_compression(x, quality=30),
    }

def plot_robustness_results(results):
    """Plot robustness test results"""
    augmentations = list(results.keys())
    accuracies = list(results.values())
    
    # Sort by accuracy
    sorted_pairs = sorted(zip(augmentations, accuracies), key=lambda x: x[1], reverse=True)
    augmentations, accuracies = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.bar(range(len(augmentations)), accuracies, color='steelblue', alpha=0.8, edgecolor='navy')
    
    # Color code: green for good, yellow for medium, red for poor
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        if acc >= 0.8:
            bar.set_color('green')
        elif acc >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    ax.set_xlabel('Augmentation Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Model Robustness under Various Augmentations', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(augmentations)))
    ax.set_xticklabels(augmentations, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=9)
    
    # Add horizontal line for baseline
    baseline = results['Original']
    ax.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline:.3f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_DIR / 'robustness_results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {ROBUSTNESS_DIR / 'robustness_results.png'}")
    plt.close()

def plot_robustness_heatmap(results):
    """Plot robustness as heatmap"""
    import seaborn as sns
    
    # Organize results into categories
    augmentation_types = {
        'Noise': ['Gaussian Noise (0.1)', 'Gaussian Noise (0.2)', 'Salt & Pepper'],
        'Blur': ['Gaussian Blur (5)', 'Gaussian Blur (9)', 'Motion Blur'],
        'Brightness': ['Brightness (0.5)', 'Brightness (1.5)'],
        'Contrast': ['Contrast (0.5)', 'Contrast (1.5)'],
        'Compression': ['JPEG Compression (10)', 'JPEG Compression (30)'],
        'Baseline': ['Original']
    }
    
    data = []
    labels = []
    
    for category, augs in augmentation_types.items():
        category_data = []
        for aug in augs:
            if aug in results:
                category_data.append(results[aug])
        if category_data:
            data.append(category_data)
            labels.append(category)
    
    if not data:
        print("[WARNING] No robustness data available for heatmap.")
        return

    max_len = max(len(row) for row in data)
    padded_data = [row + [np.nan] * (max_len - len(row)) for row in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(padded_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                yticklabels=labels, xticklabels=[f'Variation {i+1}' for i in range(max_len)],
                cbar_kws={'label': 'Accuracy'}, ax=ax, linewidths=0.5)
    
    ax.set_title('Model Robustness Heatmap by Perturbation Category', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(ROBUSTNESS_DIR / 'robustness_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {ROBUSTNESS_DIR / 'robustness_heatmap.png'}")
    plt.close()

def main():
    ensure_project_dirs()
    
    print("\n" + "="*80)
    print("ROBUSTNESS TESTING - VISION TRANSFORMER POTATO LEAF CLASSIFIER")
    print("="*80)
    
    # Load model
    print("\n[LOADING MODEL]")
    checkpoint_path = OUTPUT_DIR / 'checkpoints' / 'best_vit_potato_model.pth'
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"Error: {checkpoint_path} not found")
        return
    
    class_names = checkpoint['class_names']
    vit_model_name = checkpoint.get('vit_model_name', 'google/vit-base-patch16-224')
    
    model = ViTPotatoLeafClassifier(num_classes=len(class_names), model_name=vit_model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded: {vit_model_name}")
    print(f"  Classes: {class_names}")
    
    # Collect test images
    print("\n[COLLECTING TEST IMAGES]")
    image_paths = collect_test_images()
    print(f"  Test images: {len(image_paths)}")
    
    # Create evaluator
    image_processor = ViTImageProcessor.from_pretrained(vit_model_name)
    evaluator = RobustnessEvaluator(model, image_processor, device, class_names=class_names)
    
    # Run robustness evaluation
    print("\n[EVALUATING ROBUSTNESS]")
    augmentations = create_augmentation_suite()
    results = evaluator.evaluate_robustness(image_paths, augmentations)
    
    # Print summary
    print("\n" + "="*80)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("="*80)
    print(f"\n{'Augmentation Type':<30} {'Accuracy':<15}")
    print("-"*80)
    
    for aug_name in sorted(results.keys(), key=lambda x: results[x], reverse=True):
        print(f"{aug_name:<30} {results[aug_name]:<15.4f}")
    
    # Calculate robustness metrics
    baseline = results['Original']
    worst_case = min(results.values())
    average_accuracy = np.mean(list(results.values()))
    robustness_score = (average_accuracy - worst_case) / baseline if baseline > 0 else 0
    
    print("\n[ROBUSTNESS METRICS]")
    print(f"  Baseline Accuracy: {baseline:.4f}")
    print(f"  Average Accuracy: {average_accuracy:.4f}")
    print(f"  Worst Case Accuracy: {worst_case:.4f}")
    print(f"  Robustness Score: {robustness_score:.4f}")
    print("="*80)
    
    # Plot results
    print("\n[GENERATING VISUALIZATIONS]")
    plot_robustness_results(results)
    plot_robustness_heatmap(results)
    
    print("\n✓ Robustness testing completed!")

if __name__ == '__main__':
    main()
