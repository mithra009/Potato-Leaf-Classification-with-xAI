"""
Inference script to test the trained Vision Transformer potato leaf disease classifier
"""

import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from project_paths import CHECKPOINT_DIR, INFERENCE_DIR, ensure_project_dirs, resolve_data_dir

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ViTPotatoLeafClassifier(nn.Module):
    """Vision Transformer for potato leaf disease classification"""
    
    def __init__(self, num_classes=3, model_name='google/vit-base-patch16-224'):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

def load_model(checkpoint_path, num_classes=3):
    """Load trained Vision Transformer from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = checkpoint.get('vit_model_name', 'google/vit-base-patch16-224')
    
    model = ViTPotatoLeafClassifier(num_classes=num_classes, model_name=model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    class_names = checkpoint.get('class_names', ['Early_blight', 'Healthy', 'Late_blight'])
    return model, model_name, class_names

def predict(image_path, model, image_processor, class_names):
    """Predict class for a single image"""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocess with ViT image processor
    processed = image_processor(image, return_tensors='pt')
    pixel_values = processed['pixel_values'].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(pixel_values)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return class_names[predicted_class], confidence, image, probabilities[0].cpu().numpy()

def main():
    ensure_project_dirs()
    # Load model
    candidate_paths = [
        CHECKPOINT_DIR / 'best_vit_potato_model.pth',
        Path('best_vit_potato_model.pth')
    ]
    model_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    print(f"Loading model from {model_path}...")
    
    try:
        model, model_name, class_names = load_model(str(model_path))
        image_processor = ViTImageProcessor.from_pretrained(model_name)
        print(f"✓ Model loaded successfully!")
        print(f"  Model: {model_name}")
        print(f"  Classes: {class_names}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please train the model first.")
        return
    
    # Test on sample images
    data_root = resolve_data_dir()
    test_dirs = [
        data_root / "Potato___Early_blight",
        data_root / "Potato___healthy",
        data_root / "Potato___Late_blight"
    ]
    
    all_images = []
    for test_dir in test_dirs:
        if Path(test_dir).exists():
            image_files = list(Path(test_dir).glob('*.jpg'))
            if not image_files:
                image_files = list(Path(test_dir).glob('*.png'))
            all_images.extend(image_files[:1])  # Get 1 from each class
    
    if not all_images:
        print(f"No images found in {test_dirs}")
        return
    
    # Test on first 3 images
    num_test = min(3, len(all_images))
    fig, axes = plt.subplots(1, num_test, figsize=(15, 4))
    if num_test == 1:
        axes = [axes]
    
    print(f"\n[INFERENCE RESULTS]")
    for idx, img_path in enumerate(all_images[:num_test]):
        predicted_class, confidence, image, probabilities = predict(str(img_path), model, image_processor, class_names)
        
        print(f"\nImage: {img_path.name}")
        print(f"  Predicted: {predicted_class} (Confidence: {confidence:.2%})")
        for cls, prob in zip(class_names, probabilities):
            print(f"    {cls}: {prob:.2%}")
        
        # Plot
        axes[idx].imshow(image)
        axes[idx].set_title(f"{predicted_class}\n({confidence:.1%})", fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(INFERENCE_DIR / 'inference_results.png', dpi=150)
    print(f"\n✓ Results saved to '{INFERENCE_DIR / 'inference_results.png'}'")
    plt.close()

if __name__ == '__main__':
    main()
