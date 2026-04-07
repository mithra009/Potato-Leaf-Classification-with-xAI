"""
Explainable AI (XAI) Module for Vision Transformer
Implements Grad-CAM and Attention Rollout for interpretability
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification
import warnings
from project_paths import CHECKPOINT_DIR, OUTPUT_DIR, XAI_DIR, ensure_project_dirs, resolve_data_dir
warnings.filterwarnings('ignore')

class GradCAM:
    """Gradient-weighted Class Activation Mapping for ViT"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: The ViT model
            target_layer: Layer to compute gradients for
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
    def hook_backward(self, module, grad_input, grad_output):
        grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        self.gradients = grad.detach() if hasattr(grad, 'detach') else grad
    
    def hook_forward(self, module, input, output):
        act = output[0] if isinstance(output, (tuple, list)) else output
        self.activations = act.detach() if hasattr(act, 'detach') else act
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap
        Args:
            input_tensor: Input image tensor
            target_class: Target class for which to generate CAM
        """
        # Register hooks
        h_forward = self.target_layer.register_forward_hook(self.hook_forward)
        h_backward = self.target_layer.register_full_backward_hook(self.hook_backward)
        self.handles = [h_forward, h_backward]
        
        # Forward pass
        outputs = self.model(input_tensor)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        logits[0, target_class].backward()
        
        # Compute CAM
        gradients = self.gradients[0].mean(dim=0, keepdim=True)
        activations = self.activations[0]
        cam = (gradients * activations).sum(dim=1)
        cam = cam.squeeze().cpu().detach().numpy()
        
        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Remove hooks
        for h in self.handles:
            h.remove()
        
        return cam, target_class

class AttentionRollout:
    """Attention Rollout for ViT - visualize which patches attend to predictions"""
    
    def __init__(self, model, discard_ratio=0.9):
        """
        Args:
            model: The ViT model
            discard_ratio: Discard lowest attention values
        """
        self.model = model
        self.discard_ratio = discard_ratio
        
    def generate(self, input_tensor):
        """
        Generate attention rollout heatmap
        Args:
            input_tensor: Input image tensor (1, 3, 224, 224)
        """
        output = self.model(input_tensor, output_attentions=True)
        attentions = output.attentions
        
        # Process attention heads
        attention_maps = {}
        for i, attention in enumerate(attentions):
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            attention = attention.detach().cpu()
            
            # Average over heads
            attention = attention.mean(dim=1)  # (batch_size, seq_len, seq_len)
            
            # Use cls token attention (first row)
            attention = attention[0, 0, 1:]  # Exclude cls token self-attention
            
            # Discard low attention values
            threshold = np.percentile(attention.numpy(), int(100 * self.discard_ratio))
            attention[attention < threshold] = 0
            
            attention_maps[f'layer_{i}'] = attention
        
        # Simple rollout: normalize and average
        rollout = torch.ones(196)  # 14x14 patches
        for attention in attention_maps.values():
            if len(attention) == 196:
                rollout = rollout * attention
        
        rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min() + 1e-8)
        
        # Reshape to image grid
        h = w = int(np.sqrt(len(rollout)))
        rollout_img = rollout.view(h, w).numpy()
        rollout_img = cv2.resize(rollout_img, (224, 224))
        
        return rollout_img, output.logits.argmax(dim=1).item()

def blend_heatmap(image_np, heatmap, alpha=0.5):
    """Blend heatmap with original image"""
    if isinstance(image_np, torch.Tensor):
        image_np = image_np.cpu().numpy()
    
    # Normalize image to 0-255
    if image_np.max() <= 1:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Ensure proper shape
    if image_np.shape[0] == 3:
        image_np = np.transpose(image_np, (1, 2, 0))

    # Ensure 3 channels and same spatial size as heatmap
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    target_h, target_w = heatmap.shape[:2]
    if image_np.shape[0] != target_h or image_np.shape[1] != target_w:
        image_np = cv2.resize(image_np, (target_w, target_h))
    
    # Convert heatmap to color
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend
    blended = cv2.addWeighted(image_np, 1 - alpha, heatmap_color, alpha, 0)
    
    return blended

def visualize_xai(image_path, model, image_processor, class_names, device):
    """
    Visualize both Grad-CAM and Attention Rollout for an image
    """
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    processed = image_processor(image, return_tensors='pt')
    input_tensor = processed['pixel_values'].to(device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        prediction = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, prediction].item()
    
    # Grad-CAM
    gradcam = GradCAM(model, model.vit.encoder.layer[-1].attention.attention)
    try:
        gradcam_map, _ = gradcam.generate(input_tensor, target_class=prediction)
    except:
        gradcam_map = None
    
    # Attention Rollout
    rollout = AttentionRollout(model)
    rollout_map, _ = rollout.generate(input_tensor)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grad-CAM heatmap
    if gradcam_map is not None:
        axes[0, 1].imshow(gradcam_map, cmap='hot')
        axes[0, 1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Grad-CAM blended
        blended_gradcam = blend_heatmap(image_np, gradcam_map, alpha=0.4)
        axes[0, 2].imshow(blended_gradcam)
        axes[0, 2].set_title('Grad-CAM Overlay', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
    else:
        axes[0, 1].text(0.5, 0.5, 'Grad-CAM not available', ha='center', va='center')
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
    
    # Attention Rollout heatmap
    axes[1, 0].imshow(rollout_map, cmap='hot')
    axes[1, 0].set_title('Attention Rollout Heatmap', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Attention Rollout blended
    blended_rollout = blend_heatmap(image_np, rollout_map, alpha=0.4)
    axes[1, 1].imshow(blended_rollout)
    axes[1, 1].set_title('Attention Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Prediction info
    axes[1, 2].axis('off')
    info_text = f"""
    Prediction: {class_names[prediction]}
    Confidence: {confidence:.2%}
    
    XAI Interpretation:
    • Grad-CAM shows gradients
      flowing to target class
    
    • Attention Rollout shows
      which image patches the
      model focuses on
    
    • Bright regions indicate
      important areas for
      classification
    """
    axes[1, 2].text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Explainable AI Analysis - ViT Potato Leaf Disease Detection', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

class XAIEvaluator:
    """Batch XAI evaluation on multiple images"""
    
    def __init__(self, model, image_processor, class_names, device):
        self.model = model
        self.image_processor = image_processor
        self.class_names = class_names
        self.device = device
    
    def evaluate_directory(self, image_dir, output_dir=XAI_DIR / 'vit', num_images=5):
        """
        Evaluate XAI on multiple images from a directory
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        image_files = list(Path(image_dir).glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        image_files = image_files[:min(num_images, len(image_files))]
        
        print(f"\n[XAI EVALUATION] Analyzing {len(image_files)} images from {image_dir}")
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"  Processing {idx}/{len(image_files)}: {img_path.name}")
            
            try:
                fig = visualize_xai(str(img_path), self.model, self.image_processor, 
                                   self.class_names, self.device)
                
                output_file = Path(output_dir) / f"xai_{img_path.stem}.png"
                fig.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
        
        print(f"✓ XAI analysis saved to {output_dir}")

def main():
    # Example usage
    import torch
    from transformers import ViTImageProcessor, ViTForImageClassification
    ensure_project_dirs()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model_name = 'google/vit-base-patch16-224'
    image_processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    checkpoint_path = CHECKPOINT_DIR / 'best_vit_potato_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint

        # Handle checkpoints saved from wrapper/DataParallel models.
        normalized_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if new_key.startswith('module.'):
                new_key = new_key[len('module.'):]
            if new_key.startswith('vit.vit.'):
                new_key = new_key[len('vit.'):]
            elif new_key.startswith('vit.classifier.'):
                new_key = new_key[len('vit.'):]
            normalized_state_dict[new_key] = value

        model.load_state_dict(normalized_state_dict, strict=False)
        print(f"✓ Loaded trained checkpoint: {checkpoint_path}")
    else:
        print("[WARNING] Trained checkpoint not found. Using base pretrained ViT weights.")

    model = model.to(device)
    model.eval()
    
    class_names = ['Early Blight', 'Healthy', 'Late Blight']
    
    # Test on one image
    test_image = resolve_data_dir() / "Potato___Early_blight"
    image_files = list(Path(test_image).glob('*.jpg'))
    
    if image_files:
        fig = visualize_xai(str(image_files[0]), model, image_processor, class_names, device)
        (XAI_DIR / 'vit').mkdir(parents=True, exist_ok=True)
        fig.savefig(XAI_DIR / 'vit' / 'xai_example.png', dpi=150, bbox_inches='tight')
        print("✓ XAI visualization saved")
    
if __name__ == '__main__':
    main()
