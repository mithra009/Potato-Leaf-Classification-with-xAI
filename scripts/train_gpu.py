"""
Potato Leaf Disease Classification - Vision Transformer (ViT) GPU Training
Explainable AI Framework for Disease Detection
Task: Implement ViT with proper 70/20/10 train/val/test split
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
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import warnings
from project_paths import CHECKPOINT_DIR, METRICS_DIR, OUTPUT_DIR, PLOTS_DIR, ensure_project_dirs, resolve_data_dir
warnings.filterwarnings('ignore')

# ==================== DEVICE SETUP ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==================== HYPERPARAMETERS ====================
BATCH_SIZE = 32              # Reduce if CUDA out of memory
LEARNING_RATE = 5e-5         # ViT learning rate
WEIGHT_DECAY = 0.01          # L2 regularization
NUM_EPOCHS = 10            
WARMUP_STEPS = 100           
IMAGE_SIZE = 224             
DATA_DIR = resolve_data_dir()
VIT_MODEL = 'google/vit-base-patch16-224'

# DATA SPLIT: 70% train, 20% validation, 10% test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

class PotatoLeafViTDataset(Dataset):
    """Custom dataset for potato leaf images with ViT preprocessing"""
    
    def __init__(self, root_dir, image_processor, transform=None):
        self.root_dir = Path(root_dir)
        self.image_processor = image_processor
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        print(f"\n[DATASET] Loading images from {root_dir}...")
        print(f"[DATASET] Classes: {self.class_names}")
        
        # Load all images
        for class_name in self.class_names:
            class_path = self.root_dir / class_name
            class_images = []
            for img_path in class_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
                    class_images.append(img_path.name)
            print(f"  {class_name}: {len(class_images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply augmentation if provided
            if self.transform:
                image = self.transform(image)
            
            # Apply ViT image processor
            if isinstance(image, np.ndarray):
                processed = self.image_processor(image, return_tensors='pt')
            else:
                processed = self.image_processor(image, return_tensors='pt')
            
            pixel_values = processed['pixel_values'].squeeze(0)
            return pixel_values, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), label

class ViTPotatoLeafClassifier(nn.Module):
    """Vision Transformer for potato leaf disease classification"""
    
    def __init__(self, num_classes=3, model_name=VIT_MODEL):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
    
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits

def get_train_val_test_loaders(batch_size=BATCH_SIZE):
    """Create data loaders with 70/20/10 split"""
    
    image_processor = ViTImageProcessor.from_pretrained(VIT_MODEL)
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.Lambda(lambda x: np.array(x))
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Lambda(lambda x: np.array(x))
    ])
    
    # Create full dataset
    full_dataset = PotatoLeafViTDataset(DATA_DIR, image_processor, transform=train_transform)
    num_samples = len(full_dataset)
    
    # Create 70/20/10 split
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_split = int(num_samples * TRAIN_RATIO)
    val_split = int(num_samples * (TRAIN_RATIO + VAL_RATIO))
    
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    print(f"\n[SPLIT STATISTICS]")
    print(f"  Total samples: {num_samples}")
    print(f"  Train (70%): {len(train_indices)}")
    print(f"  Validation (20%): {len(val_indices)}")
    print(f"  Test (10%): {len(test_indices)}")
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Change transform for validation and test (no augmentation)
    for idx in val_indices:
        full_dataset.transform = val_transform
    for idx in test_indices:
        full_dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, full_dataset.class_names

def train_epoch(model, train_loader, optimizer, criterion, scheduler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc='Training')
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_labels, all_preds

def plot_training_history(history):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Loss Curve', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_title('Accuracy Curve', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 0].plot(history['learning_rates'], label='Learning Rate', linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=11)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Remove last subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved as '{PLOTS_DIR / 'training_history.png'}'")
    plt.close()

def plot_confusion_matrix(all_labels, all_preds, class_names, filename='confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved as '{PLOTS_DIR / filename}'")
    plt.close()

def main():
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("POTATO LEAF DISEASE CLASSIFICATION - VISION TRANSFORMER (ViT) TRAINING")
    print("="*80)
    
    # Show hyperparameters
    print("\n[HYPERPARAMETERS]")
    print(f"  Model: {VIT_MODEL}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Warmup Steps: {WARMUP_STEPS}")
    print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    
    # Create data loaders with 70/20/10 split
    print("\n[LOADING DATASET WITH 70/20/10 SPLIT]")
    train_loader, val_loader, test_loader, class_names = get_train_val_test_loaders(BATCH_SIZE)
    
    # Initialize model
    print("\n[INITIALIZING MODEL]")
    model = ViTPotatoLeafClassifier(num_classes=len(class_names), model_name=VIT_MODEL)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_STEPS)
    decay_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - (WARMUP_STEPS // len(train_loader))
    )
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], 
                            milestones=[WARMUP_STEPS])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_model_path = CHECKPOINT_DIR / 'best_vit_potato_model.pth'
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-"*80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Validate
        val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Calculate additional metrics
        precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'class_names': class_names,
                'vit_model_name': VIT_MODEL
            }, best_model_path)
            print(f"  ✓ Best model saved (Val Acc: {best_val_acc:.4f})")
    
    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"TRAINING COMPLETED")
    print(f"  Time: {elapsed_time/60:.2f} minutes")
    print(f"  Best validation accuracy: {best_val_acc:.4f}")
    print("="*80)
    
    # Evaluate on test set
    print("\n[TEST SET EVALUATION]")
    model_test = ViTPotatoLeafClassifier(num_classes=len(class_names), model_name=VIT_MODEL)
    checkpoint = torch.load(best_model_path, map_location=device)
    model_test.load_state_dict(checkpoint['model_state_dict'])
    model_test = model_test.to(device)
    
    test_loss, test_acc, test_labels, test_preds = validate(model_test, test_loader, criterion)
    test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=0)
    
    print(f"\n  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    
    # Plot results
    print("\n[GENERATING PLOTS]")
    plot_training_history(history)
    plot_confusion_matrix(val_labels, val_preds, class_names, 'confusion_matrix_validation.png')
    plot_confusion_matrix(test_labels, test_preds, class_names, 'confusion_matrix_test.png')
    
    # Save training summary
    summary = {
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'epochs': NUM_EPOCHS,
            'warmup_steps': WARMUP_STEPS,
            'image_size': IMAGE_SIZE,
            'vit_model': VIT_MODEL
        },
        'dataset': {
            'total_samples': len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset),
            'train_samples': len(train_loader.dataset),
            'val_samples': len(val_loader.dataset),
            'test_samples': len(test_loader.dataset),
            'classes': class_names
        },
        'results': {
            'best_val_accuracy': float(best_val_acc),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(test_f1)
        },
        'training_time_minutes': elapsed_time / 60
    }
    
    with open(METRICS_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\n✓ Training summary saved to '{METRICS_DIR / 'training_summary.json'}'")
    print(f"\nTraining completed successfully!")

if __name__ == '__main__':
    main()
