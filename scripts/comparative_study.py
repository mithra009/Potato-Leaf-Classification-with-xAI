"""
Comparative Study: Vision Transformer vs CNN Models
Compares ViT with ResNet, VGG, and MobileNet on potato leaf classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTImageProcessor, ViTForImageClassification
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json
import time
from project_paths import COMPARISON_DIR, METRICS_DIR, OUTPUT_DIR, ensure_project_dirs, resolve_data_dir
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PotatoLeafDataset(Dataset):
    """Unified dataset for all models"""
    
    def __init__(self, root_dir, transform=None, use_vit=False, image_processor=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_vit = use_vit
        self.image_processor = image_processor
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        for class_name in self.class_names:
            class_path = self.root_dir / class_name
            for img_path in class_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.use_vit:
                processed = self.image_processor(image, return_tensors='pt')
                pixel_values = processed['pixel_values'].squeeze(0)
                return pixel_values, label
            else:
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
                return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            if self.use_vit:
                return torch.zeros(3, 224, 224), label
            else:
                return torch.zeros(3, 224, 224), label

# ==================== MODEL DEFINITIONS ====================

class ViTModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224', num_labels=num_classes, ignore_mismatched_sizes=True
        )
    
    def forward(self, x):
        return self.vit(x).logits

class ResNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.model(x)

class VGGModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.model.classifier[6] = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        return self.model(x)

class MobileNetModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.model(x)

# ==================== TRAINING AND EVALUATION ====================

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """Train a single model"""
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    training_history = {'train_loss': [], 'val_acc': [], 'train_time': []}
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training
        model.train()
        running_loss = 0.0
        train_preds = []
        train_labels = []
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        training_history['train_loss'].append(train_loss)
        training_history['val_acc'].append(val_acc)
        training_history['train_time'].append(time.time() - epoch_start)
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
    
    total_time = time.time() - start_time
    training_history['total_time'] = total_time
    
    return training_history

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def get_model_size(model):
    """Get model parameter count"""
    return sum(p.numel() for p in model.parameters())

def run_comparative_study(num_epochs=10):
    """Run comparative study"""
    Path('outputs').mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print("COMPARATIVE STUDY: VISION TRANSFORMER vs CNN MODELS")
    print("="*80)
    
    # Data loading setup
    print("\n[LOADING DATASET]")
    
    # Create datasets for each model type
    transform_cnn = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_cnn = PotatoLeafDataset(resolve_data_dir(), transform=transform_cnn, use_vit=False)
    
    image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    dataset_vit = PotatoLeafDataset(resolve_data_dir(), use_vit=True, image_processor=image_processor)
    
    class_names = dataset_cnn.class_names
    num_samples = len(dataset_cnn)
    
    # 70/20/10 split
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_split = int(num_samples * 0.7)
    val_split = int(num_samples * 0.9)
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split:val_split]
    test_idx = indices[val_split:]
    
    print(f"  Total: {num_samples} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    
    # Models to compare
    models_config = {
        'ViT': {
            'class': ViTModel,
            'dataset': dataset_vit,
            'lr': 5e-5,
            'batch_size': 16,
            'epochs': num_epochs
        },
        'ResNet50': {
            'class': ResNetModel,
            'dataset': dataset_cnn,
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': num_epochs
        },
        'VGG16': {
            'class': VGGModel,
            'dataset': dataset_cnn,
            'lr': 1e-4,
            'batch_size': 32,
            'epochs': num_epochs
        },
        'MobileNetV2': {
            'class': MobileNetModel,
            'dataset': dataset_cnn,
            'lr': 1e-3,
            'batch_size': 32,
            'epochs': num_epochs
        }
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, config in models_config.items():
        print(f"\n[TRAINING {model_name}]")
        print("-" * 80)
        
        # Create dataloaders
        dataset = config['dataset']
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_subset, batch_size=config['batch_size'], shuffle=False)
        
        # Create and train model
        model = config['class'](num_classes=len(class_names))
        model = model.to(device)
        
        model_size = get_model_size(model)
        print(f"  Model: {model_name}")
        print(f"  Parameters: {model_size:,}")
        print(f"  Learning Rate: {config['lr']}")
        print(f"  Batch Size: {config['batch_size']}")
        
        # Train
        print(f"  Training...")
        training_history = train_model(
            model, train_loader, val_loader, 
            num_epochs=config['epochs'],
            learning_rate=config['lr'],
            device=device
        )
        
        # Evaluate
        print(f"  Evaluating...")
        test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, device)
        
        # Store results
        results[model_name] = {
            'parameters': model_size,
            'learning_rate': config['lr'],
            'batch_size': config['batch_size'],
            'training_time': training_history['total_time'],
            'epochs': config['epochs'],
            'test_accuracy': float(test_acc),
            'test_precision': float(test_prec),
            'test_recall': float(test_rec),
            'test_f1': float(test_f1),
            'training_history': {
                'train_loss': [float(x) for x in training_history['train_loss']],
                'val_acc': [float(x) for x in training_history['val_acc']]
            }
        }
        
        print(f"\n  Results:")
        print(f"    Accuracy: {test_acc:.4f}")
        print(f"    Precision: {test_prec:.4f}")
        print(f"    Recall: {test_rec:.4f}")
        print(f"    F1-Score: {test_f1:.4f}")
        print(f"    Training Time: {training_history['total_time']/60:.2f} minutes")
    
    return results, class_names

def plot_comparison(results, class_names):
    """Plot comparison results"""
    models = list(results.keys())
    
    # Accuracy comparison
    accuracies = [results[m]['test_accuracy'] for m in models]
    precisions = [results[m]['test_precision'] for m in models]
    recalls = [results[m]['test_recall'] for m in models]
    f1_scores = [results[m]['test_f1'] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy
    axes[0, 0].bar(models, accuracies, color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy', fontsize=11)
    axes[0, 0].set_title('Test Accuracy Comparison', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Precision
    axes[0, 1].bar(models, precisions, color='green', alpha=0.8)
    axes[0, 1].set_ylabel('Precision', fontsize=11)
    axes[0, 1].set_title('Test Precision Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precisions):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # Recall
    axes[1, 0].bar(models, recalls, color='orange', alpha=0.8)
    axes[1, 0].set_ylabel('Recall', fontsize=11)
    axes[1, 0].set_title('Test Recall Comparison', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # F1-Score
    axes[1, 1].bar(models, f1_scores, color='red', alpha=0.8)
    axes[1, 1].set_ylabel('F1-Score', fontsize=11)
    axes[1, 1].set_title('Test F1-Score Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.suptitle('Model Comparison - All Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {COMPARISON_DIR / 'model_comparison.png'}")
    plt.close()
    
    # Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    times = [results[m]['training_time']/60 for m in models]
    ax.bar(models, times, color='purple', alpha=0.8)
    ax.set_ylabel('Training Time (minutes)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(times):
        ax.text(i, v + 0.5, f'{v:.1f}m', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {COMPARISON_DIR / 'training_time_comparison.png'}")
    plt.close()

def print_summary(results):
    """Print summary table"""
    print("\n" + "="*80)
    print("COMPARATIVE STUDY RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Model':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Time (min)':<12}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        acc = metrics['test_accuracy']
        prec = metrics['test_precision']
        rec = metrics['test_recall']
        f1 = metrics['test_f1']
        time_min = metrics['training_time'] / 60
        
        print(f"{model_name:<15} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {time_min:<12.2f}")
    
    print("="*80)

def main():
    # Run comparative study
    results, class_names = run_comparative_study(num_epochs=15)
    
    # Save results
    output_data = {}
    for model, data in results.items():
        output_data[model] = {
            k: v for k, v in data.items() if k != 'training_history'
        }
    
    with open(METRICS_DIR / 'comparative_study_results.json', 'w') as f:
        json.dump(output_data, f, indent=4)
    
    # Plot comparison
    plot_comparison(results, class_names)
    
    # Print summary
    print_summary(results)
    
    print("\n✓ Comparative study completed!")
    print(f"✓ Results saved to {METRICS_DIR}")

if __name__ == '__main__':
    main()
