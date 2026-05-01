
import os
import json
import time
import copy
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

try:
    from transformers import ViTImageProcessor, ViTForImageClassification
    TRANSFORMERS_AVAILABLE = True
except Exception as exc:
    TRANSFORMERS_AVAILABLE = False
    print('Transformers import failed. ViT will be skipped unless transformers is available:', exc)
print("done")

# ## Configuration
# 
# If Kaggle does not auto-detect the dataset, set `DATA_ROOT` to the folder that directly contains `Potato___Early_blight`, `Potato___Late_blight`, and `Potato___healthy`.


CONFIG = {
    'DATA_ROOT': None,
    'EXPECTED_CLASSES': ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    'OUTPUT_DIR': '/kaggle/working/potato_xai_results',
    'SEED': 42,
    'IMAGE_SIZE': 224,
    'BATCH_SIZE': 16,
    'NUM_WORKERS': 0,
    'USE_AMP': True,
    'USE_PRETRAINED': True,
    'NUM_FOLDS': 3,
    'TEST_SIZE': 0.10,

    # Main overfitting-controlled CV model.
    'RUN_3_FOLD_CV': True,
    'CV_MODEL': 'resnet50',
    'CV_EPOCHS': 8,
    'CV_PATIENCE': 2,

    # ViT run used for old-repo ViT-style results and attention visualization.
    'RUN_VIT': True,
    'VIT_MODEL_NAME': 'google/vit-base-patch16-224',
    'VIT_EPOCHS': 5,
    'VIT_PATIENCE': 2,
    'VIT_BATCH_SIZE': 8,
    'VIT_LR': 5e-5,
    'VIT_WEIGHT_DECAY': 0.01,

    # Comparative study. Kept shorter because it trains 4 models.
    'RUN_COMPARISON': True,
    'COMPARISON_MODELS': ['vit', 'resnet50', 'vgg16'],
    'COMPARISON_EPOCHS': 4,
    'COMPARISON_PATIENCE': 2,

    'LEARNING_RATE': 1e-4,
    'UNFROZEN_LR_MULTIPLIER': 0.25,
    'WEIGHT_DECAY': 1e-4,
    'LABEL_SMOOTHING': 0.05,
    'DROPOUT': 0.30,
    'FREEZE_BACKBONE_EPOCHS': 2,
    'GRAD_CLIP_NORM': 1.0,

    'RUN_XAI': True,
    'RUN_ROBUSTNESS': True,
    'ROBUSTNESS_MAX_IMAGES_PER_CLASS': 10,
    'RUN_INFERENCE_VISUALIZATION': True,
}

OUTPUT_DIR = Path(CONFIG['OUTPUT_DIR'])
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
PLOT_DIR = OUTPUT_DIR / 'plots'
METRIC_DIR = OUTPUT_DIR / 'metrics'
XAI_DIR = OUTPUT_DIR / 'xai'
ROBUSTNESS_DIR = OUTPUT_DIR / 'robustness'
COMPARISON_DIR = OUTPUT_DIR / 'comparison'
INFERENCE_DIR = OUTPUT_DIR / 'inference'
for folder in [OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, METRIC_DIR, XAI_DIR, ROBUSTNESS_DIR, COMPARISON_DIR, INFERENCE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('CUDA devices:', torch.cuda.device_count())
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        print(f'GPU {idx}:', torch.cuda.get_device_name(idx))


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def maybe_parallel(model):
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f'Using DataParallel on {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    return model


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def model_state_dict(model):
    return unwrap_model(model).state_dict()


seed_everything(CONFIG['SEED'])


# ## Dataset Discovery


from pathlib import Path
import pandas as pd

# Auto-find dataset root
matches = list(Path("/kaggle/input").rglob("PLD_3_Classes_256"))

if not matches:
    print("Available Kaggle folders:")
    for p in Path("/kaggle/input").rglob("*"):
        if p.is_dir():
            print(p)
    raise FileNotFoundError("Could not find PLD_3_Classes_256 under /kaggle/input")

DATA_ROOT = matches[0]
print("Using DATA_ROOT:", DATA_ROOT)

CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]
SPLITS = ["Training", "Validation", "Testing"]
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def build_split_dataframe(split_name):
    rows = []
    split_dir = DATA_ROOT / split_name

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({
                    "path": str(image_path),
                    "label": class_to_idx[class_name],
                    "class_name": class_name,
                    "split": split_name,
                })

    return pd.DataFrame(rows, columns=["path", "label", "class_name", "split"])

train_df = build_split_dataframe("Training")
val_df = build_split_dataframe("Validation")
test_df = build_split_dataframe("Testing")

df = pd.concat([train_df, val_df, test_df], ignore_index=True)
cv_df = pd.concat([train_df, val_df], ignore_index=True)

class_names = CLASS_NAMES
num_classes = len(class_names)

print("Classes:", class_names)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))
print("CV images:", len(cv_df))
print("Total images:", len(df))

if len(df) == 0:
    raise RuntimeError("No images found. Check dataset folder names and image extensions.")

display(df.groupby(["split", "class_name"]).size().rename("count").reset_index())


# This Kaggle dataset already provides Training, Validation, and Testing folders.
# Keep those predefined splits instead of creating a second random split.
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
cv_df = pd.concat([train_df, val_df], ignore_index=True)

print(f'Train images for single-model runs: {len(train_df)}')
print(f'Validation images for single-model runs: {len(val_df)}')
print(f'Final untouched test images: {len(test_df)}')
print(f'CV images for 3-fold split: {len(cv_df)}')


# ## Datasets, Transforms, and Model Builders


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(CONFIG['IMAGE_SIZE'], scale=(0.70, 1.00), ratio=(0.90, 1.10)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.10),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), shear=8),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random'),
])

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(CONFIG['IMAGE_SIZE']),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

plain_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(CONFIG['IMAGE_SIZE']),
])

class PotatoLeafDataset(Dataset):
    def __init__(self, dataframe, transform=None, return_path=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_path = return_path

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = int(row['label'])
        if self.return_path:
            return image, label, row['path']
        return image, label


def make_loader(dataframe, transform, shuffle=False, batch_size=None, return_path=False):
    dataset = PotatoLeafDataset(dataframe, transform=transform, return_path=return_path)
    return DataLoader(
        dataset,
        batch_size=batch_size or CONFIG['BATCH_SIZE'],
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def pretrained_weights(weight_enum):
    if not CONFIG['USE_PRETRAINED']:
        return None
    try:
        return weight_enum.DEFAULT
    except Exception:
        return None


def build_cnn_model(model_name, num_classes):
    model_name = model_name.lower()
    if model_name == 'resnet50':
        weights = pretrained_weights(models.ResNet50_Weights)
        try:
            model = models.resnet50(weights=weights)
        except Exception as exc:
            print('ResNet50 pretrained weights unavailable, using random init:', exc)
            model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(CONFIG['DROPOUT']), nn.Linear(in_features, num_classes))
        final_prefix = 'fc.'
    elif model_name == 'vgg16':
        weights = pretrained_weights(models.VGG16_Weights)
        try:
            model = models.vgg16(weights=weights)
        except Exception as exc:
            print('VGG16 pretrained weights unavailable, using random init:', exc)
            model = models.vgg16(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        model.classifier[5] = nn.Dropout(CONFIG['DROPOUT'])
        final_prefix = 'classifier.'
    else:
        raise ValueError(f'Unsupported CNN model: {model_name}')
    return model, final_prefix


class ViTWrapper(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, pixel_values):
        return self.vit(pixel_values=pixel_values).logits


def build_model(model_name, num_classes):
    if model_name == 'vit':
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError('transformers is not available, so ViT cannot be trained in this runtime.')
        return ViTWrapper(CONFIG['VIT_MODEL_NAME'], num_classes), 'vit.classifier.'
    return build_cnn_model(model_name, num_classes)


def set_backbone_trainable(model, final_prefix, trainable):
    base = unwrap_model(model)
    for name, parameter in base.named_parameters():
        parameter.requires_grad = trainable or name.startswith(final_prefix)


def make_optimizer(model, learning_rate):
    return optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=CONFIG['WEIGHT_DECAY'],
    )


# ## Training and Evaluation Helpers


def run_one_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, all_labels, all_preds, all_probs = [], [], [], []
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch in tqdm(loader, leave=False, desc='train' if is_train else 'eval'):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            amp_enabled = bool(CONFIG['USE_AMP'] and device.type == 'cuda')
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                scaler.scale(loss).backward()
                if CONFIG['GRAD_CLIP_NORM'] is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRAD_CLIP_NORM'])
                scaler.step(optimizer)
                scaler.update()

            probs = torch.softmax(logits.detach(), dim=1)
            preds = probs.argmax(dim=1)
            losses.append(float(loss.item()))
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

    return {
        'loss': float(np.mean(losses)),
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
    }


def train_model(model_name, train_part, val_part, epochs, patience, checkpoint_name, batch_size=None, lr=None):
    train_loader = make_loader(train_part, train_transform, shuffle=True, batch_size=batch_size)
    val_loader = make_loader(val_part, eval_transform, shuffle=False, batch_size=batch_size)

    raw_model, final_prefix = build_model(model_name, num_classes)
    model = maybe_parallel(raw_model)
    set_backbone_trainable(model, final_prefix, trainable=CONFIG['FREEZE_BACKBONE_EPOCHS'] == 0 or model_name == 'vit')

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['LABEL_SMOOTHING'])
    base_lr = lr or CONFIG['LEARNING_RATE']
    optimizer = make_optimizer(model, base_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(CONFIG['USE_AMP'] and device.type == 'cuda'))

    best_f1, best_epoch, best_state = -1.0, 0, None
    patience_counter = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if model_name != 'vit' and epoch == CONFIG['FREEZE_BACKBONE_EPOCHS'] + 1 and CONFIG['FREEZE_BACKBONE_EPOCHS'] > 0:
            set_backbone_trainable(model, final_prefix, trainable=True)
            optimizer = make_optimizer(model, base_lr * CONFIG['UNFROZEN_LR_MULTIPLIER'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, min_lr=1e-7)
            print(f'{model_name}: backbone unfrozen for fine-tuning.')

        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer=optimizer, scaler=scaler)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer=None, scaler=None)
        scheduler.step(val_metrics['loss'])

        row = {
            'model': model_name,
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'lr': float(optimizer.param_groups[0]['lr']),
        }
        history.append(row)
        print(f"{model_name} epoch {epoch:02d}/{epochs} | train acc {row['train_acc']:.4f} f1 {row['train_f1']:.4f} | val acc {row['val_acc']:.4f} f1 {row['val_f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            best_state = copy.deepcopy(model_state_dict(model))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'{model_name}: early stopping at epoch {epoch}; best epoch {best_epoch}.')
                break

    unwrap_model(model).load_state_dict(best_state)
    final_val = run_one_epoch(model, val_loader, criterion, optimizer=None, scaler=None)
    elapsed_minutes = (time.time() - start_time) / 60

    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    torch.save({
        'model_name': model_name,
        'model_state_dict': model_state_dict(model),
        'class_names': class_names,
        'config': CONFIG,
        'best_epoch': best_epoch,
        'best_val_f1': final_val['f1'],
    }, checkpoint_path)

    summary = {
        'model': model_name,
        'best_epoch': best_epoch,
        'val_loss': final_val['loss'],
        'val_accuracy': final_val['accuracy'],
        'val_precision': final_val['precision'],
        'val_recall': final_val['recall'],
        'val_f1': final_val['f1'],
        'training_minutes': elapsed_minutes,
        'checkpoint': str(checkpoint_path),
    }
    return model, pd.DataFrame(history), summary


# ## 3-Fold Cross-Validation with Overfitting Control


cv_models, cv_histories, cv_summaries = [], [], []
if CONFIG['RUN_3_FOLD_CV']:
    skf = StratifiedKFold(n_splits=CONFIG['NUM_FOLDS'], shuffle=True, random_state=CONFIG['SEED'])
    for fold, (train_idx, val_idx) in enumerate(skf.split(cv_df['path'], cv_df['label']), start=1):
        print('\n' + '=' * 90)
        print(f"CV fold {fold}/{CONFIG['NUM_FOLDS']} using {CONFIG['CV_MODEL']}")
        print('=' * 90)
        model, history, summary = train_model(
            CONFIG['CV_MODEL'],
            cv_df.iloc[train_idx].reset_index(drop=True),
            cv_df.iloc[val_idx].reset_index(drop=True),
            epochs=CONFIG['CV_EPOCHS'],
            patience=CONFIG['CV_PATIENCE'],
            checkpoint_name=f"best_{CONFIG['CV_MODEL']}_fold_{fold}.pth",
        )
        summary['fold'] = fold
        history['fold'] = fold
        cv_models.append(model)
        cv_histories.append(history)
        cv_summaries.append(summary)
        torch.cuda.empty_cache()

    cv_history_df = pd.concat(cv_histories, ignore_index=True)
    cv_summary_df = pd.DataFrame(cv_summaries)
    cv_history_df.to_csv(METRIC_DIR / 'cv_training_history.csv', index=False)
    cv_summary_df.to_csv(METRIC_DIR / 'cv_fold_summary.csv', index=False)
    display(cv_summary_df)
else:
    cv_history_df = pd.DataFrame()
    cv_summary_df = pd.DataFrame()


if not cv_history_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(data=cv_history_df, x='epoch', y='train_loss', hue='fold', marker='o', ax=axes[0])
    sns.lineplot(data=cv_history_df, x='epoch', y='val_loss', hue='fold', marker='s', ax=axes[0], linestyle='--')
    axes[0].set_title('3-Fold CV Loss')
    sns.lineplot(data=cv_history_df, x='epoch', y='train_f1', hue='fold', marker='o', ax=axes[1])
    sns.lineplot(data=cv_history_df, x='epoch', y='val_f1', hue='fold', marker='s', ax=axes[1], linestyle='--')
    axes[1].set_title('3-Fold CV Weighted F1')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / 'training_history.png', dpi=220, bbox_inches='tight')
    plt.show()


# ## ViT Training


import torch
import torch.nn as nn
from transformers import ViTForImageClassification
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import time

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time=time.time()

vit_train_loader=make_loader(
    train_df,
    transform=train_transform,
    shuffle=True,
    batch_size=CONFIG['VIT_BATCH_SIZE']
)

vit_val_loader=make_loader(
    val_df,
    transform=eval_transform,
    shuffle=False,
    batch_size=CONFIG['VIT_BATCH_SIZE']
)

model=ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=3,
    ignore_mismatched_sizes=True
)

model.config.hidden_dropout_prob=0.2
model.config.attention_probs_dropout_prob=0.2

model=model.to(device)

for param in model.vit.parameters():
    param.requires_grad=False


criterion=nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=3e-5,
    weight_decay=0.05
)

history=[]
best_val_acc=0.0
best_epoch=0
best_metrics=None

patience=CONFIG['VIT_PATIENCE']
patience_counter=0

for epoch in range(CONFIG['VIT_EPOCHS']):
    model.train()
    train_correct=0
    train_total=0

    for images,labels in vit_train_loader:
        images=images.to(device)
        labels=labels.to(device)

        optimizer.zero_grad()
        outputs=model(images).logits
        loss=criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        _,preds=torch.max(outputs,1)
        train_correct+=(preds==labels).sum().item()
        train_total+=labels.size(0)

    train_acc=train_correct/train_total

    model.eval()
    val_correct=0
    val_total=0
    val_loss=0

    all_preds=[]
    all_labels=[]

    with torch.no_grad():
        for images,labels in vit_val_loader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images).logits
            loss=criterion(outputs,labels)

            val_loss+=loss.item()

            _,preds=torch.max(outputs,1)

            val_correct+=(preds==labels).sum().item()
            val_total+=labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc=val_correct/val_total
    val_loss=val_loss/len(vit_val_loader)

    val_precision=precision_score(all_labels,all_preds,average='macro')
    val_recall=recall_score(all_labels,all_preds,average='macro')
    val_f1=f1_score(all_labels,all_preds,average='macro')

    print(f"vit epoch {epoch+1}/{CONFIG['VIT_EPOCHS']} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

    history.append({
        "epoch":epoch+1,
        "train_acc":train_acc,
        "val_acc":val_acc
    })

    # unfreeze after 1 epoch
    if epoch==1:
        for param in model.vit.parameters():
            param.requires_grad=True
        print("vit backbone unfrozen")

    # best tracking
    if val_acc>best_val_acc:
        best_val_acc=val_acc
        best_epoch=epoch+1
        patience_counter=0

        best_metrics={
            "val_loss":val_loss,
            "val_accuracy":val_acc,
            "val_precision":val_precision,
            "val_recall":val_recall,
            "val_f1":val_f1
        }

        torch.save(model.state_dict(),"/kaggle/working/best_vit.pth")
    else:
        patience_counter+=1

    if patience_counter>=patience:
        print(f"vit: early stopping at epoch {epoch+1}")
        break


training_minutes=(time.time()-start_time)/60

vit_summary={
    "model":"vit",
    "best_epoch":best_epoch,
    "val_loss":best_metrics["val_loss"],
    "val_accuracy":best_metrics["val_accuracy"],
    "val_precision":best_metrics["val_precision"],
    "val_recall":best_metrics["val_recall"],
    "val_f1":best_metrics["val_f1"],
    "training_minutes":training_minutes
}

vit_history_df=pd.DataFrame(history)

METRIC_DIR=Path("/kaggle/working")

vit_history_df.to_csv(METRIC_DIR/"vit_training_history.csv",index=False)

with open(METRIC_DIR/"vit_training_summary.json","w") as f:
    json.dump(vit_summary,f,indent=2)

display(pd.DataFrame([vit_summary]))

# ## ViT vs ResNet/VGG Comparative Study


comparison_models = {}
comparison_histories = []
comparison_summaries = []

if CONFIG['RUN_COMPARISON']:
    for model_name in CONFIG['COMPARISON_MODELS']:
        if model_name == 'vit' and not TRANSFORMERS_AVAILABLE:
            print('Skipping ViT in comparison because transformers is unavailable.')
            continue
        if model_name == 'vit' and vit_model is not None and vit_summary is not None:
            print('Reusing the already trained ViT for the comparison table.')
            comparison_models['vit'] = vit_model
            comparison_summaries.append({**vit_summary, 'model': 'vit'})
            if not vit_history_df.empty:
                comparison_histories.append(vit_history_df.assign(model='vit'))
            continue
        print('\n' + '=' * 90)
        print(f'Comparative training: {model_name}')
        print('=' * 90)
        batch_size = CONFIG['VIT_BATCH_SIZE'] if model_name == 'vit' else CONFIG['BATCH_SIZE']
        lr = CONFIG['VIT_LR'] if model_name == 'vit' else CONFIG['LEARNING_RATE']
        try:
            model, history, summary = train_model(
                model_name,
                train_df,
                val_df,
                epochs=CONFIG['COMPARISON_EPOCHS'],
                patience=CONFIG['COMPARISON_PATIENCE'],
                checkpoint_name=f'comparison_{model_name}.pth',
                batch_size=batch_size,
                lr=lr,
            )
        except Exception as exc:
            print(f'Skipping {model_name} after training/setup failure: {exc}')
            continue
        comparison_models[model_name] = model
        comparison_histories.append(history)
        comparison_summaries.append(summary)
        torch.cuda.empty_cache()

    comparison_df = pd.DataFrame(comparison_summaries)
    comparison_df.to_csv(METRIC_DIR / 'comparative_study_results.csv', index=False)
    with open(METRIC_DIR / 'comparative_study_results.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_summaries, f, indent=2)
    display(comparison_df)
else:
    comparison_df = pd.DataFrame()


import pandas as pd
import json
from pathlib import Path

METRIC_DIR=Path("/kaggle/working")

old_df=pd.read_csv(METRIC_DIR/"comparative_study_results.csv")

old_df=old_df[old_df["model"]!="vit"]

new_vit_df=pd.DataFrame([vit_summary])

final_df=pd.concat([old_df,new_vit_df],ignore_index=True)

final_df.to_csv(METRIC_DIR/"comparative_study_results.csv",index=False)

with open(METRIC_DIR/"comparative_study_results.json","w") as f:
    json.dump(final_df.to_dict(orient="records"),f,indent=2)

display(final_df)

for col in vit_summary.keys():
    comparison_df.loc[comparison_df["model"]=="vit", col] = vit_summary[col]

for col in vit_summary.keys():
    if col not in comparison_df.columns:
        comparison_df[col] = None
        comparison_df.loc[comparison_df["model"]=="vit", col] = vit_summary[col]

print(comparison_df)

if not comparison_df.empty:
    metric_cols = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
    plot_df = comparison_df.melt(id_vars='model', value_vars=metric_cols, var_name='metric', value_name='score')
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x='model', y='score', hue='metric')
    plt.ylim(0, 1.02)
    plt.title('ViT vs CNN Model Comparison')
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'model_comparison.png', dpi=220, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=comparison_df, x='model', y='training_minutes')
    plt.title('Training Time Comparison')
    plt.ylabel('Minutes')
    plt.tight_layout()
    plt.savefig(COMPARISON_DIR / 'training_time_comparison.png', dpi=220, bbox_inches='tight')
    plt.show()

# ## Final Test Evaluation, Confusion Matrix, and ROC Curves


def evaluate_model_on_loader(model, loader):
    criterion = nn.CrossEntropyLoss()
    return run_one_epoch(model, loader, criterion, optimizer=None, scaler=None)


def evaluate_ensemble(models_list, loader):
    for model in models_list:
        model.eval()
    losses, all_labels, all_preds, all_probs = [], [], [], []
    nll_loss = nn.NLLLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc='test ensemble'):
            images, labels = batch[:2]
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probs = [torch.softmax(model(images), dim=1) for model in models_list]
            mean_probs = torch.stack(probs, dim=0).mean(dim=0)
            loss = nll_loss(torch.log(mean_probs.clamp_min(1e-8)), labels)
            preds = mean_probs.argmax(dim=1)
            losses.append(float(loss.item()))
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(mean_probs.cpu().numpy().tolist())
    return {
        'loss': float(np.mean(losses)),
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs,
    }


test_loader = make_loader(test_df, eval_transform, shuffle=False)
if cv_models:
    final_model_name = f"{CONFIG['CV_MODEL']}_3fold_ensemble"
    test_metrics = evaluate_ensemble(cv_models, test_loader)
elif vit_model is not None:
    final_model_name = 'vit'
    test_metrics = evaluate_model_on_loader(vit_model, test_loader)
elif comparison_models:
    best_name = comparison_df.sort_values('val_f1', ascending=False).iloc[0]['model']
    final_model_name = best_name
    test_metrics = evaluate_model_on_loader(comparison_models[best_name], test_loader)
else:
    raise RuntimeError('No trained model is available for final test evaluation.')

print('Final model:', final_model_name)
print(f"Test loss:      {test_metrics['loss']:.4f}")
print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Test precision: {test_metrics['precision']:.4f}")
print(f"Test recall:    {test_metrics['recall']:.4f}")
print(f"Test F1 score:  {test_metrics['f1']:.4f}")
print('\nClassification report:')
print(classification_report(test_metrics['labels'], test_metrics['preds'], target_names=class_names, zero_division=0))


cm = confusion_matrix(test_metrics['labels'], test_metrics['preds'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Final Test Confusion Matrix')
plt.tight_layout()
plt.savefig(PLOT_DIR / 'confusion_matrix_test.png', dpi=220, bbox_inches='tight')
plt.show()

labels_bin = label_binarize(test_metrics['labels'], classes=list(range(num_classes)))
probs = np.array(test_metrics['probs'])
plt.figure(figsize=(8, 6))
roc_auc = {}
for idx, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(labels_bin[:, idx], probs[:, idx])
    roc_auc[class_name] = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{class_name} AUC={roc_auc[class_name]:.3f}')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / 'roc_curves.png', dpi=220, bbox_inches='tight')
plt.show()


# ## Grad-CAM / XAI Visualizations


def pick_xai_model():
    if comparison_models.get('resnet50') is not None:
        return comparison_models['resnet50'], 'resnet50'
    if cv_models:
        return cv_models[0], CONFIG['CV_MODEL']
    for name in ['vgg16']:
        if comparison_models.get(name) is not None:
            return comparison_models[name], name
    return None, None


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def target_layer_for_model(model, name):
    base = unwrap_model(model)
    if name == 'resnet50':
        return base.layer4[-1]
    if name == 'vgg16':
        return base.features[-1]
    return None


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def save_gradcam_examples(model, model_name):
    target_layer = target_layer_for_model(model, model_name)
    if target_layer is None:
        print('Grad-CAM target layer is unavailable for', model_name)
        return
    cammer = GradCAM(model, target_layer)
    examples = test_df.groupby('label', group_keys=False).head(1).reset_index(drop=True)
    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, row in examples.iterrows():
        image = Image.open(row['path']).convert('RGB')
        x = eval_transform(image).unsqueeze(0).to(device)
        cam, pred_idx = cammer(x)
        img_np = denormalize(x.squeeze(0))
        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"Original: {row['class_name']}")
        axes[row_idx, 1].imshow(cam, cmap='jet')
        axes[row_idx, 1].set_title('Grad-CAM heatmap')
        axes[row_idx, 2].imshow(img_np)
        axes[row_idx, 2].imshow(cam, cmap='jet', alpha=0.45)
        axes[row_idx, 2].set_title(f'Overlay pred: {class_names[pred_idx]}')
        for ax in axes[row_idx]:
            ax.axis('off')
    plt.tight_layout()
    out = XAI_DIR / f'gradcam_{model_name}_summary.png'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    plt.show()
    cammer.close()
    print('Saved:', out)


if CONFIG['RUN_XAI']:
    xai_model, xai_model_name = pick_xai_model()
    if xai_model is not None:
        save_gradcam_examples(xai_model, xai_model_name)
    else:
        print('No CNN model available for Grad-CAM.')


def save_vit_attention_like_examples(model):
    if model is None:
        print('No ViT model available for ViT XAI visualization.')
        return

    base = unwrap_model(model)
    hf_model = base.vit

    if hasattr(hf_model, "set_attn_implementation"):
        hf_model.set_attn_implementation("eager")
    else:
        hf_model.config._attn_implementation = "eager"

    hf_model.config.output_attentions = True

    examples = test_df.groupby('label', group_keys=False).head(1).reset_index(drop=True)

    fig, axes = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)))
    if len(examples) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, row in examples.iterrows():
        image = Image.open(row['path']).convert('RGB')
        x = eval_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = hf_model(pixel_values=x, output_attentions=True, return_dict=True)

            if outputs.attentions is None:
                print("attentions still None")
                continue

            pred_idx = int(outputs.logits.argmax(dim=1).item())

            attn = outputs.attentions[-1].mean(dim=1)[0, 0, 1:]
            grid_size = int(np.sqrt(attn.numel()))
            attn_map = attn.reshape(grid_size, grid_size).cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        img_np = denormalize(x.squeeze(0))

        axes[row_idx, 0].imshow(img_np)
        axes[row_idx, 0].set_title(f"Original: {row['class_name']}")

        axes[row_idx, 1].imshow(attn_map, cmap='magma')
        axes[row_idx, 1].set_title('Attention map')

        axes[row_idx, 2].imshow(img_np)
        axes[row_idx, 2].imshow(
            Image.fromarray((attn_map * 255).astype(np.uint8)).resize(
                (CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])
            ),
            cmap='magma',
            alpha=0.45
        )
        axes[row_idx, 2].set_title(f'Overlay: {class_names[pred_idx]}')

        for ax in axes[row_idx]:
            ax.axis('off')

    plt.tight_layout()
    out = XAI_DIR / 'vit_attention_summary.png'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    plt.show()
    print('Saved:', out)


if CONFIG['RUN_XAI'] and vit_model is not None:
    save_vit_attention_like_examples(vit_model)

# ## Robustness Testing


def perturb_image(image, perturbation, severity):
    image = image.convert('RGB')
    arr = np.array(image).astype(np.float32)
    if perturbation == 'gaussian_noise':
        noise = np.random.normal(0, severity * 255, arr.shape)
        arr = np.clip(arr + noise, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if perturbation == 'salt_pepper':
        prob = severity
        rnd = np.random.rand(*arr.shape[:2])
        arr[rnd < prob / 2] = 0
        arr[rnd > 1 - prob / 2] = 255
        return Image.fromarray(arr.astype(np.uint8))
    if perturbation == 'gaussian_blur':
        return image.filter(ImageFilter.GaussianBlur(radius=severity))
    if perturbation == 'brightness':
        return ImageEnhance.Brightness(image).enhance(severity)
    if perturbation == 'contrast':
        return ImageEnhance.Contrast(image).enhance(severity)
    raise ValueError(perturbation)


def predict_dataframe_with_transform(model, dataframe, transform):
    loader = make_loader(dataframe, transform, shuffle=False, batch_size=CONFIG['BATCH_SIZE'])
    metrics = evaluate_model_on_loader(model, loader)
    return metrics['accuracy'], metrics['f1']


class PerturbedDataset(Dataset):
    def __init__(self, dataframe, perturbation, severity, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.perturbation = perturbation
        self.severity = severity
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row['path']).convert('RGB')
        image = perturb_image(image, self.perturbation, self.severity)
        return self.transform(image), int(row['label'])


def evaluate_robustness(model):
    sample_df = test_df.groupby('label', group_keys=False).head(CONFIG['ROBUSTNESS_MAX_IMAGES_PER_CLASS']).reset_index(drop=True)
    tests = {
        'gaussian_noise': [0.03, 0.06, 0.10],
        'salt_pepper': [0.01, 0.03, 0.06],
        'gaussian_blur': [1, 2, 4],
        'brightness': [0.60, 1.40, 1.80],
        'contrast': [0.60, 1.40, 1.80],
    }
    rows = []
    for perturbation, severities in tests.items():
        for severity in severities:
            loader = DataLoader(
                PerturbedDataset(sample_df, perturbation, severity, eval_transform),
                batch_size=CONFIG['BATCH_SIZE'],
                shuffle=False,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            metrics = evaluate_model_on_loader(model, loader)
            rows.append({
                'perturbation': perturbation,
                'severity': severity,
                'accuracy': metrics['accuracy'],
                'f1': metrics['f1'],
            })
            print(f"{perturbation:16s} severity={severity} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
    return pd.DataFrame(rows)


if CONFIG['RUN_ROBUSTNESS']:
    robustness_model = cv_models[0] if cv_models else (comparison_models.get('resnet50') or vit_model)
    if robustness_model is None:
        print('No model available for robustness testing.')
        robustness_df = pd.DataFrame()
    else:
        robustness_df = evaluate_robustness(robustness_model)
        robustness_df.to_csv(METRIC_DIR / 'robustness_results.csv', index=False)
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=robustness_df, x='severity', y='accuracy', hue='perturbation', marker='o')
        plt.title('Robustness Accuracy Under Perturbations')
        plt.tight_layout()
        plt.savefig(ROBUSTNESS_DIR / 'robustness_results.png', dpi=220, bbox_inches='tight')
        plt.show()

        heatmap_df = robustness_df.pivot(index='perturbation', columns='severity', values='accuracy')
        plt.figure(figsize=(10, 5))
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Robustness Heatmap - Accuracy')
        plt.tight_layout()
        plt.savefig(ROBUSTNESS_DIR / 'robustness_heatmap.png', dpi=220, bbox_inches='tight')
        plt.show()
else:
    robustness_df = pd.DataFrame()


# ## Inference Visualization


def visualize_inference(model):
    examples = test_df.groupby('label', group_keys=False).head(3).reset_index(drop=True)
    n = len(examples)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4 * rows))
    axes = np.array(axes).reshape(-1)
    model.eval()
    with torch.no_grad():
        for idx, row in examples.iterrows():
            image = Image.open(row['path']).convert('RGB')
            x = eval_transform(image).unsqueeze(0).to(device)
            probs = torch.softmax(model(x), dim=1).squeeze(0).cpu().numpy()
            pred_idx = int(np.argmax(probs))
            axes[idx].imshow(plain_transform(image))
            color = 'green' if pred_idx == int(row['label']) else 'red'
            axes[idx].set_title(
                f"True: {row['class_name']}\nPred: {class_names[pred_idx]} ({probs[pred_idx]:.2%})",
                color=color,
                fontsize=10,
            )
            axes[idx].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    out = INFERENCE_DIR / 'inference_results.png'
    plt.savefig(out, dpi=220, bbox_inches='tight')
    plt.show()
    print('Saved:', out)


if CONFIG['RUN_INFERENCE_VISUALIZATION']:
    inference_model = cv_models[0] if cv_models else (comparison_models.get('resnet50') or vit_model)
    visualize_inference(inference_model)


# ## Save Final Summary


summary = {
    'data_root': str(DATA_ROOT),
    'class_names': class_names,
    'num_images_total': int(len(df)),
    'num_images_train_single': int(len(train_df)),
    'num_images_val_single': int(len(val_df)),
    'num_images_cv': int(len(cv_df)),
    'num_images_test': int(len(test_df)),
    'used_cuda_devices': int(torch.cuda.device_count()),
    'data_parallel_enabled': bool(torch.cuda.device_count() > 1),
    'config': CONFIG,
    'cv_summary': cv_summary_df.to_dict(orient='records') if not cv_summary_df.empty else [],
    'vit_summary': vit_summary,
    'comparison_summary': comparison_df.to_dict(orient='records') if not comparison_df.empty else [],
    'test_summary': {k: v for k, v in test_metrics.items() if k not in ['labels', 'preds', 'probs']},
    'roc_auc': roc_auc,
    'robustness_summary': robustness_df.to_dict(orient='records') if not robustness_df.empty else [],
}

with open(METRIC_DIR / 'training_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print('All outputs saved under:', OUTPUT_DIR)
print('Checkpoints:', CHECKPOINT_DIR)
print('Plots:', PLOT_DIR)
print('Metrics:', METRIC_DIR)
print('XAI:', XAI_DIR)
print('Robustness:', ROBUSTNESS_DIR)
print('Comparison:', COMPARISON_DIR)
print('Inference:', INFERENCE_DIR)
