# Explainable Potato Leaf Disease Classification

This repository implements potato leaf disease classification using Vision Transformer and ResNet models, with evaluation, explainability, robustness testing, and comparative analysis. The project has been reorganized for a GitHub-ready structure with all source code in `scripts/`, dataset assets in `data/`, and grouped outputs in `outputs/`.

## Overview

The project classifies three PlantVillage classes:
- `Potato___Early_blight`
- `Potato___Late_blight`
- `Potato___healthy`

It includes:
- ViT training and evaluation
- ResNet50 baseline training and explainability
- Grad-CAM and attention-based visualizations
- Robustness testing under synthetic perturbations
- ViT vs CNN comparative study
- Clean directory handling through a shared path helper

## Repository Layout

```text
potato_leaf/
├── data/
│   ├── PlantVillage/                 # Dataset root used by the scripts
│   └── ViT_XAI_Potato_Assignment-G2.pdf
├── outputs/
│   ├── checkpoints/                  # Trained model checkpoints
│   ├── plots/                        # Training curves, confusion matrices, ROC curves
│   ├── metrics/                      # JSON reports and summaries
│   ├── xai/                          # Explainability images
│   │   ├── vit/
│   │   └── resnet/
│   ├── robustness/                   # Robustness plots
│   ├── comparison/                   # Comparative study plots
│   └── inference/                   # Inference result images
├── scripts/
│   ├── train_gpu.py                  # ViT training script
│   ├── evaluate_model.py             # ViT evaluation script
│   ├── inference.py                  # ViT inference script
│   ├── xai_explainability.py         # ViT explainability script
│   ├── robustness_test.py            # ViT robustness testing
│   ├── comparative_study.py          # ViT vs CNN comparison
│   ├── train_resnet.py               # ResNet50 training script
│   ├── xai_resnet.py                 # ResNet50 explainability script
│   └── project_paths.py              # Shared root/path resolver
├── requirements.txt
├── .gitignore
└── README.md
```

## Directory Rules

The code now assumes the repository root is `potato_leaf/`, and the scripts resolve paths from there.

### Dataset lookup order
The scripts look for the dataset in this order:
1. `data/PlantVillage/`
2. `PlantVillage/` at the repository root as a fallback

### Output structure
All generated artifacts are grouped under `outputs/`:
- `outputs/checkpoints/` for `.pth` checkpoints
- `outputs/plots/` for training plots, confusion matrices, ROC curves
- `outputs/metrics/` for JSON reports and summaries
- `outputs/xai/vit/` for ViT explainability images
- `outputs/xai/resnet/` for ResNet explainability images
- `outputs/robustness/` for robustness plots
- `outputs/comparison/` for comparative study plots
- `outputs/inference/` for inference visualizations

## Installation

### 1. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verify GPU support

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## How to Run

Run the scripts from the repository root.

### Train Vision Transformer

```powershell
python scripts/train_gpu.py
```

Outputs:
- `outputs/checkpoints/best_vit_potato_model.pth`
- `outputs/plots/training_history.png`
- `outputs/plots/confusion_matrix_validation.png`
- `outputs/plots/confusion_matrix_test.png`
- `outputs/metrics/training_summary.json`

### Evaluate Vision Transformer

```powershell
python scripts/evaluate_model.py
```

Outputs:
- `outputs/metrics/evaluation_report.json`
- `outputs/plots/confusion_matrix_eval.png`
- `outputs/plots/metrics_comparison.png`
- `outputs/plots/roc_curves.png`

### Run ViT Explainability

```powershell
python scripts/xai_explainability.py
```

Outputs:
- `outputs/xai/vit/xai_example.png`
- `outputs/xai/vit/` batch visualizations if directory evaluation is used

### Run ViT Inference

```powershell
python scripts/inference.py
```

Outputs:
- `outputs/inference/inference_results.png`

### Train ResNet50 Baseline

```powershell
python scripts/train_resnet.py
```

Outputs:
- `outputs/checkpoints/best_resnet_potato_model.pth`
- `outputs/metrics/resnet_training_summary.json`

### Run ResNet Explainability

```powershell
python scripts/xai_resnet.py
```

Outputs:
- `outputs/xai/resnet/` class-wise explainability images
- `outputs/xai/resnet/xai_resnet_summary.png`

### Comparative Study

```powershell
python scripts/comparative_study.py
```

Compares:
- Vision Transformer
- ResNet50
- VGG16
- MobileNetV2

Outputs:
- `outputs/comparison/model_comparison.png`
- `outputs/comparison/training_time_comparison.png`
- `outputs/metrics/comparative_study_results.json`

### Robustness Testing

```powershell
python scripts/robustness_test.py
```

Tests the trained ViT under:
- Gaussian noise
- Salt and pepper noise
- Gaussian blur
- Motion blur
- Brightness changes
- Contrast changes
- JPEG compression

Outputs:
- `outputs/robustness/robustness_results.png`
- `outputs/robustness/robustness_heatmap.png`

## Training Configuration

### Vision Transformer
- Backbone: `google/vit-base-patch16-224`
- Split: 70% train, 20% validation, 10% test
- Batch size: 16
- Learning rate: `5e-5`
- Weight decay: `0.01`
- Epochs: `10`
- Optimizer: AdamW
- Scheduler: warmup + cosine annealing

### ResNet50
- Backbone: `torchvision.models.resnet50`
- Epochs: `10`
- Batch size: `32`
- Learning rate: `1e-4`

## Explainability Methods

### ViT
The ViT explainability script generates:
- Grad-CAM heatmaps
- Attention rollout visualizations
- Side-by-side overlays for qualitative inspection

### ResNet
The ResNet explainability script generates:
- Grad-CAM heatmaps
- CNN feature rollout-style visualizations
- One example image per class

## Evaluation Metrics

The evaluation scripts report:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curves
- ROC-AUC
- Per-class metric breakdown

## Results Summary

### ViT Training Summary

`outputs/metrics/training_summary.json`

| Metric | Value |
| --- | ---: |
| Total samples | 2,152 |
| Train samples | 1,506 |
| Validation samples | 430 |
| Test samples | 216 |
| Classes | 3 |
| Backbone | `google/vit-base-patch16-224` |
| Batch size | 32 |
| Learning rate | `5e-5` |
| Weight decay | `0.01` |
| Epochs | 5 |
| Warmup steps | 100 |
| Image size | 224 |
| Best validation accuracy | 1.0 |
| Test accuracy | 1.0 |
| Test precision | 1.0 |
| Test recall | 1.0 |
| Test F1-score | 1.0 |
| Training time | 3.81 minutes |

### ViT Evaluation Report

`outputs/metrics/evaluation_report.json`

| Metric | Value |
| --- | ---: |
| Accuracy | 1.0 |
| Precision macro | 1.0 |
| Recall macro | 1.0 |
| F1 macro | 1.0 |
| Precision weighted | 1.0 |
| Recall weighted | 1.0 |
| F1 weighted | 1.0 |

Per-class metrics:

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Potato___Early_blight | 1.0 | 1.0 | 1.0 |
| Potato___Late_blight | 1.0 | 1.0 | 1.0 |
| Potato___healthy | 1.0 | 1.0 | 1.0 |

Confusion matrix:

| Actual \ Predicted | Early blight | Late blight | Healthy |
| --- | ---: | ---: | ---: |
| Early blight | 1000 | 0 | 0 |
| Late blight | 0 | 1000 | 0 |
| Healthy | 0 | 0 | 152 |

### ResNet50 Training Summary

`outputs/metrics/resnet_training_summary.json`

| Metric | Value |
| --- | ---: |
| Best validation accuracy | 0.9976744186 |
| Test loss | 0.0352684294 |
| Test accuracy | 0.9953703704 |
| Test precision | 0.9957561728 |
| Test recall | 0.9953703704 |
| Test F1-score | 0.9954597227 |
| Epochs | 10 |

### Comparative Study

`outputs/metrics/comparative_study_results.json`

| Model | Parameters | Learning rate | Batch size | Epochs | Training time (s) | Test accuracy | Test precision | Test recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT | 85,800,963 | `5e-5` | 16 | 15 | 579.16 | 1.0 | 1.0 | 1.0 | 1.0 |
| ResNet50 | 23,514,179 | `0.001` | 32 | 15 | 193.63 | 0.9907407407 | 0.9909979424 | 0.9907407407 | 0.9907928491 |
| VGG16 | 134,272,835 | `0.0001` | 32 | 15 | 9367.10 | 1.0 | 1.0 | 1.0 | 1.0 |
| MobileNetV2 | 2,227,715 | `0.001` | 32 | 15 | 229.17 | 0.9907407407 | 0.9908976773 | 0.9907407407 | 0.9907238513 |

### Result Artifacts

Key result files generated in the repository:
- `outputs/plots/training_history.png`
- `outputs/plots/confusion_matrix_validation.png`
- `outputs/plots/confusion_matrix_test.png`
- `outputs/plots/confusion_matrix_eval.png`
- `outputs/plots/roc_curves.png`
- `outputs/plots/metrics_comparison.png`
- `outputs/comparison/model_comparison.png`
- `outputs/comparison/training_time_comparison.png`
- `outputs/inference/inference_results.png`
- `outputs/robustness/robustness_results.png`
- `outputs/robustness/robustness_heatmap.png`
- `outputs/xai/vit/xai_example.png`
- `outputs/xai/resnet/xai_resnet_summary.png`
- `outputs/xai/resnet/xai_Potato___Early_blight.png`
- `outputs/xai/resnet/xai_Potato___Late_blight.png`
- `outputs/xai/resnet/xai_Potato___healthy.png`

## GitHub Guidance

When pushing to GitHub, keep these rules in mind:
- Do not commit `venv/`
- Do not commit the raw dataset if the repository should stay lightweight
- Keep `data/ViT_XAI_Potato_Assignment-G2.pdf` if you want the assignment document tracked in the repo
- Generated files are already grouped under `outputs/`
- Large trained weight files are stored locally in `models/` in this checkout; if you want them versioned on GitHub, use Git LFS or another artifact store because GitHub rejects files larger than 100 MB.

## Troubleshooting

### Checkpoint not found
Make sure you have already trained the model so the checkpoint exists under `outputs/checkpoints/`.

### Dataset not found
Place the PlantVillage dataset in one of these locations:
- `data/PlantVillage/`
- `PlantVillage/` at the repository root

### CUDA not available
Verify that your local PyTorch installation is the CUDA-enabled build and that the NVIDIA driver is installed correctly.

### Script path issues
Run commands from the repository root, not from inside `scripts/`.

## Deliverables

- Source code in `scripts/`
- Dataset in `data/`
- Grouped outputs in `outputs/`
- Installation instructions in this README
- Dependency list in `requirements.txt`

## Group Members

- ARKO BERA (BT23CSD001)
- VISHAL SINGH (BT23CSD002)
- TANISHQ PARIHAR (BT23CSD005)
- MITHRA (BT23CSD025)
- UTKARSH GAUR (BT23CSD055)

## License

Educational project for computer vision and deep learning coursework.
