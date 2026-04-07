"""Shared path helpers for the potato leaf disease classification project."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [ROOT_DIR / "data" / "PlantVillage", ROOT_DIR / "PlantVillage"]
OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
XAI_DIR = OUTPUT_DIR / "xai"
ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
INFERENCE_DIR = OUTPUT_DIR / "inference"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"


def resolve_data_dir() -> Path:
    """Return the first existing PlantVillage dataset directory."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATA_CANDIDATES[0]


def ensure_project_dirs() -> None:
    """Create the standard output directories if they do not exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    XAI_DIR.mkdir(parents=True, exist_ok=True)
    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
