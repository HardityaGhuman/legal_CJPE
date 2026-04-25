# src/config.py
"""
Central configuration for the CJPE project.
All hyperparameters, paths, and constants defined here.
Never hardcode values elsewhere — always import from config.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).resolve().parent.parent   # legal_cjpe/
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
OUT_CKPT   = ROOT / "outputs" / "checkpoints"
OUT_RES    = ROOT / "outputs" / "results"
OUT_FIG    = ROOT / "outputs" / "figures"
OUT_LOG    = ROOT / "outputs" / "logs"

# Create directories if they don't exist
for p in [DATA_RAW, DATA_PROC, OUT_CKPT, OUT_RES, OUT_FIG, OUT_LOG]:
    p.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────

DATASET_NAME    = "Exploration-Lab/IL-TUR"

TRAIN_SPLIT = "single_train"   # 5,000 single-appeal cases
DEV_SPLIT   = "single_dev"     # 2,500
TEST_SPLIT  = "test"           # 1,500
EXPERT_SPLIT = "expert"        # 56 expert-annotated docs

LABEL2ID = {"Rejected": 0, "Accepted": 1}
ID2LABEL = {0: "Rejected", 1: "Accepted"}

# ── Model ─────────────────────────────────────────────────────────────────────

MODEL_NAME  = "law-ai/InLegalBERT"   # InLegalBERT: pretrained on 5.4M Indian legal docs
NUM_LABELS  = 2
MAX_LEN     = 512    # BERT max. Court docs are long; first 512 tokens capture facts.
DOC_STRATEGY = "head_tail"  # Preserve both opening facts and final decision cues.
HEAD_TOKENS = 255           # Leaves room for [CLS] + 2x[SEP] within MAX_LEN=512.
TAIL_TOKENS = 254

# ── Training Hyperparameters ──────────────────────────────────────────────────

EPOCHS              = 4       # Fits within ~3-4h budget on MPS while allowing convergence.
TRAIN_BATCH_SIZE    = 8       # Safe for M4 Air 16GB
EVAL_BATCH_SIZE     = 16
LEARNING_RATE       = 2e-5
WEIGHT_DECAY        = 0.01
WARMUP_RATIO        = 0.1
SEED                = 42
DATALOADER_WORKERS  = 0
EARLY_STOPPING_PATIENCE = 2

# ── Device ────────────────────────────────────────────────────────────────────

import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
