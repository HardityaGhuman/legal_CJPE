# src/run_all.py
"""
Single entry point. Runs the full pipeline end-to-end:
  1. Download + inspect dataset
  2. Tokenize + save to disk
  3. Train InLegalBERT
  4. Evaluate on test set
  5. Print final summary

Usage:
    cd legal_cjpe/src
    python run_all.py
"""

import json
import time

import config
from data_loader import download_dataset, inspect_dataset, tokenize_dataset
from train import run_training
from evaluate import run_evaluation


def print_banner(text: str) -> None:
    width = 70
    print("\n" + "█" * width)
    print(f"  {text}")
    print("█" * width)


def main():
    total_start = time.time()

    print_banner("INDIAN LEGAL AI — CJPE PIPELINE")
    print(f"  Model   : {config.MODEL_NAME}")
    print(f"  Device  : {config.DEVICE}")
    print(f"  Epochs  : {config.EPOCHS}")
    print(f"  MaxLen  : {config.MAX_LEN}")
    print(f"  Batch   : {config.TRAIN_BATCH_SIZE}")
    print(f"  LR      : {config.LEARNING_RATE}")

    # ── STEP 1: Dataset ───────────────────────────────────────────────────────
    print_banner("STEP 1/4 — DATASET DOWNLOAD & INSPECTION")
    raw_ds = download_dataset()
    inspect_dataset(raw_ds)

    # ── STEP 2: Tokenize ──────────────────────────────────────────────────────
    print_banner("STEP 2/4 — TOKENIZATION")
    tokenized = tokenize_dataset(raw_ds)
    print(f"[run_all] Tokenized splits: {list(tokenized.keys())}")

    # ── STEP 3: Train ─────────────────────────────────────────────────────────
    print_banner("STEP 3/4 — TRAINING")
    t_start = time.time()
    trainer = run_training(raw_ds=raw_ds, tokenized=tokenized)
    t_train = round(time.time() - t_start, 1)
    print(f"\n[run_all] Training completed in {t_train:.0f}s ({t_train/60:.1f} min)")

    # ── STEP 4: Evaluate ──────────────────────────────────────────────────────
    print_banner("STEP 4/4 — TEST SET EVALUATION")
    test_metrics = run_evaluation(tokenized=tokenized)

    # ── Final summary ─────────────────────────────────────────────────────────
    total_time = round(time.time() - total_start, 1)

    print_banner("PIPELINE COMPLETE — FINAL SUMMARY")
    print(f"\n  Device              : {config.DEVICE}")
    print(f"  Total runtime       : {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"\n  TEST SET METRICS:")
    print(f"    Accuracy          : {test_metrics['accuracy']:.4f}")
    print(f"    F1-Macro          : {test_metrics['f1_macro']:.4f}")
    print(f"    F1 (Accepted)     : {test_metrics['f1_accepted']:.4f}")
    print(f"    F1 (Rejected)     : {test_metrics['f1_rejected']:.4f}")
    print(f"    ROC-AUC           : {test_metrics['roc_auc']:.4f}")
    print(f"\n  OUTPUTS SAVED TO:")
    print(f"    Checkpoints       : {config.OUT_CKPT}")
    print(f"    Results (JSON)    : {config.OUT_RES}")
    print(f"    Figures (PNG)     : {config.OUT_FIG}")
    print(f"    Logs              : {config.OUT_LOG}")

    # Save final combined summary
    final = {
        "test_metrics":    test_metrics,
        "runtime_seconds": total_time,
        "config": {
            "model":      config.MODEL_NAME,
            "epochs":     config.EPOCHS,
            "max_len":    config.MAX_LEN,
            "batch_size": config.TRAIN_BATCH_SIZE,
            "lr":         config.LEARNING_RATE,
            "device":     config.DEVICE,
        }
    }
    with open(config.OUT_RES / "final_summary.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\n  Final summary     : {config.OUT_RES / 'final_summary.json'}")
    print("\n" + "█" * 70)


if __name__ == "__main__":
    main()
