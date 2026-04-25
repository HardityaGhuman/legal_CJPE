# src/evaluate.py
"""
Full evaluation on held-out test set.
Produces:
  - outputs/results/test_metrics.json        ← all numeric metrics
  - outputs/figures/confusion_matrix.png     ← confusion matrix heatmap
  - outputs/figures/confidence_histogram.png ← prediction confidence distribution
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from transformers import Trainer, DataCollatorWithPadding, TrainingArguments

import config
from data_loader import tokenize_dataset, download_dataset, get_tokenizer
from model import load_model
from train import compute_metrics


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def find_best_threshold(y_true: np.ndarray, y_prob_pos: np.ndarray) -> tuple:
    """
    Select decision threshold on dev set using macro-F1.
    Returns (best_threshold, best_macro_f1).
    """
    thresholds = np.arange(0.1, 0.91, 0.02)
    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        y_pred = (y_prob_pos >= t).astype(int)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1m > best_f1:
            best_f1 = f1m
            best_t = float(t)
    return best_t, float(best_f1)


def plot_confusion_matrix(y_true, y_pred, save_path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Rejected (0)", "Accepted (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Confusion Matrix — CJPE Test Set", fontsize=13, pad=12)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved to {save_path}")


def plot_confidence_histogram(y_prob, y_true, save_path) -> None:
    """
    Histogram of model confidence on correct vs incorrect predictions.
    Useful for showing calibration in report.
    """
    max_conf   = np.max(y_prob, axis=1)
    y_pred     = np.argmax(y_prob, axis=1)
    correct    = max_conf[y_pred == y_true]
    incorrect  = max_conf[y_pred != y_true]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(correct,   bins=20, alpha=0.7, color="steelblue",  label=f"Correct (n={len(correct)})")
    ax.hist(incorrect, bins=20, alpha=0.7, color="tomato",     label=f"Incorrect (n={len(incorrect)})")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.6, label="0.5 threshold")
    ax.set_title("Prediction Confidence Distribution", fontsize=13)
    ax.set_xlabel("Max Softmax Probability", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confidence histogram saved to {save_path}")


def run_evaluation(best_model_path: str = None, tokenized=None):
    """
    Full evaluation pipeline on test set.
    Call this after training is complete.
    """
    model_path = best_model_path or str(config.OUT_CKPT / "best_model")

    # ── Load data ─────────────────────────────────────────────────────────────
    tokenized = tokenized or tokenize_dataset(download_dataset())
    tokenizer = get_tokenizer()
    dev_ds    = tokenized[config.DEV_SPLIT]
    test_ds   = tokenized[config.TEST_SPLIT]

    print(f"\n[evaluate] Test set size: {len(test_ds)}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(from_checkpoint=model_path)

    # ── Run inference ─────────────────────────────────────────────────────────
    eval_args = TrainingArguments(
        output_dir=str(config.OUT_CKPT),
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        report_to="none",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print("[evaluate] Tuning threshold on dev set...")
    dev_output = trainer.predict(dev_ds)
    dev_prob = softmax(dev_output.predictions)
    dev_labels = dev_output.label_ids
    best_threshold, best_dev_f1_macro = find_best_threshold(dev_labels, dev_prob[:, 1])
    print(
        f"[evaluate] Best threshold from dev: {best_threshold:.2f} "
        f"(dev f1_macro={best_dev_f1_macro:.4f})"
    )

    print("[evaluate] Running inference on test set...")
    output = trainer.predict(test_ds)

    logits = output.predictions
    labels = output.label_ids
    y_prob = softmax(logits)
    y_pred = (y_prob[:, 1] >= best_threshold).astype(int)

    # ── Compute metrics ───────────────────────────────────────────────────────
    metrics = {
        "accuracy":         round(float(accuracy_score(labels, y_pred)), 4),
        "f1_macro":         round(float(f1_score(labels, y_pred, average="macro")), 4),
        "f1_micro":         round(float(f1_score(labels, y_pred, average="micro")), 4),
        "f1_accepted":      round(float(f1_score(labels, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_rejected":      round(float(f1_score(labels, y_pred, pos_label=0, zero_division=0)), 4),
        "precision_macro":  round(float(precision_score(labels, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro":     round(float(recall_score(labels, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc":          round(float(roc_auc_score(labels, y_prob[:, 1])), 4),
        "decision_threshold": round(float(best_threshold), 4),
        "dev_f1_macro_at_threshold": round(float(best_dev_f1_macro), 4),
        "test_size":        len(test_ds),
    }

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k:<25}: {v}")
    print("\nPer-class Report:")
    print(classification_report(labels, y_pred, target_names=["Rejected", "Accepted"]))

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(config.OUT_RES / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[evaluate] Metrics saved to {config.OUT_RES / 'test_metrics.json'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, y_pred, config.OUT_FIG / "confusion_matrix.png")
    plot_confidence_histogram(y_prob, labels, config.OUT_FIG / "confidence_histogram.png")

    return metrics


if __name__ == "__main__":
    run_evaluation()
