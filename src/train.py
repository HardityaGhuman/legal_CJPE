# src/train.py
"""
Fine-tunes InLegalBERT on CJPE task.
Saves best checkpoint (by F1-macro on dev set) to outputs/checkpoints/.
Saves training history to outputs/results/training_history.json.
Saves loss curve plot to outputs/figures/training_curves.png.
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score

import config
from data_loader import download_dataset, tokenize_dataset, get_tokenizer
from model import load_model


def compute_metrics(eval_pred):
    """
    Called by HuggingFace Trainer after each validation step.
    Returns dict of metrics to log.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc        = accuracy_score(labels, preds)
    f1_macro   = f1_score(labels, preds, average="macro", zero_division=0)
    f1_accepted = f1_score(labels, preds, pos_label=1, zero_division=0)
    f1_rejected = f1_score(labels, preds, pos_label=0, zero_division=0)
    pred_accept_rate = float(np.mean(preds == 1))
    true_accept_rate = float(np.mean(labels == 1))

    return {
        "accuracy":     round(float(acc), 4),
        "f1_macro":     round(float(f1_macro), 4),
        "f1_accepted":  round(float(f1_accepted), 4),
        "f1_rejected":  round(float(f1_rejected), 4),
        "pred_accept_rate": round(pred_accept_rate, 4),
        "true_accept_rate": round(true_accept_rate, 4),
    }


class WeightedLossTrainer(Trainer):
    """
    Trainer with class-weighted cross-entropy to reduce majority-class collapse.
    """
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def plot_training_curves(log_history: list, save_path) -> None:
    """
    Parse Trainer log_history and save loss + f1 curves as PNG.
    """
    eval_loss, eval_f1 = [], []

    for entry in log_history:
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_f1.append(entry.get("eval_f1_macro", 0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curve
    axes[0].plot(eval_loss, marker="o", color="steelblue", label="Eval Loss")
    axes[0].set_title("Validation Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 curve
    axes[1].plot(eval_f1, marker="s", color="darkorange", label="Eval F1-Macro")
    axes[1].set_title("Validation F1-Macro per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-Macro")
    axes[1].axhline(y=0.78, color="red", linestyle="--", alpha=0.7, label="Published SOTA (0.78)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[train] Training curves saved to {save_path}")


def run_training(raw_ds=None, tokenized=None):
    # ── Step 1: Load and prepare data ────────────────────────────────────────
    raw_ds = raw_ds or download_dataset()
    tokenized = tokenized or tokenize_dataset(raw_ds)
    tokenizer = get_tokenizer()

    train_ds = tokenized[config.TRAIN_SPLIT]
    dev_ds   = tokenized[config.DEV_SPLIT]

    print(f"\n[train] Train size : {len(train_ds)}")
    print(f"[train] Dev size   : {len(dev_ds)}")
    print(f"[train] Doc strategy: {config.DOC_STRATEGY} | Max length: {config.MAX_LEN}")

    # Compute inverse-frequency class weights (balanced loss).
    # This directly penalizes ignoring the minority class.
    train_labels = np.array(train_ds["labels"])
    class_counts = np.bincount(train_labels, minlength=config.NUM_LABELS)
    class_weights = len(train_labels) / (config.NUM_LABELS * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(
        f"[train] Label counts: Rejected={class_counts[0]} | Accepted={class_counts[1]} "
        f"| Class weights={class_weights.tolist()}"
    )

    # ── Step 2: Load model ───────────────────────────────────────────────────
    model = load_model()

    # ── Step 3: Training arguments ───────────────────────────────────────────
    # Transformers >=5 uses `eval_strategy` (not `evaluation_strategy`).
    ta_kwargs = dict(
        output_dir=str(config.OUT_CKPT),

        num_train_epochs=config.EPOCHS,

        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,

        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,

        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        logging_dir=str(config.OUT_LOG),
        logging_steps=50,
        logging_first_step=True,

        save_total_limit=2,        # Keep only best + last checkpoint on disk
        seed=config.SEED,
        data_seed=config.SEED,
        dataloader_num_workers=config.DATALOADER_WORKERS,
        dataloader_pin_memory=(config.DEVICE != "mps"),
        report_to="none",          # Disable W&B / tensorboard
        fp16=False,                # MPS does not support fp16
        bf16=False,
    )

    try:
        training_args = TrainingArguments(
            **ta_kwargs,
            eval_strategy="epoch",
        )
    except TypeError:
        # Backward compatibility with older transformers
        training_args = TrainingArguments(
            **ta_kwargs,
            evaluation_strategy="epoch",
        )

    # ── Step 4: Trainer ──────────────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE)],
        class_weights=class_weights,
    )

    # ── Step 5: Train ────────────────────────────────────────────────────────
    print(f"\n[train] Starting training on device: {config.DEVICE}")
    print(f"[train] Epochs: {config.EPOCHS} | Batch: {config.TRAIN_BATCH_SIZE} | LR: {config.LEARNING_RATE}\n")

    train_result = trainer.train()

    # ── Step 6: Save training history ────────────────────────────────────────
    history_path = config.OUT_RES / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"[train] Training history saved to {history_path}")

    # ── Step 7: Plot curves ───────────────────────────────────────────────────
    plot_training_curves(
        trainer.state.log_history,
        config.OUT_FIG / "training_curves.png"
    )

    # ── Step 8: Save final model ─────────────────────────────────────────────
    final_model_path = config.OUT_CKPT / "best_model"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    print(f"[train] Best model saved to {final_model_path}")

    # ── Step 9: Save train summary ────────────────────────────────────────────
    summary = {
        "model": config.MODEL_NAME,
        "train_split": config.TRAIN_SPLIT,
        "train_size": len(train_ds),
        "dev_size": len(dev_ds),
        "epochs": config.EPOCHS,
        "batch_size": config.TRAIN_BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "max_len": config.MAX_LEN,
        "device": config.DEVICE,
        "train_runtime_sec": round(train_result.metrics.get("train_runtime", 0), 1),
        "train_samples_per_sec": round(train_result.metrics.get("train_samples_per_second", 0), 2),
    }
    with open(config.OUT_RES / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] Train summary saved.")

    return trainer


if __name__ == "__main__":
    run_training()
