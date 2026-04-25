# src/explain.py
"""
Qualitative analysis on the 56 expert-annotated CJPE documents.
This is the 'explanation' sub-task and gives your report a strong
qualitative section with zero extra training.

Produces:
  - outputs/results/expert_analysis.json     ← per-doc predictions vs expert labels
  - outputs/results/expert_summary.json      ← aggregate stats
  - outputs/figures/expert_vs_model.png      ← agreement chart
"""

import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config
from data_loader import download_dataset


def load_fine_tuned(model_path: str = None):
    path = model_path or str(config.OUT_CKPT / "best_model")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    device = config.DEVICE
    model.to(device)
    return model, tokenizer, device


def predict_single(text: str, model, tokenizer, device) -> dict:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LEN,
        padding=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred  = int(np.argmax(probs))
    return {
        "prediction":    pred,
        "label_text":    config.ID2LABEL[pred],
        "confidence":    round(float(probs[pred]), 4),
        "prob_rejected": round(float(probs[0]), 4),
        "prob_accepted": round(float(probs[1]), 4),
    }


def run_expert_analysis():
    print("\n[explain] Running qualitative analysis on expert split (56 docs)...")

    raw_ds   = download_dataset()
    expert_ds = raw_ds[config.EXPERT_SPLIT]
    model, tokenizer, device = load_fine_tuned()

    results     = []
    correct     = 0
    agreements  = {"all_agree": 0, "majority_agree": 0}

    for sample in expert_ds:
        pred_info = predict_single(sample["text"], model, tokenizer, device)
        true_label = sample["label"]
        is_correct = (pred_info["prediction"] == true_label)

        # Collect expert label votes (5 experts, each with their own label)
        expert_votes = []
        for i in range(1, 6):
            key = f"expert_{i}"
            if key in sample and sample[key] is not None:
                expert_label = sample[key].get("label")
                if expert_label is not None:
                    expert_votes.append(int(expert_label))

        # Expert agreement
        model_agrees_all    = all(v == pred_info["prediction"] for v in expert_votes)
        model_agrees_majority = (
            expert_votes.count(pred_info["prediction"]) > len(expert_votes) / 2
            if expert_votes else False
        )

        if model_agrees_all:
            agreements["all_agree"] += 1
        if model_agrees_majority:
            agreements["majority_agree"] += 1

        if is_correct:
            correct += 1

        result_entry = {
            "id":             sample["id"],
            "true_label":     int(true_label),
            "true_label_text": config.ID2LABEL[int(true_label)],
            "expert_votes":   expert_votes,
            "model_prediction": pred_info["prediction"],
            "model_label_text": pred_info["label_text"],
            "model_confidence": pred_info["confidence"],
            "is_correct":     is_correct,
            "agrees_all_experts": model_agrees_all,
            "agrees_majority_experts": model_agrees_majority,
        }
        results.append(result_entry)

    total = len(results)
    accuracy = correct / total

    summary = {
        "total_docs":             total,
        "model_correct":          correct,
        "model_accuracy":         round(accuracy, 4),
        "agrees_all_experts":     agreements["all_agree"],
        "agrees_majority_experts": agreements["majority_agree"],
        "pct_agrees_all":         round(agreements["all_agree"] / total, 4),
        "pct_agrees_majority":    round(agreements["majority_agree"] / total, 4),
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EXPERT SPLIT ANALYSIS")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k:<35}: {v}")
    print("="*60)

    # Print individual predictions
    print("\nPer-document predictions:")
    print(f"  {'ID':<15} {'True':>10} {'Pred':>10} {'Conf':>8}  {'OK?':>5}")
    print("  " + "-"*55)
    for r in results:
        ok = "✓" if r["is_correct"] else "✗"
        print(
            f"  {r['id']:<15} {r['true_label_text']:>10} {r['model_label_text']:>10} "
            f"{r['model_confidence']:>8.4f}  {ok:>5}"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    with open(config.OUT_RES / "expert_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(config.OUT_RES / "expert_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[explain] Expert analysis saved.")

    # ── Plot ─────────────────────────────────────────────────────────────────
    labels  = ["Correct\nPredictions", "Incorrect\nPredictions"]
    values  = [correct, total - correct]
    colors  = ["steelblue", "tomato"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, values, color=colors, width=0.4, edgecolor="white")
    axes[0].set_title(f"Model Accuracy on Expert Split\n(n={total})", fontsize=12)
    axes[0].set_ylabel("Count")
    for i, v in enumerate(values):
        axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    agree_labels  = ["Agrees All\n5 Experts", "Agrees\nMajority", "Total\nCorrect"]
    agree_values  = [agreements["all_agree"], agreements["majority_agree"], correct]
    agree_colors  = ["mediumseagreen", "darkorange", "steelblue"]

    axes[1].bar(agree_labels, agree_values, color=agree_colors, width=0.4, edgecolor="white")
    axes[1].set_title("Model vs Expert Agreement", fontsize=12)
    axes[1].set_ylabel("Count (out of 56)")
    for i, v in enumerate(agree_values):
        axes[1].text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    axes[1].set_ylim(0, total + 5)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.OUT_FIG / "expert_vs_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[explain] Expert agreement chart saved.")

    return summary


if __name__ == "__main__":
    run_expert_analysis()
