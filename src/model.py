# src/model.py
"""
Model definition: InLegalBERT for sequence classification.
InLegalBERT is pretrained on 5.4 million Indian legal documents
by IIT Kharagpur — far better domain fit than vanilla BERT.
Paper: https://arxiv.org/abs/2209.06049
HuggingFace: https://huggingface.co/law-ai/InLegalBERT
"""

from transformers import AutoModelForSequenceClassification
import config


def load_model(from_checkpoint: str = None):
    """
    Load InLegalBERT for binary sequence classification.

    Args:
        from_checkpoint: path to a saved checkpoint directory.
                         If None, loads pretrained InLegalBERT.
    Returns:
        model: ready for training or inference
    """
    local_best = config.OUT_CKPT / "best_model"
    source = from_checkpoint if from_checkpoint else config.MODEL_NAME
    if from_checkpoint is None and (local_best / "model.safetensors").exists():
        print(f"[model] Local checkpoint detected at {local_best}; using it as initialization source.")
        source = str(local_best)

    print(f"[model] Loading model from: {source}")

    model = AutoModelForSequenceClassification.from_pretrained(
        source,
        num_labels=config.NUM_LABELS,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
        ignore_mismatched_sizes=True
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Total parameters    : {total_params:,}")
    print(f"[model] Trainable parameters: {trainable_params:,}")

    return model


def count_parameters(model) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = load_model()
    params = count_parameters(model)
    print(f"\nModel architecture summary:")
    print(f"  Model name    : {config.MODEL_NAME}")
    print(f"  Max seq length: {config.MAX_LEN}")
    print(f"  Num labels    : {config.NUM_LABELS}")
    print(f"  Labels        : {config.ID2LABEL}")
    print(f"  Total params  : {params['total']:,} (~{params['total']/1e6:.0f}M)")
