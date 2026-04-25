# src/data_loader.py
"""
Downloads IL-TUR CJPE dataset from HuggingFace,
inspects it, tokenizes, and saves processed splits to disk.
"""

import json
from collections import Counter

from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

import config


def _get_tokenizer_source() -> str:
    """
    Prefer a local checkpoint tokenizer when available so tokenization stays
    reproducible and can run without network access.
    """
    local_best = config.OUT_CKPT / "best_model"
    if (local_best / "tokenizer.json").exists():
        print(f"[data_loader] Using local tokenizer from {local_best}")
        return str(local_best)
    return config.MODEL_NAME


def _normalize_label_column(dataset: DatasetDict) -> DatasetDict:
    """
    Ensure label values are numeric (Rejected=0, Accepted=1).
    """
    def map_label(example):
        label = example["label"]
        if isinstance(label, str):
            key = label.strip().title()
            if key in config.LABEL2ID:
                example["label"] = config.LABEL2ID[key]
            elif label.strip().upper() == "ACCEPTED":
                example["label"] = 1
            elif label.strip().upper() == "REJECTED":
                example["label"] = 0
        return example

    normalized = {}
    for split, ds in dataset.items():
        if "label" in ds.column_names:
            normalized[split] = ds.map(map_label, desc=f"Normalizing labels ({split})")
        else:
            normalized[split] = ds
    return DatasetDict(normalized)


def download_dataset() -> DatasetDict:
    """
    Load IL-TUR CJPE dataset.

    Preferred path: load from local HuggingFace cache files already present under
    `data/raw/Exploration-Lab___il-tur/cjpe/**`.

    Fallback: if cache files aren't found, call `load_dataset` to download.

    Returns DatasetDict with train, dev, test splits only (skips gated expert split).
    """
    print(f"[data_loader] Loading {config.DATASET_NAME} / cjpe from cache...")

    cache_root = config.DATA_RAW / "Exploration-Lab___il-tur" / "cjpe"

    # Newer HF datasets cache uses Arrow files, not parquet. Your cache contains e.g.
    # `il-tur-single_train.arrow`, `il-tur-single_dev.arrow`, `il-tur-test.arrow`.
    arrow_mapping = {
        config.TRAIN_SPLIT: "*single_train*.arrow",
        config.DEV_SPLIT: "*single_dev*.arrow",
        config.TEST_SPLIT: "*test*.arrow",
    }

    splits_dict = {}
    for split_key, pattern in arrow_mapping.items():
        arrow_files = sorted(cache_root.glob(f"**/{pattern}"))
        if not arrow_files:
            continue
        src = arrow_files[0]
        rel = src
        try:
            rel = src.relative_to(config.ROOT)
        except Exception:
            pass
        print(f"[data_loader] Loading {split_key} from {rel}...")
        # Arrow cache shards are themselves dataset files.
        splits_dict[split_key] = load_dataset("arrow", data_files=str(src))["train"]

    if len(splits_dict) == 3:
        dataset = DatasetDict(splits_dict)
        print(f"[data_loader] Loaded splits: {list(dataset.keys())}")
        return dataset

    # Fallback A: rebuild from locally downloaded JSONL files in HF downloads cache.
    # This avoids hard-failing when optional expert files are unavailable.
    downloads_dir = config.DATA_RAW / "downloads"
    split_to_filename = {
        config.TRAIN_SPLIT: "single_train.jsonl",
        config.DEV_SPLIT: "single_dev.jsonl",
        config.TEST_SPLIT: "test.jsonl",
    }
    local_data_files = {}
    if downloads_dir.exists():
        for meta_path in downloads_dir.glob("*.json"):
            try:
                meta = json.loads(meta_path.read_text())
                url = meta.get("url", "")
                filename = url.rsplit("/", 1)[-1] if url else ""
                data_path = meta_path.with_suffix("")
                if filename in split_to_filename.values() and data_path.exists():
                    local_data_files[filename] = str(data_path)
            except Exception:
                continue

    reverse_lookup = {v: k for k, v in split_to_filename.items()}
    mapped_local = {
        reverse_lookup[name]: path
        for name, path in local_data_files.items()
        if name in reverse_lookup
    }
    if len(mapped_local) == 3:
        print("[data_loader] Rebuilding splits from local downloads cache...")
        rebuilt = load_dataset("json", data_files=mapped_local)
        rebuilt = _normalize_label_column(DatasetDict(rebuilt))
        print(f"[data_loader] Loaded splits: {list(rebuilt.keys())}")
        return rebuilt

    # Fallback B: download via datasets (will also populate cache)
    print("[data_loader] Local cache missing some splits; falling back to HuggingFace download...")
    ds = load_dataset(config.DATASET_NAME, "cjpe")

    # Map expected split keys used in this project to dataset splits.
    mapped = DatasetDict({
        config.TRAIN_SPLIT: ds.get("single_train") or ds.get("train"),
        config.DEV_SPLIT: ds.get("single_dev") or ds.get("validation") or ds.get("dev"),
        config.TEST_SPLIT: ds.get("test"),
    })

    # Drop any missing splits
    mapped = DatasetDict({k: v for k, v in mapped.items() if v is not None})

    missing = [k for k in [config.TRAIN_SPLIT, config.DEV_SPLIT, config.TEST_SPLIT] if k not in mapped]
    if missing:
        raise FileNotFoundError(
            "Could not load required splits (" + ", ".join(missing) + ") from local cache or HuggingFace. "
            "Check that the dataset config contains single_train/single_dev/test."
        )

    mapped = _normalize_label_column(mapped)
    print(f"[data_loader] Loaded splits: {list(mapped.keys())}")
    return mapped


def inspect_dataset(dataset: DatasetDict) -> None:
    """
    Print dataset statistics for EDA / report Section 4.1.
    """
    splits_to_show = [
        config.TRAIN_SPLIT,
        config.DEV_SPLIT,
        config.TEST_SPLIT,
    ]

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    for split in splits_to_show:
        if split not in dataset:
            print(f"  Split '{split}' not found, skipping.")
            continue

        ds = dataset[split]
        labels = [ex["label"] for ex in ds]
        counter = Counter(labels)
        total = len(ds)

        print(f"\n  Split: {split}")
        print(f"    Total docs  : {total}")
        print(f"    Accepted (1): {counter.get(1, 0)} ({counter.get(1, 0)/total*100:.1f}%)")
        print(f"    Rejected (0): {counter.get(0, 0)} ({counter.get(0, 0)/total*100:.1f}%)")

        # Text length stats (in whitespace tokens)
        lengths = [len(ex["text"].split()) for ex in ds]
        print(f"    Avg doc len : {sum(lengths)//len(lengths)} words")
        print(f"    Max doc len : {max(lengths)} words")
        print(f"    Min doc len : {min(lengths)} words")

    print("\n  Sample from training split:")
    sample = dataset[config.TRAIN_SPLIT][0]
    print(f"    ID    : {sample['id']}")
    print(f"    Label : {config.ID2LABEL[sample['label']]}")
    print(f"    Text  : {sample['text'][:300]}...")
    print("="*60 + "\n")

    # Save stats to results folder
    stats = {}
    for split in splits_to_show:
        if split in dataset:
            ds = dataset[split]
            labels = [ex["label"] for ex in ds]
            counter = Counter(labels)
            stats[split] = {
                "total": len(ds),
                "accepted": counter.get(1, 0),
                "rejected": counter.get(0, 0)
            }
    with open(config.OUT_RES / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[data_loader] Stats saved to {config.OUT_RES / 'dataset_stats.json'}")


def tokenize_dataset(dataset: DatasetDict) -> DatasetDict:
    """
    Tokenize train, dev, and test splits.
    Saves tokenized datasets to disk for fast reload.
    """
    save_path = config.DATA_PROC / f"tokenized_cjpe_{config.DOC_STRATEGY}_{config.MAX_LEN}"

    if save_path.exists():
        print(f"[data_loader] Tokenized data found at {save_path}. Loading from disk...")
        return load_from_disk(str(save_path))

    tokenizer_source = _get_tokenizer_source()
    print(f"[data_loader] Loading tokenizer: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    def tokenize_fn(examples):
        if config.DOC_STRATEGY == "head_tail":
            encoded = tokenizer(
                examples["text"],
                add_special_tokens=False,
                truncation=False,
                padding=False,
            )

            cls_id = tokenizer.cls_token_id
            sep_id = tokenizer.sep_token_id
            input_ids = []
            attention_masks = []
            token_type_ids = []

            for ids in encoded["input_ids"]:
                head = ids[:config.HEAD_TOKENS]
                tail = ids[-config.TAIL_TOKENS:] if len(ids) > config.HEAD_TOKENS else []
                merged = [cls_id] + head + [sep_id]
                if tail:
                    merged += tail + [sep_id]

                merged = merged[:config.MAX_LEN]
                input_ids.append(merged)
                attention_masks.append([1] * len(merged))
                token_type_ids.append([0] * len(merged))

            return {
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids,
            }

        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_LEN,
            padding=False,   # Dynamic padding via DataCollator saves memory
        )

    splits_to_tokenize = [
        config.TRAIN_SPLIT,
        config.DEV_SPLIT,
        config.TEST_SPLIT,
    ]

    tokenized = {}
    for split in splits_to_tokenize:
        if split not in dataset:
            continue
        print(f"[data_loader] Tokenizing split: {split}...")
        tok = dataset[split].map(
            tokenize_fn,
            batched=True,
            batch_size=16 if config.DOC_STRATEGY == "head_tail" else 64,
            desc=f"Tokenizing {split}",
            remove_columns=["id", "text"]   # Keep only model inputs + label
        )
        # Rename 'label' to 'labels' for HuggingFace Trainer compatibility
        if "label" in tok.column_names:
            tok = tok.rename_column("label", "labels")
        tok.set_format("torch")
        tokenized[split] = tok

    tokenized_ds = DatasetDict(tokenized)
    tokenized_ds.save_to_disk(str(save_path))
    print(f"[data_loader] Tokenized dataset saved to {save_path}")
    return tokenized_ds


def get_tokenizer():
    return AutoTokenizer.from_pretrained(_get_tokenizer_source())


if __name__ == "__main__":
    ds = download_dataset()
    inspect_dataset(ds)
    tokenized = tokenize_dataset(ds)
    print("[data_loader] Done. Splits ready:", list(tokenized.keys()))
