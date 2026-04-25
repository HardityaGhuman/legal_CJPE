# Indian Legal AI — Court Judgment Prediction & Explanation (CJPE)

This project is implemented exactly per `../INDIAN_LEGAL_AI_CJPE.md`.

## Quickstart

1. Create + activate the virtual environment from the workspace root:
   - `python3 -m venv legal_ai_env`
   - `source legal_ai_env/bin/activate`

2. Install deps:
   - `pip install -r legal_cjpe/requirements.txt`

3. Run full pipeline:
   - `cd legal_cjpe/src`
   - `python run_all.py`

## What The Pipeline Does

`run_all.py` performs the full workflow end-to-end:

1. Loads the IL-TUR `cjpe` splits.
2. Prints dataset statistics once.
3. Tokenizes documents and caches the processed splits on disk.
4. Fine-tunes InLegalBERT for binary judgment prediction.
5. Evaluates on the held-out test split and saves plots + JSON outputs.

## Current Modeling Setup

- Base model: `law-ai/InLegalBERT`
- Task: binary classification (`Rejected=0`, `Accepted=1`)
- Input strategy: `head_tail`
- Max sequence length: `512`
- Training objective: class-weighted cross-entropy
- Model selection: best checkpoint by dev `f1_macro`
- Test prediction rule: threshold tuned on the dev split

The `head_tail` strategy keeps both the opening portion of the judgment and the ending portion, which works better for long court documents than using only the first 512 tokens.

If a local checkpoint exists under `outputs/checkpoints/best_model`, the code prefers that tokenizer/model for reproducibility and offline-friendly reruns.

## Outputs

After a successful run, artifacts are written under:

- `outputs/checkpoints/`
- `outputs/results/`
- `outputs/figures/`
- `outputs/logs/`

Key result files include:

- `outputs/results/test_metrics.json`
- `outputs/results/final_summary.json`
- `outputs/results/benchmark_table.json`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/confidence_histogram.png`
