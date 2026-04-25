# Indian Legal AI: Court Judgment Prediction and Explanation (CJPE)

## Overview

This project presents a deep learning approach for predicting judicial outcomes from Indian court judgments. The task is formulated as binary text classification: given the full text of a case, the model predicts whether the judgment outcome is `Accepted` or `Rejected`.

The work is centered on the `CJPE` task from the `IL-TUR` dataset and explores the practical challenges of applying transformer-based models to long legal documents. A major focus of the project was not just model training, but diagnosing failure modes, improving class balance behavior, and adapting the input pipeline to the structure of real judgments.

## Project Objective

The core objective is to build a stable and interpretable deep learning pipeline for legal judgment prediction using domain-specific language models. In particular, the project aims to:

- model long-form legal documents with a transformer-based classifier
- reduce failure modes such as majority-class collapse
- evaluate the system with metrics that reflect class balance
- document the experimental process in a report-friendly form

## Task Definition

Each document is a court judgment, and the model predicts one of two labels:

- `Rejected = 0`
- `Accepted = 1`

This is a long-document classification problem, since many judgments are far longer than the standard 512-token context window of BERT-style models.

## Dataset

The project uses the `CJPE` configuration from the `IL-TUR` dataset. The official data split used in this work is:

- `single_train`: 5,082 cases
- `single_dev`: 2,511 cases
- `test`: 1,517 cases

Observed split characteristics in this project:

- the training split is moderately imbalanced toward `Rejected`
- the development and test splits are close to balanced
- judgments are very long, with average document lengths around 3,800 to 4,000 words
- some cases are extremely long, making truncation strategy an important modeling decision

## Why This Problem Is Challenging

Legal judgment prediction is difficult for several reasons:

- judgments are long and exceed the native context size of standard transformer encoders
- critical decision cues often appear near the end of the document
- the language is formal, domain-specific, and procedurally dense
- naive fine-tuning can lead to class imbalance issues and collapsed predictions

These challenges make legal NLP different from short-text classification tasks and require deliberate choices in preprocessing, model setup, and evaluation.

## Model And Methodology

The final model uses `law-ai/InLegalBERT` as the backbone. This model is particularly suitable for the task because it is pretrained on Indian legal text and therefore offers a stronger domain match than a general-purpose BERT encoder.

### Final Modeling Choices

- Backbone: `InLegalBERT`
- Output layer: sequence classification head for binary prediction
- Max sequence length: `512`
- Document representation: `head_tail`
- Loss: class-weighted cross-entropy
- Model selection criterion: development-set macro-F1
- Test-time decision rule: threshold tuned on the development set

### Long-Document Handling

One of the most important design decisions in this project was changing how long judgments are represented.

An initial straightforward setup used only the first 512 tokens of each judgment. That approach discarded most of the document and often missed the concluding legal reasoning and outcome cues near the end.

To address this, the final system uses a `head_tail` strategy:

- the beginning of the judgment is preserved to retain factual and procedural context
- the end of the judgment is preserved to capture outcome-oriented reasoning and final orders

This was a lightweight but effective alternative to implementing a heavier hierarchical or multi-chunk architecture within the available time budget.

### Training Strategy

The training pipeline includes class-weighted loss to reduce bias toward the majority class in the training split. This was especially important because an early run showed severe class collapse, where the model predicted almost entirely one class and produced `F1_accepted = 0`.

To improve stability, the final pipeline also:

- selects the best checkpoint based on development macro-F1
- tracks per-class F1 during validation
- logs the predicted acceptance rate during training
- tunes the final decision threshold on the development set before testing

## Experimental Progression

The project evolved through a sequence of practical improvements rather than a single successful run.

### Initial Failure Mode

An earlier run exhibited class collapse:

- the model overwhelmingly predicted `Rejected`
- `F1_rejected` remained non-trivial
- `F1_accepted` dropped to `0`

This indicated that raw fine-tuning with naive truncation was not sufficient for the task.

### Corrective Changes

The final pipeline addressed this through three key changes:

- retaining class-weighted cross-entropy
- replacing naive first-window truncation with `head_tail` document encoding
- applying development-set threshold tuning before final test prediction

These changes substantially improved class balance behavior and produced meaningful performance on both labels.

## Evaluation Metrics

The main evaluation metric is macro-F1 because it treats both classes equally and is more informative than raw accuracy when class behavior matters.

Reported metrics include:

- Accuracy
- Macro-F1
- F1 for `Accepted`
- F1 for `Rejected`
- Precision (macro)
- Recall (macro)
- ROC-AUC

## Final Results

Final held-out test performance:

- Accuracy: `0.6506`
- Macro-F1: `0.6482`
- F1 (Accepted): `0.6187`
- F1 (Rejected): `0.6776`
- Precision (macro): `0.6556`
- Recall (macro): `0.6510`
- ROC-AUC: `0.7074`

The final decision threshold chosen from the development set was `0.10`.

## Result Interpretation

The final outcome is best interpreted as a stable deep learning result for legal judgment prediction rather than a collapsed or degenerate classifier.

The most important takeaways are:

- the model successfully predicts both classes
- the earlier collapse issue was resolved
- long-document representation had a meaningful effect on performance
- class-aware training and threshold selection materially improved evaluation quality

From a project perspective, the result is valuable because it demonstrates an end-to-end legal NLP pipeline, surfaces realistic training challenges, and shows a clear improvement path from failure analysis to a working final system.

## Limitations

This project still has several limitations:

- the model only uses a lightweight `head_tail` strategy rather than full multi-chunk reasoning
- very long judgments are still compressed into a limited representation
- probability calibration remains imperfect, as seen from the low optimal decision threshold
- the work focuses on prediction and evaluation, not detailed legal explanation generation

These limitations suggest several natural directions for future improvement.

## Future Work

Possible next steps include:

- multi-chunk or hierarchical transformer architectures for long judgments
- attention-based aggregation over multiple document segments
- calibration-aware post-processing
- stronger explanation components aligned with legal reasoning sections
- error analysis by case type, judgment length, or court-specific language patterns

## Repository Structure

- `src/`: source code for data loading, modeling, training, evaluation, and utilities
- `data/`: local dataset placeholders and processed-data directories
- `outputs/`: checkpoints, plots, metrics, and generated summaries
- `requirements.txt`: project dependencies

Generated artifacts such as checkpoints, processed datasets, caches, and plots are excluded from version control.

## Files Of Interest

The most important project files are:

- [src/config.py](/Users/harditya_ghuman/Desktop/CJPE/legal_cjpe/src/config.py): central configuration and hyperparameters
- [src/data_loader.py](/Users/harditya_ghuman/Desktop/CJPE/legal_cjpe/src/data_loader.py): dataset loading, preprocessing, and tokenization
- [src/model.py](/Users/harditya_ghuman/Desktop/CJPE/legal_cjpe/src/model.py): model loading and initialization
- [src/train.py](/Users/harditya_ghuman/Desktop/CJPE/legal_cjpe/src/train.py): fine-tuning pipeline and weighted-loss training
- [src/evaluate.py](/Users/harditya_ghuman/Desktop/CJPE/legal_cjpe/src/evaluate.py): threshold tuning, test evaluation, and result reporting
- [src/run_all.py](/Users/harditya_ghuman/Desktop/CJPE/legal_cjpe/src/run_all.py): end-to-end orchestration script

## Summary

This project demonstrates a practical deep learning workflow for Indian legal judgment prediction using a domain-specific transformer. The final system does not simply report a score; it documents a meaningful modeling journey from failure diagnosis to a stable and defensible legal NLP result.
