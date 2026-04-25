# Indian Legal NLP: Judgment Prediction Module (CJPE)

## Overview

This project implements a deep learning–based **judgment prediction module** for Indian legal documents. The task is framed as binary classification: given the text of a court judgment, the model predicts whether the outcome is **Accepted** or **Rejected**.

The work is based on the **CJPE (Court Judgment Prediction)** task from the **IL-TUR benchmark** and focuses on handling long legal documents, addressing class imbalance, and building a stable training pipeline.

---

## Task Definition

- Input: Full court judgment text  
- Output: Binary label  
  - `Rejected = 0`  
  - `Accepted = 1`  

This is a **long-document classification problem**, as judgments often exceed standard transformer input limits.

---

## Dataset

Dataset: **IL-TUR (CJPE split)**

- `single_train`: 5,082 cases (imbalanced)  
- `single_dev`: 2,511 cases (balanced)  
- `test`: 1,517 cases (balanced)  

### Key Characteristics
- Average length: ~3,800–4,000 words  
- Some documents exceed 40,000 words  
- Training data skewed toward `Rejected`

---

## Approach

### Model
- Backbone: `law-ai/InLegalBERT`  
- Domain-specific transformer pretrained on Indian legal text  
- Fine-tuned for binary classification  

### Long Document Handling

Standard truncation (first 512 tokens) loses critical information.  
This project uses a **head-tail strategy**:

- First ~255 tokens → factual and procedural context  
- Last ~254 tokens → reasoning and final order  

This preserves the most informative parts without modifying the model architecture.

### Training Setup

- Loss: class-weighted cross-entropy  
- Model selection: dev macro-F1  
- Threshold tuning: performed on dev set before testing  

---

## Key Design Decisions

Initial experiments showed **class collapse** (predicting mostly `Rejected`).  
The final pipeline resolves this through:

- head-tail document representation  
- class-weighted loss  
- decision threshold tuning  

These changes enable **balanced predictions across both classes**.

---

## Evaluation

**Primary metric:** Macro-F1  

### Additional Metrics
- Accuracy  
- F1 (Accepted / Rejected)  
- Precision (macro)  
- Recall (macro)  
- ROC-AUC  

---

## Results

Final test performance:

- Accuracy: **65.06%**  
- Macro-F1: **64.82%**  
- F1 (Accepted): **61.87%**  
- F1 (Rejected): **67.76%**  
- Precision (macro): **65.56%**  
- Recall (macro): **65.10%**  
- ROC-AUC: **70.74%**  

Decision threshold (dev-tuned): **0.10**

---

## Interpretation

The final model provides a **stable and non-collapsed classification system**:

- Predicts both classes effectively  
- Handles long-document structure reasonably well  
- Demonstrates the importance of preprocessing and training design  

---

## Limitations

- Uses a lightweight head-tail strategy instead of full document modeling  
- Very long documents are still compressed  
- Probability estimates are not fully calibrated  
- Focuses only on prediction (no explanation or retrieval components)

---

## Future Work

The module can be extended through:

- full long-document models (Longformer, BigBird)  
- hierarchical / multi-chunk architectures  
- probability calibration techniques  
- explanation or rationale extraction  
- integration with retrieval or structured legal components  

---

## Repository Structure