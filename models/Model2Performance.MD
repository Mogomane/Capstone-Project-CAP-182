## 1. Evaluation Overview & Dataset Split
Model 2 (Cost-Sensitive XGBoost Classifier with `scale_pos_weight` tuning) was evaluated on the identical unseen test set of **750 customer transaction records** (626 non-returned orders, 124 returned orders).

---

## 2. Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 63.73% (0.6373) |
| **Precision (Class 1 - Returned)** | 15.00% (0.15) |
| **Recall (Class 1 - Returned)** | 26.00% (0.26) |
| **F1-Score (Class 1 - Returned)** | 19.00% (0.19) |
| **Macro Average F1-Score** | 48.00% (0.48) |
| **Weighted Average F1-Score** | 67.00% (0.67) |
| **ROC-AUC Score** | 0.5052 |

---

## 3. Confusion Matrix Breakdown

              Predicted Not Returned (0)    Predicted Returned (1)
True Not Returned (0)        446 (TN)                   180 (FP)
True Returned (1)             92 (FN)                    32 (TP)


* **True Negatives (TN = 446):** Correctly identified non-returned purchases.
* **False Positives (FP = 180):** Orders flagged as returns that were retained by the customer.
* **False Negatives (FN = 92):** Returned orders undetected prior to dispatch.
* **True Positives (TP = 32):** Correctly flagged return risks.

---

## 4. Script Linkage & Execution Guide

* **Associated Script:** `src/06_eval_model2.py`
* **Execution Command:**
  ```bash
  python src/06_eval_model2.py