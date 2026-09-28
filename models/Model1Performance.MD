## 1. Evaluation Overview & Dataset Split
Model 1 (Logistic Regression with L2 Regularization and Balanced Class Weighting) was evaluated on an unseen test dataset consisting of **750 customer transaction records** (626 non-returned orders [`label = 0`], 124 returned orders [`label = 1`]).

---

## 2. Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 49.73% (0.4973) |
| **Precision (Class 1 - Returned)** | 17.00% (0.17) |
| **Recall (Class 1 - Returned)** | 55.00% (0.55) |
| **F1-Score (Class 1 - Returned)** | 27.00% (0.27) |
| **Macro Average F1-Score** | 44.00% (0.44) |
| **Weighted Average F1-Score** | 56.00% (0.56) |
| **ROC-AUC Score** | 0.5082 |

---

## 3. Confusion Matrix Breakdown

              Predicted Not Returned (0)    Predicted Returned (1)
True Not Returned (0)        305 (TN)                   321 (FP)
True Returned (1)             56 (FN)                    68 (TP)


* **True Negatives (TN = 305):** Correctly identified non-returned purchases.
* **False Positives (FP = 321):** Non-returned orders misclassified as return risks due to balanced weighting threshold shifts.
* **False Negatives (FN = 56):** Returned purchases missed by the model.
* **True Positives (TP = 68):** Correctly identified high-risk return orders.

---

## 4. Script Linkage & Execution Guide

* **Associated Script:** `src/05_eval_model1.py`
* **Execution Command:**
  ```bash
  python src/05_eval_model1.py