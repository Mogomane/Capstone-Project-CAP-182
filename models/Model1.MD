# Model 1 Architecture & Training Specification (`Model1.MD`)

## 1. Technique Details & Justification

* **Algorithm:** Logistic Regression (L2 Regularized)
* **Role in Project:** Baseline linear classifier establishing minimum performance benchmarks for return prediction (`label = 1` vs `label = 0`).
* **Hyperparameter Specifications:**
  * `penalty='l2'`: Prevents coefficient blowing up on correlated features.
  * `C=1.0`: Inverse regularization strength balancing bias and variance.
  * `solver='lbfgs'`: Optimization algorithm for small-to-medium scale tabular datasets.
  * `class_weight='balanced'`: Automatically adjusts class weights inversely proportional to class frequencies to address target imbalance.
  * `max_iter=1000`: Ensures complete solver convergence.

---

## 2. Script Linkage & Execution Guide

* **Associated Script:** `src/03_model1_logistic_regression.py`
* **Execution Command:**
  ```bash
  python src/03_model1_logistic_regression.py