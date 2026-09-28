## 1. Technique Details & Justification

* **Algorithm:** XGBoost Classifier (Gradient Boosted Decision Trees)
* **Role in Project:** Primary production-candidate model engineered to capture complex non-linear feature interactions (e.g., high discount ratio combined with high category risk score).
* **Hyperparameter Specifications:**
  * `n_estimators=150`: Number of sequential gradient boosted trees.
  * `max_depth=5`: Limits tree depth to control model complexity and prevent overfitting.
  * `learning_rate=0.05`: Step size shrinkage used to prevent overshooting minima.
  * `scale_pos_weight=4.88`: Cost-sensitive weighting ratio (count of negative samples / count of positive samples) prioritizing recall on true returns (`label = 1`).
  * `subsample=0.8`: Fraction of samples randomly trained per tree to increase generalization.
  * `colsample_bytree=0.8`: Fraction of features evaluated per split.

---

## 2. Script Linkage & Execution Guide

* **Associated Script:** `src/04_model2_xgboost.py`
* **Execution Command:**
  ```bash
  python src/04_model2_xgboost.py