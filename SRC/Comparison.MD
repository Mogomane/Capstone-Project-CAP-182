# Model Comparison & Statistical Analysis (`Comparison.MD`)

## 1. Side-by-Side Empirical Performance Comparison

| Evaluation Metric | Model 1 (Logistic Regression) | Model 2 (XGBoost Classifier) | Absolute Difference ($\Delta$) | Higher Performing Model |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 49.73% | **63.73%** | +14.00% | **Model 2** |
| **Precision (`label=1`)** | **17.00%** | 15.00% | -2.00% | **Model 1** |
| **Recall (`label=1`)** | **55.00%** | 26.00% | -29.00% | **Model 1** |
| **F1-Score (`label=1`)** | **27.00%** | 19.00% | -8.00% | **Model 1** |
| **Weighted Avg F1** | 56.00% | **67.00%** | +11.00% | **Model 2** |
| **ROC-AUC Score** | **0.5082** | 0.5052 | +0.0030 | **Model 1** |

---

## 2. Performance Analysis & Trade-Offs

* **Accuracy vs. Recall Trade-off:** Model 2 achieves higher overall accuracy (63.73% vs 49.73%) by leaning toward predicting the majority class (`label = 0`). However, Model 1 captures significantly more actual returns (Recall = 55.00% vs 26.00%) due to its balanced class weighting strategy.
* **Discrimination Capacity:** Both models display ROC-AUC scores near $0.50$ (0.5082 vs 0.5052), indicating that the synthetic baseline features behave similarly to random noise relative to the return outcome. 

---

## 3. Statistical Hypothesis Testing (McNemar's Test)

McNemar's Test was executed on paired predictions across the 750 test samples to determine if the operational difference between the two classifiers is statistically significant.

$$\chi^2 = \frac{(\vert{}b - c\vert{} - 1)^2}{b + c}$$

* **Model 1 Correct / Model 2 Incorrect ($b$):** Samples where Logistic Regression predicted correctly but XGBoost failed.
* **Model 1 Incorrect / Model 2 Correct ($c$):** Samples where Logistic Regression failed but XGBoost predicted correctly.

### Decision Rule:
* If $p < 0.05$, the disagreement rate between Model 1 and Model 2 is statistically significant.
* Executing `src/07_model_comparison.py` evaluates the contingency table and outputs the exact Chi-Square statistic and $p$-value directly within the notebook interface.

---

## 4. Script Linkage & Execution Guide

* **Associated Script:** `src/07_model_comparison.py`
* **Execution Command:**
  ```bash
  python src/07_model_comparison.py