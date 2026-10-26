# STADIO CAP182 — SS3 Part B
## Order Return Prediction Using Machine Learning

**Project:** STADIOalot Order-Return Prediction  
**Module:** CAP182 — Capstone  
**Assessment:** Summative Assessment 3, Part B  
**Primary language:** Python  
**Machine-learning framework:** XGBoost and scikit-learn  
**Problem type:** Supervised binary classification   
**Student deatials:** Moggomane Digashu


---

## 1. Project Overview

This project develops and evaluates a machine-learning model that predicts whether an order will be returned.

The project applies the XGBoost classification approach carried forward from Summative Assessment 2 (SS2) to the dataset supplied for Summative Assessment 3 (SS3).

The objective is to identify orders that may have an elevated probability of being returned, using information available before the return outcome occurs.

The model is intended to support business decision-making by helping identify orders that may benefit from further review or proportionate interventions. It is not intended to automatically penalise customers, cancel orders, or deny services.

The notebook, `SS3_PartB.ipynb`, documents the complete modelling workflow:

1. Dataset loading and inspection.
2. Data cleaning and preprocessing.
3. Construction of an order-level modelling dataset.
4. Feature engineering and target definition.
5. Chronological training, validation, and test splitting.
6. XGBoost model training.
7. Model evaluation using appropriate classification metrics.
8. Visualisation of model performance and feature importance.
9. Interpretation of results and limitations.

The workflow is designed to be reproducible and transparent, with preprocessing decisions explained in the notebook.

---

## 2. Business Problem

Product returns can create operational and financial costs for an online retailer. These costs may include reverse logistics, inspection, restocking, refunds, and additional customer-service activity.

A predictive model may help the business identify orders with a higher probability of being returned before the outcome occurs.

The business problem is therefore formulated as a binary classification task.

### 2.1 Prediction target

The target variable is:

`return_status`

The target contains two classes:

| Class | Meaning |
|---|---|
| `Returned` | The order was recorded as returned. |
| `Not Returned` | The order was recorded as not returned. |

The model estimates the probability that an order belongs to the `Returned` class.

### 2.2 Business objective

The intended business objectives are to:

- Identify orders with elevated return risk.
- Support informed operational review.
- Understand patterns associated with returned orders.
- Evaluate the trade-off between missed returns and unnecessary interventions.
- Establish a baseline for future model improvement.

The model's performance must be assessed using more than accuracy alone because the target classes are imbalanced and false-positive and false-negative predictions may have different business consequences.

---

## 3. Dataset Description

The supplied dataset consists of six CSV files representing different aspects of the order, product, customer, browsing, return, and financial data.

| Dataset | Filename | Description |
|---|---|---|
| Orders and transactions | `table1_orders_transactions.csv` | Order and transaction information used to construct order-level features. |
| Product and seller | `table2_product_seller.csv` | Product and seller attributes associated with order lines. |
| Customer | `table3_customer.csv` | Customer-related attributes linked to orders. |
| Clickstream | `table4_clickstream.csv` | Browsing and session-related information. |
| Returns | `table5_returns.csv` | Return outcomes and related return-event information. |
| Financial and sustainability | `table6_financial_sustainability.csv` | Financial and sustainability-related outcomes. |

The datasets are synthetic and use South African rand (ZAR) for relevant monetary values.

### 3.1 Target definition

The `return_status` field from the returns dataset is used as the classification target.

The model must predict the outcome using eligible information available before the return event. Information recorded only after a return occurs must not be used as a predictor.

### 3.2 Dataset size and class distribution

After exact duplicate removal, table 1 contains 12,871 unique order records.

The product, customer, and returns tables are joined to construct the modelling dataset. Order-level aggregation is applied to the transaction data so that the modelling dataset contains one record per order rather than one record per order line.

Records with missing or invalid order dates are excluded from the chronological modelling dataset.

The recorded preprocessing results are:

| Measure | Result |
|---|---:|
| Orders after exact duplicate removal | 12,871 |
| Model-ready orders | 12,734 |
| `Not Returned` | 10,081 |
| `Returned` | 2,653 |
| Proportion `Not Returned` | 79.17% |
| Proportion `Returned` | 20.83% |

The target is imbalanced because approximately one-fifth of model-ready orders belong to the `Returned` class.

The class distribution is visualised in the notebook.

---

## 4. Data Loading and Preprocessing

Data preprocessing is important because inconsistent values, duplicated records, incorrect joins, missing values, and inappropriate features can reduce model reliability.

The notebook implements preprocessing steps that aim to preserve the correct prediction unit and reduce the risk of data leakage.

### 4.1 Loading the datasets

The notebook loads the six CSV files using pandas.

The files must be available in the configured data directory or in a location searched by the notebook.

After loading, the notebook inspects the datasets to establish their dimensions, column names, data types, and relevant identifiers.

### 4.2 Text standardisation

Relevant string fields are standardised by removing unnecessary leading and trailing whitespace.

The target labels are normalised to consistent values so that variations in capitalisation or whitespace do not create unintended additional classes.

This improves consistency during joining, filtering, and target construction.

### 4.3 Exact duplicate removal

Exact duplicate source rows are removed where appropriate.

Duplicate removal reduces the risk of identical records being counted more than once.

However, identical rows and repeated identifiers are not necessarily the same problem. Identifier uniqueness is checked separately for tables expected to contain one record per key.

### 4.4 Identifier checks

The workflow checks keys used to construct the modelling dataset.

The product table is expected to provide product-level attributes. Product records are handled to support a consistent product lookup.

The customer and returns information are linked to orders using the relevant order identifier.

These checks help reduce unintended row multiplication during joins.

### 4.5 Order-level aggregation

The order transactions data may contain multiple product lines belonging to a single order.

Because the target represents whether an order was returned, the prediction unit must be an order rather than an individual line.

The transaction records are therefore aggregated to one feature row per order.

Features include suitable summaries of the order's line items, such as:

- Number of order lines.
- Number of distinct products.
- Total quantity.
- Total order value.
- Sum of discounts.
- Sum and mean of product prices.
- Average delivery distance, where available.
- Number of distinct product categories.
- Number of distinct sellers.
- Representative categorical product attributes.
- Aggregated product ratings and other eligible product attributes.

The specific feature list is defined in the notebook.

This approach avoids simply retaining the first line of an order and discarding the remaining transaction information.

### 4.6 Joining product and customer attributes

Product attributes are attached to the transaction records using the product identifier before order-level aggregation.

The customer and target information are then linked to the aggregated order records using the order identifier.

The resulting modelling dataset is checked to ensure that each order appears only once.

The notebook also reports relevant row counts to make the data preparation process easier to audit.

### 4.7 Date processing

The order date is converted to a datetime representation.

Rows with missing or invalid order dates are excluded from the chronological modelling dataset because their temporal position cannot be established reliably.

Calendar-based features are derived from the order date where appropriate.

The order date is also used to determine the chronological ordering of observations before splitting the dataset.

### 4.8 Missing-value handling

Missing values are handled using a scikit-learn preprocessing pipeline.

For numerical features, missing values are imputed using the median calculated from the training data.

For categorical features, missing values are imputed using the most frequent category calculated from the training data.

Fitting the imputers on training data only reduces the risk of information from validation or test records influencing preprocessing decisions.

### 4.9 Categorical encoding

Categorical predictors are transformed using one-hot encoding.

The encoder is fitted within the training pipeline and configured to handle categories that may appear in validation or test data but were not observed during training.

This allows the model to process categorical features without treating arbitrary category labels as numerical quantities.

### 4.10 Data leakage prevention

Data leakage occurs when information unavailable at prediction time influences the model's training or evaluation.

The notebook excludes post-outcome variables, including return reasons, return dates, refund amounts, return-condition information, return-processing details, and financial outcomes that depend on the return event.

The financial and sustainability table is not incorporated into the predictor set where its fields represent post-outcome results or otherwise cannot be justified as available before the prediction decision.

The clickstream table is not directly joined to orders because it does not contain an order identifier. Joining it on a repeated user identifier alone could create many-to-many relationships and introduce browsing activity that occurred after the order or prediction point.

Clickstream features should only be introduced in future work if a defensible order-level linkage and appropriate temporal cutoff can be established.

These exclusions prioritise a realistic pre-outcome prediction setting over simply maximising measured model performance.

---

## 5. Training, Validation, and Test Strategy

The dataset is split chronologically rather than randomly.

A chronological split is appropriate because the model is intended to predict future order outcomes using patterns learned from earlier observations.

A random split could place later records in the training set while earlier records appear in the test set, making evaluation less representative of future deployment.

### 5.1 Split proportions

The modelling dataset is divided into:

- 70% training data.
- 15% validation data.
- 15% final test data.

The data is ordered by the order date before splitting.

### 5.2 Recorded split results

| Dataset | Records | Returned share |
|---|---:|---:|
| Training | 8,913 | 20.7% |
| Validation | 1,910 | 20.1% |
| Unseen test | 1,911 | 22.4% |
| **Total** | **12,734** | **20.8% overall** |

The recorded date ranges are:

| Dataset | Start | End |
|---|---|---|
| Training | 1 September 2022 | 14 June 2025 |
| Validation | 14 June 2025 | 25 January 2026 |
| Unseen test | 25 January 2026 | 30 August 2026 |

The notebook contains the precise timestamps.

The training subset is used to fit the model and preprocessing pipeline. The validation subset supports evaluation and model-selection decisions. The final test subset is reserved for evaluating performance on later, unseen records.

---

## 6. Model Selection and Training

### 6.1 Selected algorithm

The project uses XGBoost for binary classification.

XGBoost is a gradient-boosted decision-tree algorithm. It builds an ensemble of decision trees sequentially, with later trees learning patterns that help improve the ensemble's predictions.

The algorithm can model nonlinear relationships and interactions between structured numerical and categorical features.

XGBoost is retained to align the SS3 implementation with the approach selected in SS2.

### 6.2 Hyperparameters

The recorded baseline configuration uses the following parameters:

| Parameter | Value | Purpose |
|---|---:|---|
| `n_estimators` | 150 | Number of boosting trees. |
| `max_depth` | 5 | Limits individual tree depth. |
| `learning_rate` | 0.05 | Controls the contribution of each boosting step. |
| `subsample` | 0.8 | Uses a proportion of training rows for each boosting iteration. |
| `colsample_bytree` | 0.8 | Uses a proportion of features for each tree. |
| `random_state` | Fixed seed | Supports reproducibility. |
| `scale_pos_weight` | 3.839 in the recorded run | Compensates for class imbalance using the training subset. |

The exact model configuration is defined in `SS3_PartB.ipynb`.

The `scale_pos_weight` value is calculated from the training class distribution rather than the complete dataset. This prevents the test labels from influencing the training configuration.

These hyperparameters constitute a baseline configuration, not proof that the selected values are globally optimal.

### 6.3 Preprocessing pipeline

The numerical imputation, categorical imputation, and categorical encoding operations are combined into the model's preprocessing pipeline.

The pipeline is fitted on the training subset and then applied to validation and test data.

This ensures that preprocessing parameters are learned from the training data rather than from the full dataset.

### 6.4 Classification threshold

The recorded evaluation uses a probability threshold of 0.50.

An order is classified as `Returned` when its predicted probability meets the notebook's classification threshold; otherwise, it is classified as `Not Returned`.

A threshold of 0.50 provides a baseline for comparison. It should not automatically be assumed to be the best business threshold.

A threshold selected for operational use should be evaluated using validation predictions and the business costs associated with false positives and false negatives.

---

## 7. Model Evaluation

The model is evaluated using both the validation subset and the final unseen test subset.

The notebook calculates several classification metrics because no single metric fully describes performance on an imbalanced classification problem.

### 7.1 Evaluation metrics

**Accuracy**

The proportion of all predictions that are correct.

Accuracy is useful as a general measure but may conceal poor performance on the minority class.

**Precision for `Returned`**

The proportion of orders predicted as returned that were actually returned.

Higher precision means fewer false-positive flags among orders selected for review.

**Recall for `Returned`**

The proportion of actual returned orders identified by the model.

Higher recall means fewer actual returns are missed.

**F1-score for `Returned`**

The harmonic mean of precision and recall for the `Returned` class.

The F1-score provides a combined measure of the model's ability to identify returned orders while limiting false-positive predictions.

**ROC-AUC**

The area under the receiver operating characteristic curve.

ROC-AUC measures the model's ability to rank returned orders above non-returned orders across different classification thresholds.

AUC does not, by itself, establish that a particular operating threshold is profitable or appropriate.

### 7.2 Recorded evaluation results

The following results were generated by the updated notebook execution.

| Metric | Validation | Unseen test |
|---|---:|---:|
| Accuracy | 72.77% | 72.42% |
| Precision — `Returned` | 38.60% | 41.87% |
| Recall — `Returned` | 60.57% | 59.58% |
| F1-score — `Returned` | 0.4715 | 0.4918 |
| ROC-AUC | 0.7581 | 0.7254 |

The unseen test subset contains 1,911 orders, including 428 returned orders and 1,483 non-returned orders.

These are recorded execution results, not a guarantee that every Python environment will produce identical results. Re-run the notebook in the intended submission environment to verify reproducibility.

### 7.3 Interpretation

The unseen test ROC-AUC of 0.7254 indicates that the model has some ability to distinguish between returned and non-returned orders.

The recall of 59.58% indicates that the model identifies approximately six out of every ten actual returns at the recorded threshold.

The precision of 41.87% means that fewer than half of the orders flagged as returned were actually returned. A substantial number of flagged orders would therefore be false positives.

The results support treating the model as a baseline decision-support tool rather than a fully validated production system.

Further threshold analysis, cost evaluation, and validation on representative operational data are required before recommending deployment.

---

## 8. Visualisation

The notebook generates visualisations to communicate the dataset characteristics and model performance.

The available figures are stored in the `figures/` directory.

### 8.1 Return class distribution

File: `figures/return_class_balance.png`

This figure shows the distribution of the `Returned` and `Not Returned` classes.

It helps demonstrate the target imbalance and explains why evaluation should include class-specific metrics rather than accuracy alone.

### 8.2 Test confusion matrix

File: `figures/test_confusion_matrix.png`

The confusion matrix compares the model's predicted classes with the actual test labels.

It identifies:

- True positives: returned orders correctly identified.
- True negatives: non-returned orders correctly identified.
- False positives: non-returned orders incorrectly flagged as returns.
- False negatives: returned orders missed by the model.

These outcomes are important for understanding the model's potential operational workload.

### 8.3 ROC curve

File: `figures/test_roc_curve.png`

The ROC curve shows the relationship between the true-positive rate and false-positive rate at different classification thresholds.

It complements the single ROC-AUC value by showing how discrimination changes as the threshold changes.

### 8.4 Precision-recall curve

File: `figures/test_precision_recall_curve.png`

The precision-recall curve illustrates the trade-off between precision and recall at different thresholds.

It is particularly relevant when the positive class is less common and the business needs to balance missed returns against unnecessary interventions.

### 8.5 Feature importance

File: `figures/top_feature_importance.png`

The feature-importance visualisation identifies predictors that contribute to the trained XGBoost model according to the importance measure used in the notebook.

Feature importance should be interpreted cautiously. It does not establish that a feature causes returns, and importance values can be affected by correlated predictors and the chosen importance method.

---

## 9. Business Interpretation

The model provides a starting point for prioritising orders for review, but its predictions should be interpreted in the context of operational costs.

The SS2 executive recommendation report used the following business assumptions:

| Outcome | Assumed cost |
|---|---:|
| Missed return (false negative) | R180 |
| Unnecessary flag (false positive) | R15 |

These values are carried forward from SS2. They are business assumptions, not costs independently measured from the SS3 dataset.

Because the assumed cost of a missed return is greater than the cost of an unnecessary flag, a threshold below 0.50 may be worth investigating if the objective is to identify more actual returns.

However, lowering the threshold will generally increase the number of orders flagged, which can increase false positives and the workload required to review them.

The threshold should therefore be selected using validation data and a clearly defined cost calculation. The final test set should not be repeatedly used to choose the threshold.

A cost analysis should quantify:

- The number of false negatives.
- The number of false positives.
- The assumed cost of each outcome.
- The total expected cost under each candidate threshold.
- The volume of orders that would require review.
- The expected benefit compared with a suitable baseline.

A model should only be recommended for operational use if its performance and expected business value justify the additional complexity and intervention costs.

---

## 10. Limitations

The results should be interpreted in light of the following limitations.

### 10.1 Synthetic data

The supplied datasets are synthetic. Performance on synthetic data does not establish that the model will achieve the same results on real customer transactions.

### 10.2 Class imbalance

Returned orders represent approximately 20.83% of the model-ready data.

The minority-class precision and recall therefore require particular attention.

### 10.3 Moderate precision

The recorded test precision for `Returned` is 41.87%.

A substantial proportion of flagged orders would not be returned. An operational implementation would need to account for the cost and practicality of reviewing those orders.

### 10.4 Temporal data quality

A total of 137 records were excluded from the model-ready dataset because their order dates were missing or invalid.

The impact of these exclusions should be considered when assessing the representativeness of the final dataset.

### 10.5 Clickstream linkage

The clickstream data is not directly included because a reliable order-level link and temporal cutoff have not been established.

A future implementation could investigate clickstream features if they can be linked to orders without introducing many-to-many duplication or future information.

### 10.6 Threshold selection

The 0.50 threshold is a baseline, not a demonstrated cost-optimal decision threshold.

A cost-based threshold analysis is required before deployment.

### 10.7 Feature importance

Feature importance provides information about the trained model's use of predictors, not proof of causal relationships.

### 10.8 Deployment readiness

The model has not been established as a production-ready system through live deployment, operational monitoring, real-world validation, or a controlled business pilot.

---

## 11. Recommendations and Future Work

The following steps are recommended before operational implementation.

### 11.1 Validate the data pipeline

- Reconcile source row counts with modelling row counts.
- Confirm identifier uniqueness and join cardinality.
- Review missing and invalid dates.
- Confirm that every feature is available at the intended prediction time.
- Preserve a reproducible data-cleaning workflow.

### 11.2 Conduct threshold and cost analysis

Use validation probabilities to compare multiple thresholds against the SS2 business assumptions.

Calculate expected costs using the corresponding false-positive and false-negative counts.

Select an operating threshold before evaluating the final test set.

### 11.3 Compare against baseline models

Compare XGBoost against a simple baseline and, where appropriate, alternative classification algorithms.

All models should use the same chronological split and comparable evaluation metrics.

### 11.4 Investigate further feature engineering

Investigate additional pre-outcome features where they are supported by the available data.

Any clickstream integration should use defensible order-level linkage and exclude activity that occurs after the prediction decision point.

### 11.5 Run a controlled pilot

If further validation is satisfactory, evaluate the model in a limited, human-reviewed pilot.

Monitor precision, recall, false-positive workload, return prevalence, data drift, and performance across relevant customer or order groups.

### 11.6 Establish governance controls

Predictions should support proportionate review rather than automatically trigger adverse customer decisions.

Before deployment, establish human oversight, monitoring, clear accountability, and a process for investigating unexpected outcomes.

---

## 12. Repository Structure

The project package is organised around the notebook, supporting datasets, documentation, and visualisations.

```text
SS3_PartB_final_package/
│
├── SS3_PartB.ipynb
├── README.md
├── requirements.txt
├── EXECUTIVE_RECOMMENDATION_REPORT.md
├── SS3_PartB.html
│
├── data/
│   ├── table1_orders_transactions.csv
│   ├── table2_product_seller.csv
│   ├── table3_customer.csv
│   ├── table4_clickstream.csv
│   ├── table5_returns.csv
│   └── table6_financial_sustainability.csv
│
└── figures/
    ├── return_class_balance.png
    ├── test_confusion_matrix.png
    ├── test_roc_curve.png
    ├── test_precision_recall_curve.png
    └── top_feature_importance.png
```

The exact directory layout may vary slightly depending on how the repository is arranged. Keep the dataset paths consistent with those configured in the notebook.

The executed HTML file is provided as a convenient view of notebook outputs. The `.ipynb` file remains the primary notebook submission.

---

## 13. Installation and Setup

### 13.1 Prerequisites

Install Python and use a compatible Python environment.

Python 3.10–3.12 is recommended for this project to improve compatibility with the machine-learning dependencies.

The project requires:

- pandas
- NumPy
- Matplotlib
- scikit-learn
- XGBoost
- Jupyter Notebook support

### 13.2 Create a virtual environment

From the project root, run:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, you can use the environment's Python executable directly or activate it using the appropriate method for your system.

### 13.3 Install dependencies

With the virtual environment active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the `requirements.txt` file is unavailable, install the required libraries directly:

```bash
python -m pip install pandas numpy matplotlib scikit-learn xgboost jupyter nbclient nbconvert
```

### 13.4 Verify the environment

Run:

```bash
python -c "import pandas, numpy, sklearn, xgboost, matplotlib; print('Dependencies imported successfully')"
```

If this command fails, resolve the reported package or environment issue before running the notebook.

---

## 14. Running the Notebook

### Option A: Jupyter Notebook

From the project root, run:

```bash
python -m notebook
```

Open `SS3_PartB.ipynb` and select the active virtual environment as the notebook kernel.

Choose **Kernel → Restart Kernel and Run All Cells** or the equivalent option in your Jupyter interface.

### Option B: Visual Studio Code

1. Open the project folder in Visual Studio Code.
2. Install the Python and Jupyter extensions if required.
3. Open `SS3_PartB.ipynb`.
4. Select the Python interpreter associated with the project virtual environment.
5. Confirm that the six CSV files are available in the configured data directory.
6. Restart the notebook kernel.
7. Run all cells from the beginning.

Do not run cells out of order when verifying the complete workflow.

### 14.1 Dataset path troubleshooting

If the notebook reports that a CSV file cannot be found:

- Confirm that the ZIP package has been extracted.
- Confirm that the six CSV files are present.
- Confirm that the filenames match those expected by the notebook.
- Confirm that the notebook is searching the correct directory.
- Check the notebook's current working directory.

A common cause of file errors is opening the notebook from a different working directory than expected.

### 14.2 Dependency troubleshooting

If the notebook reports `ModuleNotFoundError`, install the missing package in the same Python environment selected as the notebook kernel.

For example:

```bash
python -m pip install xgboost
```

After installation, restart the notebook kernel and rerun the cells.

If the notebook still fails, record the full traceback and the cell in which it occurs. Do not assume that every error is caused by a missing package.

### 14.3 Reproducing the results

Run the notebook from the first cell to the last cell.

Confirm that:

- All six datasets load successfully.
- The preprocessing and joining steps complete.
- One row per order is maintained in the modelling dataset.
- The chronological splits are created correctly.
- The model trains without errors.
- Validation and test metrics are calculated.
- The visualisations are generated.
- The final output agrees with the results reported in this README.

Results may differ if the data, package versions, random seeds, or model implementation differ.

---

## 15. Git and GitHub Submission Requirements

The CAP182 SS3 Part B assessment requires the notebook to be submitted to the project repository and the commit history to contain at least five meaningful commits.

The commits should reflect actual stages of work. Do not create misleading or fabricated history.

A possible sequence is:

### Commit 1 — Dataset loading

```bash
git add SS3_PartB.ipynb
git commit -m "Add dataset loading and initial inspection"
```

### Commit 2 — Preprocessing

```bash
git add SS3_PartB.ipynb
git commit -m "Implement data cleaning and order-level aggregation"
```

### Commit 3 — Model training

```bash
git add SS3_PartB.ipynb requirements.txt
git commit -m "Implement XGBoost training pipeline"
```

### Commit 4 — Model validation

```bash
git add SS3_PartB.ipynb
git commit -m "Add chronological validation and test metrics"
```

### Commit 5 — Results and documentation

```bash
git add SS3_PartB.ipynb README.md figures
git commit -m "Document results and add performance visualisations"
```

These commands illustrate a possible workflow. They should be used at the relevant stages of real development rather than run retrospectively to imply that work happened in a different sequence.

If you have already completed the work in a single commit, continue with genuine improvements and commit those changes meaningfully. Do not rewrite history to misrepresent the development process.

### 15.1 Push to GitHub

After committing the work, push the branch to the configured repository:

```bash
git push
```

If the repository has not yet been connected to a remote, configure the correct GitHub repository before pushing.


## 16. Conclusion

This project applies an XGBoost binary classification model to the STADIOalot order-return prediction problem.

The workflow includes data cleaning, order-level feature construction, leakage-aware feature selection, chronological data splitting, model training, evaluation, and visualisation.

In the recorded execution, the model achieved a test ROC-AUC of 0.7254, a recall of 59.58% for returned orders, and a precision of 41.87% for the same class.

These results indicate a potentially useful predictive signal, but the model's precision and its performance on synthetic data limit the conclusions that can be drawn about real-world deployment.

Further validation, threshold selection based on business costs, and controlled operational testing are recommended before the model is used to support live decisions.

The notebook provides the primary technical evidence for the assessment, while this README documents the problem, workflow, results, limitations, and reproduction instructions.

---

**Primary assessment file:** `SS3_PartB.ipynb`  
**Supporting documentation:** `README.md` and `EXECUTIVE_RECOMMENDATION_REPORT.md`  
**Results and figures:** Generated by the notebook and stored in `figures/` 