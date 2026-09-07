# STADIOalot: Product Return Risk Mitigation

This repository contains the end-to-end Data Science Capstone Project (**CAP 182**) for **STADIOalot**. The primary goal of this project is to investigate whether historical customer, transaction, product, seller, and behavioural data can be used to predict the likelihood that an order will be returned **before fulfilment**.

The project aims to support a more proactive approach to identifying potentially avoidable returns, reducing unnecessary operational costs, improving data-driven decision-making, and protecting profitability.

---

##  Executive Summary

STADIOalot operates at significant scale, serving more than **4.1 million active customers** and processing almost **58 million orders annually**. Despite this scale, profitability remains a challenge, with a group operating margin of approximately **1.9%**.

Product returns represent an important operational and financial challenge. STADIOalot's overall order return rate has increased from **11% to 15%**, while the return rate for apparel and shoes has increased from **22% to 29%**.

STADIOalot has identified several factors associated with higher return rates, including:

- Products
- Sellers
- Size-and-colour combinations
- Customer purchasing behaviour
- Customer online behaviour

However, these patterns are not currently being used to identify potentially higher-risk orders before purchase confirmation.

### Project Objective

The objective of this project is to investigate whether predictive modelling can identify orders with a higher likelihood of being returned using STADIOalot's historical data.

The project will use information available at or before the point of purchase to avoid data leakage and to ensure that the predictive approach reflects a realistic pre-fulfilment scenario.

---



# PART E: RAAIDD LOG

## RAAIDD

| **RAAIDD** | **Description** |
|---|---|
| **Risks** | **1. Quality of data:** Historical order, customer, product and return data may include missing values, duplicate values, incorrect dates or identifiers, all of which could impact the reliability of the predictive model. <br><br> **2. Data leakage:** Return reasons, return dates, return conditions and return costs are typically only available once an order has been returned. Using these as predictor variables could cause the model to produce misleading results because the study aims to determine whether it is possible to identify when an order is likely to be returned **before it is fulfilled**. <br><br> **3. Class imbalance:** The number of orders returned by customers may be significantly lower than the number of orders completed without a return. A model developed using this data may therefore favour non-returning orders and fail to adequately identify orders with a high risk of being returned. <br><br> **4. Limited behavioural data:** The amount of clickstream and/or search data related to each order may be limited or unreliable and may therefore be insufficient to effectively use customer behaviour as a predictor. <br><br> **5. Changing customer behaviour:** Customer purchasing and return behaviours may have changed since the period in which the historical data was collected. Therefore, a model built on historical data may not make reliable predictions about current return behaviour. |
| **Actions** | **1. Complete a review of data-quality issues:** Assess the order and return datasets for missing values, duplicate entries, incorrect dates, inconsistent identifiers and other data-quality problems, and determine the appropriate corrective actions. <br><br> **2. Establish the prediction point:** Determine the point at which the prediction will be made and eliminate variables that would only become available after an order or return has occurred to prevent data leakage. <br><br> **3. Assess and address class imbalance:** Review the distribution of `Return Status` and, where necessary, apply appropriate techniques such as class weighting or oversampling of the minority class. <br><br> **4. Create predictive attributes:** Develop additional predictive features using prior purchases, product characteristics, seller attributes, transaction characteristics and pre-purchase clickstream/search activity. <br><br> **5. Develop and evaluate classification models:** Develop and test multiple suitable classification models and evaluate their performance using measures such as precision, recall, F1-score and ROC-AUC, rather than relying only on accuracy. |
| **Assumptions** | **1. Adequate historical data:** It is assumed that STADIOalot's historical sales and returns data are sufficiently complete and accurate to support the development of a predictive model. <br><br> **2. Data integration:** It is assumed that suitable identifiers, including `Order ID`, `Product ID` and `User ID`, are available and can reliably connect the different datasets. <br><br> **3. Customer behaviour as a predictor:** It is assumed that prior customer purchases and online behaviours, including clickstream and search activity, provide useful information about whether an order is more or less likely to be returned. <br><br> **4. Product and seller signals:** It is assumed that product and seller attributes, including category, size, colour and seller history, contain measurable signals associated with return behaviour. <br><br> **5. Historical relevance:** It is assumed that past return behaviour is sufficiently representative of future return behaviour to enable the development of a useful predictive model. |
| **Issues** | **Issue #1 – Data consistency:** Some of the most important variables required for modelling may not contain accurate or consistent values. For example, some orders may not have a valid `Product ID`, `User ID` or `Return Status`. This could prevent records from being reliably connected across datasets or included in the predictive model. |
| **Decisions** | **Decision #1 – Target variable and predictors:** `Return Status` will be selected as the primary target variable, with `Returned` and `Not Returned` as the two outcome categories. Because the objective of the project is to determine which orders are most likely to be returned **before fulfilment costs are incurred**, only information available at or before the time of purchase will be used as predictive variables. |
| **Dependencies** | **Dependency #1 – Data access → Data preparation:** STADIOalot must supply the required historical order, product, customer, behavioural and returns data before data preparation can begin. <br><br> **Dependency #2 – Data preparation → Feature engineering:** Missing values, duplicate records and inconsistent identifiers must be identified and addressed before reliable predictive features can be engineered. <br><br> **Dependency #3 – Feature engineering → Model development:** The target variable and predictive features must be finalised before the classification models can be trained. <br><br> **Dependency #4 – Model development → Model evaluation:** The classification models must first be developed and trained before their ability to identify higher-risk orders can be evaluated. <br><br> **Dependency #5 – Model evaluation → Business recommendations:** Model performance must be evaluated before recommendations can be made regarding the proactive identification of higher-risk orders and the potential reduction of avoidable returns. |

---

#  Repository Structure

```text
stadioalot-return-risk-capstone/
│
├─ README.md
├─ requirements.txt
├─ .gitignore
│
├─ data/
│   ├─ raw/
│   │   └─ Raw transaction and return datasets
│   │
│   └─ processed/
│       └─ Cleaned and engineered datasets
│
├─ notebooks/
│   ├─ 01_eda_and_data_cleaning.ipynb
│   ├─ 02_feature_engineering.ipynb
│   ├─ 03_model_training_evaluation.ipynb
│   └─ 04_business_impact_analysis.ipynb
│
├─ src/
│   ├─ __init__.py
│   ├─ data_preprocessing.py
│   ├─ feature_builder.py
│   ├─ train_pipeline.py
│   └─ evaluate.py
│
├─ models/
│   ├─ baseline_logistic_reg.pkl
│   └─ optimized_xgboost.pkl
│
└─ reports/
    ├─ figures/
    │   ├─ SHAP plots
    │   ├─ ROC curves
    │   └─ Confusion matrices
    │
    └─ CAP182_Final_Report.pdf