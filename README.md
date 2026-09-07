# STADIOalot: Product Return Risk Mitigation


This repository contains the end-to-end Data Science Capstone Project (**CAP 182**) for **STADIOalot**, South Africa's largest online e-commerce retailer. The primary goal of this project is to build an interpretable, cost-sensitive machine learning pipeline that predicts product return risk *prior* to fulfillment, mitigating reverse-logistics costs and protecting operating profit margins.

---

## 📌 Executive Summary

STADIOalot operates on a massive scale (R38 Billion annual revenue) but struggles with fragile profitability (1.9% group operating margin). Reverse logistics represent one of the fastest-growing cost drivers:
* **13%** overall order return rate across the platform.
* **29%** return rate within high-risk categories such as apparel and footwear.
* Every return forces STADIOalot to absorb double-shipping, picking, restocking, and markdown costs.

**Project Solution:** By integrating customer transaction logs, catalog attributes, discount ratios, and real-time clickstream dwell patterns, this project implements an **XGBoost classification pipeline** to flag high-risk return orders at checkout before dispatch.

---

## 📁 Repository Structure

```text
stadioalot-return-risk-capstone/
│
├─ README.md                      <- Main project overview and setup guide
├─ requirements.txt               <- Python dependencies
├─ .gitignore                     <- Untracked files and local environment rules
│
├─ data/
│   ├─ raw/                       <- Raw transaction & return datasets
│   ├─ processed/                 <- Engineered, cleaned, and scaled datasets              
│
├─ notebooks/
│   ├─ 01_eda_and_data_cleaning.ipynb   <- Data exploration & missing value handling
│   ├─ 02_feature_engineering.ipynb     <- Discount ratios & bracket-buying flags
│   ├─ 03_model_training_evaluation.ipynb <- Baseline vs Ensemble model benchmarking
│   └─ 04_business_impact_analysis.ipynb <- Cost-matrix evaluation & ZAR savings
│
├─ src/
│   ├─ __init__.py
│   ├─ data_preprocessing.py      <- Cleaning, scaling, and SMOTE resampling logic
│   ├─ feature_builder.py         <- Sizing flags & category risk score generators
│   ├─ train_pipeline.py          <- Model training & hyperparameter tuning
│   └─ evaluate.py                <- Metrics calculation & confusion matrices
│
├─ models/
│   ├─ baseline_logistic_reg.pkl  <- Saved baseline model checkpoint
│   └─ optimized_xgboost.pkl      <- Final optimized model artifact
│
└─ reports/
    ├─ figures/                   <- SHAP plots, ROC curves, and confusion matrices
    └─ CAP182_Final_Report.pdf    <- Completed Capstone Academic Submission
