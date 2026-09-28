import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

def run_comparison():
    print("=" * 60)
    print("RUNNING MODEL COMPARISON & McNEMAR STATISTICAL TEST")
    print("=" * 60)
    
    df = pd.read_csv("data/processed/engineered_returns_features.csv")
    model1 = joblib.load("models/baseline_logistic_regression.pkl")
    model2 = joblib.load("models/optimized_xgboost.pkl")
    
    drop_cols = ['Customer_ID', 'Return_Reason', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['label']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    preds1 = model1.predict(X_test)
    preds2 = model2.predict(X_test)
    
    # Comparative Metrics Table
    metrics_data = {
        'Metric': ['Accuracy', 'Precision (Return)', 'Recall (Return)', 'F1-Score (Return)', 'ROC-AUC'],
        'Model 1 (Logistic Regression)': [
            accuracy_score(y_test, preds1),
            precision_score(y_test, preds1, pos_label=1),
            recall_score(y_test, preds1, pos_label=1),
            f1_score(y_test, preds1, pos_label=1),
            roc_auc_score(y_test, model1.predict_proba(X_test)[:, 1])
        ],
        'Model 2 (XGBoost)': [
            accuracy_score(y_test, preds2),
            precision_score(y_test, preds2, pos_label=1),
            recall_score(y_test, preds2, pos_label=1),
            f1_score(y_test, preds2, pos_label=1),
            roc_auc_score(y_test, model2.predict_proba(X_test)[:, 1])
        ]
    }
    
    comp_df = pd.DataFrame(metrics_data)
    print("\n--- Performance Comparison Table ---")
    print(comp_df.to_string(index=False))
    
    # McNemar's Test Setup
    correct1 = (preds1 == y_test.values)
    correct2 = (preds2 == y_test.values)
    
    # Contingency Table:
    # [Both correct, Model 1 correct & Model 2 wrong]
    # [Model 1 wrong & Model 2 correct, Both wrong]
    n11 = np.sum(correct1 & correct2)
    n10 = np.sum(correct1 & ~correct2)
    n01 = np.sum(~correct1 & correct2)
    n00 = np.sum(~correct1 & ~correct2)
    
    contingency_table = [[n11, n10], [n01, n00]]
    
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print("\n--- McNemar's Statistical Hypothesis Test ---")
    print(f"Contingency Table: {contingency_table}")
    print(f"Chi-Square Statistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.4e}")
    if result.pvalue < 0.05:
        print("Verdict: Reject Null Hypothesis. Model 2 improvement is STATISTICALLY SIGNIFICANT.")
    else:
        print("Verdict: Fail to reject Null Hypothesis. No statistically significant difference.")

if __name__ == "__main__":
    run_comparison()