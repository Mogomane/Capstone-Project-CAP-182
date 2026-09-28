import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

def evaluate_model2():
    print("=" * 60)
    print("RUNNING MODEL 2 (XGBOOST CLASSIFIER) EVALUATION")
    print("=" * 60)
    
    # Load dataset & model artifact
    df = pd.read_csv("data/processed/engineered_returns_features.csv")
    model2 = joblib.load("models/optimized_xgboost.pkl")
    
    drop_cols = ['Customer_ID', 'Return_Reason', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['label']
    
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Predictions
    preds = model2.predict(X_test)
    probs = model2.predict_proba(X_test)[:, 1]
    
    # Metrics computation
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    
    print("\n--- Model 2 Performance Summary ---")
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC Score : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, preds, target_names=['Not Returned (0)', 'Returned (1)']))
    
    # Feature Importances
    importances = pd.Series(model2.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop Feature Importances:")
    print(importances.head(5))

if __name__ == "__main__":
    evaluate_model2()