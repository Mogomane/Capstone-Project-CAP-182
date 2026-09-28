import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_model2():
    print("Executing Step 4: Training Model 2 (XGBoost Classifier)...")
    
    df = pd.read_csv("data/processed/engineered_returns_features.csv")
    
    drop_cols = ['Customer_ID', 'Return_Reason', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    
    # Model 2 Initialization with Hyperparameters
    model2 = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model2.fit(X_train, y_train)
    
    preds = model2.predict(X_test)
    probs = model2.predict_proba(X_test)[:, 1]
    
    print("\n================ MODEL 2 (XGBOOST CLASSIFIER) METRICS ================")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model2, "models/optimized_xgboost.pkl")
    print("Model 2 saved to models/optimized_xgboost.pkl")

if __name__ == "__main__":
    train_model2()