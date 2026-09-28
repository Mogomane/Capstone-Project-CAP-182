import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

def train_model1():
    print("Executing Step 3: Training Model 1 (Logistic Regression)...")
    
    df = pd.read_csv("data/processed/engineered_returns_features.csv")
    
    drop_cols = ['Customer_ID', 'Return_Reason', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Model 1 Initialization with Hyperparameters
    model1 = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    model1.fit(X_train, y_train)
    
    preds = model1.predict(X_test)
    probs = model1.predict_proba(X_test)[:, 1]
    
    print("\n================ MODEL 1 (LOGISTIC REGRESSION) METRICS ================")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model1, "models/baseline_logistic_regression.pkl")
    print("Model 1 saved to models/baseline_logistic_regression.pkl")

if __name__ == "__main__":
    train_model1()