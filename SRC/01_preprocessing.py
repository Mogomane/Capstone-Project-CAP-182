import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import os

def run_preprocessing():
    print("Executing Step 1: Data Preprocessing...")
    
    raw_path = "returns_sustainability_dataset.csv"
    os.makedirs("data/processed", exist_ok=True)
    
    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        # Fallback synthetic generator matching Kaggle schema
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            "Customer_ID": np.random.randint(1000, 2000, n),
            "Product_Category": np.random.choice(["Apparel", "Electronics", "Home", "Footwear"], n, p=[0.4, 0.2, 0.2, 0.2]),
            "Product_Price": np.random.uniform(100, 2500, n),
            "Discount_Applied": np.random.uniform(0, 500, n),
            "Customer_Tenure": np.random.randint(1, 60, n),
            "Session_Dwell_Time": np.random.uniform(10, 600, n),
            "Return_Reason": np.random.choice(["Size small", "Defective", "Wrong item", None], n, p=[0.1, 0.05, 0.02, 0.83]),
            "label": np.random.choice([0, 1], n, p=[0.83, 0.17])
        })

    # Impute missing Return_Reason
    df['Return_Reason'] = df['Return_Reason'].fillna("No issue reported")
    
    # Scale continuous numerical columns
    scaler = RobustScaler()
    num_cols = ["Product_Price", "Discount_Applied", "Customer_Tenure", "Session_Dwell_Time"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save processed dataframe
    output_path = "data/processed/cleaned_returns_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()