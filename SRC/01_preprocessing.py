import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os

def run_preprocessing():
    print("Executing Step 1: Data Preprocessing...")
    # The raw path refers to the folder where the data is stored locally for github change to data/raw("returns_sustainability_dataset.csv")

    raw_path = "returns_sustainability_dataset.csv"
    os.makedirs("data/processed", exist_ok=True)
    
    
    df = pd.read_csv(raw_path)

    # Dynamically handle missing values if Return_Reason exists
    df['Return_Reason'] = df['Return_Reason'].fillna("No issue reported")
    
    # Scale only numerical columns that actually exist in the loaded dataframe
    potential_num_cols = [
        "Product_Price", "Discount_Applied", "Customer_Tenure", "Session_Dwell_Time",
    ]
    num_cols = [col for col in potential_num_cols if col in df.columns]
    
    if num_cols:
        scaler = RobustScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save processed dataframe
    output_path = "data/processed/cleaned_returns_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_path}")

if __name__ == "__main__":
    run_preprocessing()
