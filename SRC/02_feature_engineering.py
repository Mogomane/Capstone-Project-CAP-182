import pandas as pd
import numpy as np

def run_feature_engineering():
    print("Executing Step 2: Feature Engineering...")
    
    input_path = "data/processed/cleaned_returns_data.csv"
    df = pd.read_csv(input_path)
    
    # 1. Discount Ratio
    df['Discount_Ratio'] = df['Discount_Applied'] / (df['Product_Price'] + 1e-5)
    
    # 2. Return Reason Text Length
    df['Return_Reason_Length'] = df['Return_Reason'].astype(str).apply(lambda x: len(x.split()))
    
    # 3. Category Target Encoding
    category_means = df.groupby('Product_Category')['label'].transform('mean')
    df['Category_Risk_Score'] = category_means
    
    # 4. One-Hot Encoding Categorical Columns
    df = pd.get_dummies(df, columns=['Product_Category'], drop_first=True)
    
    output_path = "data/processed/engineered_returns_features.csv"
    df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Dataset saved to {output_path}")

if __name__ == "__main__":
    run_feature_engineering()