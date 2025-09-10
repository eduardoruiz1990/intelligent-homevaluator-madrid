import pandas as pd
import numpy as np
import os

def clean_data(raw_csv_path, processed_csv_path):
    """
    Loads the raw data, cleans it, performs basic feature engineering,
    and saves the result to a new CSV file.
    """
    print(f"Loading raw data from {raw_csv_path}...")
    try:
        df = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_csv_path}")
        return

    # Standardize column names (removes leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    print("Beginning data cleaning and feature engineering...")

    # 1. Select a core subset of potentially useful columns
    core_features = [
        'buy_price',
        'sq_mt_built',
        'n_rooms',
        'n_bathrooms',
        'is_new_development',
        'built_year',
        'has_lift',
        'is_exterior',
        'has_garden',
        'has_pool',
        'has_terrace',
        'has_balcony',
        'has_storage_room',
        'has_parking',
        'neighborhood_id' # We will use this for our socio-economic analysis later
    ]
    
    # Use .loc to ensure we are working with the actual DataFrame columns
    df_clean = df.loc[:, core_features].copy()

    # 2. Handle Missing Numerical Values (Imputation)
    for col in ['sq_mt_built', 'n_bathrooms', 'built_year']:
        if df_clean[col].isnull().any():
            missing_count = df_clean[col].isnull().sum()
            median_val = df_clean[col].median()
            # --- FIX: Use direct assignment to avoid FutureWarning ---
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"Filled {missing_count} missing values in '{col}' with median value ({median_val}).")

    # 3. Handle Missing Boolean/Categorical Values
    bool_cols = [
        'is_new_development', 'has_lift', 'is_exterior', 'has_garden', 'has_pool',
        'has_terrace', 'has_balcony', 'has_storage_room', 'has_parking'
    ]
    for col in bool_cols:
        # Convert object types to boolean, treating NaN as False
        if df_clean[col].dtype == 'object':
             df_clean[col] = df_clean[col].apply(lambda x: x == 'True')
        # --- FIX: Use direct assignment to avoid FutureWarning ---
        df_clean[col] = df_clean[col].fillna(False)

    # 4. Filter out extreme outliers and invalid data points
    initial_rows = len(df_clean)
    df_clean = df_clean[
        (df_clean['n_rooms'] > 0) &
        (df_clean['n_bathrooms'] > 0) &
        (df_clean['sq_mt_built'] > 20) &
        (df_clean['buy_price'] > 10000)
    ]
    print(f"Removed {initial_rows - len(df_clean)} rows with invalid data or extreme outliers.")

    # 5. Save the processed data
    processed_dir = os.path.dirname(processed_csv_path)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    df_clean.to_csv(processed_csv_path, index=False)
    print(f"\nCleaning complete. Processed data saved to '{processed_csv_path}'")
    print(f"New dataset has {len(df_clean)} rows and {len(df_clean.columns)} columns.")

def main():
    """Main function to run the cleaning process."""
    raw_data_path = 'data/raw/houses_Madrid.csv'
    processed_data_path = 'data/processed/madrid_houses_clean.csv'
    clean_data(raw_data_path, processed_data_path)

if __name__ == "__main__":
    main()

