import pandas as pd
import os
import re

def prepare_text_data(raw_csv_path, nlp_csv_path):
    """
    Loads the raw dataset, extracts and cleans text descriptions,
    and saves a new CSV file ready for the NLP model.
    """
    print(f"Loading raw data from {raw_csv_path}...")
    try:
        df = pd.read_csv(raw_csv_path)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_csv_path}")
        return

    # Standardize column names
    df.columns = df.columns.str.strip()
    
    print("Extracting and cleaning text data...")

    # 1. Select the relevant columns for our NLP task
    nlp_df = df[['title', 'subtitle', 'buy_price']].copy()
    
    # 2. Combine title and subtitle into a single 'description' column
    # Ensure both are strings and handle potential missing values
    nlp_df['description'] = nlp_df['title'].astype(str) + " " + nlp_df['subtitle'].astype(str)
    
    # 3. Basic text cleaning
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single one
        text = text.strip() # Remove leading/trailing spaces
        return text

    nlp_df['description'] = nlp_df['description'].apply(clean_text)
    
    # 4. Drop rows with very short or missing descriptions
    initial_rows = len(nlp_df)
    nlp_df = nlp_df[nlp_df['description'].str.len() > 10]
    print(f"Removed {initial_rows - len(nlp_df)} rows with short or invalid descriptions.")
    
    # 5. Select final columns and save
    final_df = nlp_df[['description', 'buy_price']]
    
    processed_dir = os.path.dirname(nlp_csv_path)
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        
    final_df.to_csv(nlp_csv_path, index=False)
    print(f"\nNLP data preparation complete. Processed data saved to '{nlp_csv_path}'")
    print(f"New dataset has {len(final_df)} rows and {len(final_df.columns)} columns.")


def main():
    """Main function to run the NLP data preparation process."""
    raw_data_path = 'data/raw/houses_Madrid.csv'
    nlp_data_path = 'data/processed/madrid_houses_nlp.csv'
    prepare_text_data(raw_data_path, nlp_data_path)

if __name__ == "__main__":
    main()