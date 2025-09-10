import pandas as pd
import os
from unidecode import unidecode

def create_neighborhood_map(raw_housing_path):
    """
    Creates a mapping from the housing dataset's neighborhood_id to a clean
    neighborhood name derived from the 'subtitle'.
    """
    print("Creating a neighborhood ID to name map from raw housing data...")
    df_raw_housing = pd.read_csv(raw_housing_path)
    df_raw_housing.columns = df_raw_housing.columns.str.strip()
    
    # Keep only necessary columns and drop duplicates
    df_map = df_raw_housing[['neighborhood_id', 'subtitle']].dropna().drop_duplicates(subset=['neighborhood_id'])
    
    # Extract the neighborhood name from the 'subtitle' (e.g., "Piso en Goya, Salamanca" -> "Goya")
    df_map['DESC_BARRIO_CLEAN'] = df_map['subtitle'].str.split(',').str[0].str.replace('Piso en ', '').str.replace('Ático en ', '').str.replace('Dúplex en ', '').str.strip()
    
    # Normalize for merging: uppercase and remove accents
    df_map['DESC_BARRIO_CLEAN'] = df_map['DESC_BARRIO_CLEAN'].str.upper().apply(lambda x: unidecode(x) if isinstance(x, str) else x)
    
    return df_map[['neighborhood_id', 'DESC_BARRIO_CLEAN']]


def process_socio_data(socio_path):
    """
    Loads and processes the raw socio-economic data to get features per neighborhood.
    """
    print("Processing raw socio-economic (Padron) data...")
    df_socio = pd.read_csv(socio_path)
    
    # Clean column names (remove strange characters from ï»¿COD_DISTRITO)
    df_socio.columns = df_socio.columns.str.replace(r'^\W+', '', regex=True)
    
    # Sum up population figures for each neighborhood
    df_agg = df_socio.groupby('DESC_BARRIO').agg({
        'ESPANOLESHOMBRES': 'sum',
        'ESPANOLESMUJERES': 'sum',
        'EXTRANJEROSHOMBRES': 'sum',
        'EXTRANJEROSMUJERES': 'sum'
    }).reset_index()
    
    # Feature Engineering: Calculate total population and foreign resident ratio
    df_agg['poblacion_total'] = df_agg['ESPANOLESHOMBRES'] + df_agg['ESPANOLESMUJERES'] + df_agg['EXTRANJEROSHOMBRES'] + df_agg['EXTRANJEROSMUJERES']
    df_agg['poblacion_extranjera'] = df_agg['EXTRANJEROSHOMBRES'] + df_agg['EXTRANJEROSMUJERES']
    df_agg['ratio_extranjeros'] = df_agg['poblacion_extranjera'] / df_agg['poblacion_total']
    
    # Normalize neighborhood name for merging
    df_agg['DESC_BARRIO_CLEAN'] = df_agg['DESC_BARRIO'].str.upper().apply(lambda x: unidecode(x) if isinstance(x, str) else x)

    return df_agg[['DESC_BARRIO_CLEAN', 'poblacion_total', 'ratio_extranjeros']]

def main():
    """Main function to orchestrate the data merging process."""
    clean_housing_path = 'data/processed/madrid_houses_clean.csv'
    raw_housing_path = 'data/raw/houses_Madrid.csv'
    socio_path = 'data/raw/madrid_padron_2022.csv'
    enriched_path = 'data/processed/madrid_houses_enriched.csv'

    # 1. Load the clean housing data
    df_housing = pd.read_csv(clean_housing_path)
    
    # 2. Create the neighborhood mapping
    df_map = create_neighborhood_map(raw_housing_path)
    
    # 3. Process the socio-economic data
    df_socio_processed = process_socio_data(socio_path)
    
    # 4. Merge the mapping to the clean housing data
    df_housing_with_names = pd.merge(df_housing, df_map, on='neighborhood_id', how='left')
    
    # 5. Merge the socio-economic data
    print("Merging housing data with socio-economic features...")
    df_enriched = pd.merge(df_housing_with_names, df_socio_processed, on='DESC_BARRIO_CLEAN', how='left')
    
    # Clean up intermediate columns
    df_enriched = df_enriched.drop(columns=['DESC_BARRIO_CLEAN'])
    
    # --- FIX: Use direct assignment to prevent FutureWarning ---
    # Handle any potential missing values after the merge (e.g., if a neighborhood didn't match)
    df_enriched['poblacion_total'] = df_enriched['poblacion_total'].fillna(df_enriched['poblacion_total'].median())
    df_enriched['ratio_extranjeros'] = df_enriched['ratio_extranjeros'].fillna(df_enriched['ratio_extranjeros'].median())
    
    # Save the final enriched dataset
    df_enriched.to_csv(enriched_path, index=False)
    
    print(f"\nEnrichment complete! Final dataset saved to '{enriched_path}'")
    print("The new dataset includes 'poblacion_total' and 'ratio_extranjeros' for each property.")

if __name__ == "__main__":
    main()

