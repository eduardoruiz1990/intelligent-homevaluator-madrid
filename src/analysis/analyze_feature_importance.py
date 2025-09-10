import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_importance(model_path, data_path):
    """
    Loads a trained XGBoost model and analyzes the importance of its features.
    """
    print("Loading trained model and processed data...")
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        encoder = joblib.load('models/neighborhood_encoder_enriched.pkl')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure the model has been trained and the data exists.")
        return

    # --- We need to recreate the feature names exactly as they were during training ---
    X = df.drop('buy_price', axis=1)
    
    categorical_features = ['neighborhood_id']
    numerical_features = X.columns.drop(categorical_features)

    # Use the saved encoder to get the exact feature names
    X_encoded_categorical = pd.DataFrame(
        encoder.transform(X[categorical_features]),
        index=X.index,
        columns=encoder.get_feature_names_out(categorical_features)
    )
    X_processed = pd.concat([X[numerical_features], X_encoded_categorical], axis=1)
    
    # Get feature importances from the trained model
    feature_importances = pd.DataFrame({
        'feature': X_processed.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Top 20 Most Important Features ---")
    print(feature_importances.head(20))

    # --- Visualization ---
    print("\nGenerating feature importance plot...")
    plot_dir = 'plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 12))
    
    # --- FIX: Update seaborn plotting syntax to remove warning ---
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importances.head(20),
        palette='viridis',
        hue='feature', # Assign the y-variable to hue
        legend=False   # Disable the legend
    )
    
    plt.title('Top 20 Feature Importances for Price Prediction', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plot_path = os.path.join(plot_dir, 'feature_importance.png')
    plt.savefig(plot_path)
    print(f"Feature importance plot saved to '{plot_path}'")

def main():
    """Main function to run the analysis."""
    model_path = 'models/quantitative_enriched_model.pkl'
    data_path = 'data/processed/madrid_houses_enriched.csv'
    analyze_importance(model_path, data_path)

if __name__ == "__main__":
    main()

