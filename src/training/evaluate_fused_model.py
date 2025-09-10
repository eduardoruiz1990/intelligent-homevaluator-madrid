import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

def evaluate_fused_model(processed_quant_path, processed_nlp_path):
    """
    Loads both trained models and evaluates their combined performance
    on a test set.
    """
    print("Loading datasets and trained models...")
    
    # --- Load Data ---
    try:
        df_quant = pd.read_csv(processed_quant_path)
        df_nlp = pd.read_csv(processed_nlp_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # --- Load Models and Encoders ---
    try:
        quant_model = joblib.load('models/quantitative_model.pkl')
        nlp_model = joblib.load('models/nlp_model.pkl')
        encoder = joblib.load('models/neighborhood_encoder.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading a model or supporting file: {e}")
        print("Please ensure both training scripts have been run successfully.")
        return

    print("Preparing a unified test set...")
    # Align the two datasets by index to ensure we're looking at the same properties
    df_full = df_quant.merge(df_nlp, left_index=True, right_index=True)
    
    # Define features and target
    X_quant = df_full.drop(['buy_price_x', 'buy_price_y', 'description'], axis=1)
    X_nlp = df_full['description']
    y = df_full['buy_price_x'] # Use the price from the quantitative set

    # --- Split data into train/test sets to get a consistent test set ---
    # We need to do this to evaluate on data the models haven't seen.
    # The random_state ensures we get the exact same split as during training.
    X_quant_train, X_quant_test, _, _ = train_test_split(X_quant, y, test_size=0.2, random_state=42)
    X_nlp_train, X_nlp_test, y_train, y_test = train_test_split(X_nlp, y, test_size=0.2, random_state=42)

    print("Generating predictions from both models on the test set...")
    # --- 1. Quantitative Model Prediction ---
    # One-hot encode neighborhood_id for the quantitative model
    X_quant_test_encoded = encoder.transform(X_quant_test[['neighborhood_id']])
    X_quant_test_encoded_df = pd.DataFrame(
        X_quant_test_encoded,
        index=X_quant_test.index,
        columns=encoder.get_feature_names_out(['neighborhood_id'])
    )
    X_quant_test_processed = pd.concat([X_quant_test.drop('neighborhood_id', axis=1), X_quant_test_encoded_df], axis=1)
    
    pred_quant = quant_model.predict(X_quant_test_processed)

    # --- 2. NLP Model Prediction ---
    # Vectorize text for the NLP model
    X_nlp_test_tfidf = vectorizer.transform(X_nlp_test)
    pred_nlp = nlp_model.predict(X_nlp_test_tfidf)

    # --- 3. Fuse the Predictions ---
    print("Fusing predictions with a weighted average...")
    # We give more weight to the more accurate quantitative model
    fusion_weight_quant = 0.80
    fusion_weight_nlp = 0.20
    
    pred_fused = (pred_quant * fusion_weight_quant) + (pred_nlp * fusion_weight_nlp)

    # --- Evaluate Performance ---
    print("\n--- Individual and Fused Model Performance ---")
    
    # Quantitative Model
    mae_quant = mean_absolute_error(y_test, pred_quant)
    r2_quant = r2_score(y_test, pred_quant)
    print(f"\nQuantitative Model ONLY:")
    print(f"  MAE: €{mae_quant:,.2f}")
    print(f"  R²: {r2_quant:.4f}")

    # NLP Model
    mae_nlp = mean_absolute_error(y_test, pred_nlp)
    r2_nlp = r2_score(y_test, pred_nlp)
    print(f"\nNLP Model ONLY:")
    print(f"  MAE: €{mae_nlp:,.2f}")
    print(f"  R²: {r2_nlp:.4f}")
    
    # Fused Model
    mae_fused = mean_absolute_error(y_test, pred_fused)
    r2_fused = r2_score(y_test, pred_fused)
    print(f"\nFused Model (80% Quant, 20% NLP):")
    print(f"  MAE: €{mae_fused:,.2f}")
    print(f"  R²: {r2_fused:.4f}")

    print("\n--- Final Interpretation ---")
    mae_improvement = mae_quant - mae_fused
    if mae_improvement > 0:
        print(f"Success! The fused model improved the MAE by €{mae_improvement:,.2f}.")
        print("This proves that adding qualitative text analysis provides real, measurable value.")
    else:
        print(f"The fused model did not improve upon the quantitative model (Improvement: €{mae_improvement:,.2f}).")
        print("This suggests the NLP signal, while interesting, did not add new predictive information.")


def main():
    """Main function to run the fused evaluation."""
    quant_path = 'data/processed/madrid_houses_clean.csv'
    nlp_path = 'data/processed/madrid_houses_nlp.csv'
    evaluate_fused_model(quant_path, nlp_path)

if __name__ == "__main__":
    main()
    