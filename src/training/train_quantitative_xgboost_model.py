import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import os

def train_model(processed_data_path):
    """
    Loads the clean data, trains an XGBoost model, evaluates it,
    and saves the trained model and encoder.
    """
    print(f"Loading processed data from {processed_data_path}...")
    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print(f"Error: Processed data file not found at {processed_data_path}")
        return

    print("Preparing data for training...")
    
    # 1. Separate features (X) and target (y)
    X = df.drop('buy_price', axis=1)
    y = df['buy_price']

    # 2. Handle the categorical feature: neighborhood_id
    # We use one-hot encoding to convert neighborhood IDs into numerical format
    # without implying an order.
    categorical_features = ['neighborhood_id']
    numerical_features = X.columns.drop(categorical_features)

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_encoded_categorical = pd.DataFrame(
        encoder.fit_transform(X[categorical_features]),
        index=X.index,
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # Combine numerical and encoded categorical features
    X_processed = pd.concat([X[numerical_features], X_encoded_categorical], axis=1)

    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    print("\nTraining XGBoost model...")
    # Initialize and train the XGBoost Regressor model
    xgboost_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,          # Number of trees to build
        learning_rate=0.05,         # How much to shrink the feature weights
        max_depth=5,                # Maximum depth of a tree
        subsample=0.8,              # Fraction of samples to be used for fitting each tree
        colsample_bytree=0.8,       # Fraction of columns to be used for each tree
        random_state=42,
        n_jobs=-1                   # Use all available CPU cores
    )

    # --- FINAL FIX: Remove early stopping to ensure compatibility ---
    xgboost_model.fit(
        X_train, y_train,
        verbose=False
    )
    print("Model training complete.")

    print("\nEvaluating model performance...")
    # Make predictions on the test set
    y_pred = xgboost_model.predict(X_test)

    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Performance (Enriched Data) ---")
    print(f"Mean Absolute Error (MAE): €{mae:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("-------------------------------------------")
    print(f"Interpretation: Our model's predictions are, on average, off by €{mae:,.2f}.")
    print(f"The model explains {r2:.2%} of the variance in the property prices.")

    # 5. Save the trained model and the encoder for later use
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(xgboost_model, os.path.join(model_dir, 'quantitative_enriched_model.pkl'))
    joblib.dump(encoder, os.path.join(model_dir, 'neighborhood_encoder_enriched.pkl'))
    print(f"\nTrained ENRICHED model saved to '{os.path.join(model_dir, 'quantitative_enriched_model.pkl')}'")
    print(f"Neighborhood ENRICHED encoder saved to '{os.path.join(model_dir, 'neighborhood_encoder_enriched.pkl')}'")

def main():
    """Main function to run the model training process."""
    # --- UPDATED: Use the new ENRICHED dataset ---
    processed_data_path = 'data/processed/madrid_houses_enriched.csv'
    train_model(processed_data_path)

if __name__ == "__main__":
    main()

