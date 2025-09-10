import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def train_nlp_model(nlp_data_path):
    """
    Loads the prepared text data, trains a TF-IDF + XGBoost model,
    evaluates it, and saves the trained model and vectorizer.
    """
    print(f"Loading NLP data from {nlp_data_path}...")
    try:
        df = pd.read_csv(nlp_data_path)
    except FileNotFoundError:
        print(f"Error: NLP data file not found at {nlp_data_path}")
        return

    print("Preparing text data for NLP model...")
    
    # 1. Separate features (X) and target (y)
    X = df['description']
    y = df['buy_price']

    # 2. Split data into training and testing sets BEFORE vectorizing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Vectorize the text data using TF-IDF
    print("Vectorizing text with TF-IDF...")
    # We create a vectorizer that looks at the top 5000 most frequent words
    # This helps keep the model focused and computationally efficient.
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') # Using 'english' stop words for now, can be customized to Spanish
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Text vectorized into {X_train_tfidf.shape[1]} features.")

    print("\nTraining XGBoost model on NLP features...")
    # Initialize and train the XGBoost Regressor model
    nlp_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500, # Fewer estimators needed for this type of data
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    nlp_model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    print("\nEvaluating NLP model performance...")
    y_pred = nlp_model.predict(X_test_tfidf)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- NLP Model Performance ---")
    print(f"Mean Absolute Error (MAE): €{mae:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")
    print("-----------------------------")
    print(f"Interpretation: Using only text, the model's predictions are, on average, off by €{mae:,.2f}.")
    print(f"The text alone explains {r2:.2%} of the variance in property prices.")

    # 5. Save the trained model and the vectorizer
    model_dir = 'models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(nlp_model, os.path.join(model_dir, 'nlp_model.pkl'))
    joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
    print(f"\nTrained NLP model saved to '{os.path.join(model_dir, 'nlp_model.pkl')}'")
    print(f"TF-IDF Vectorizer saved to '{os.path.join(model_dir, 'tfidf_vectorizer.pkl')}'")

def main():
    """Main function to run the NLP model training process."""
    nlp_data_path = 'data/processed/madrid_houses_nlp.csv'
    train_nlp_model(nlp_data_path)

if __name__ == "__main__":
    main()