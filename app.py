import gradio as gr
import pandas as pd
import joblib
import os

# --- 1. Load Trained Models and Supporting Files ---
print("Loading models and necessary data...")
try:
    MODEL = joblib.load('models/quantitative_enriched_model.pkl')
    ENCODER = joblib.load('models/neighborhood_encoder_enriched.pkl')
    
    # --- FIX: Load the ENRICHED dataset that the model was trained on ---
    DATA = pd.read_csv('data/processed/madrid_houses_enriched.csv')
    
    NEIGHBORHOODS = sorted(DATA['neighborhood_id'].unique().tolist())
    print("Models and data loaded successfully.")
except FileNotFoundError:
    print("Error: Model files not found. Please run the training script first.")
    MODEL, ENCODER, DATA, NEIGHBORHOODS = None, None, None, []

# --- 2. Define the Prediction Function ---
def predict_price(sq_mt_built, n_rooms, n_bathrooms, neighborhood):
    """
    Takes user inputs, processes them, and returns a price prediction.
    """
    if MODEL is None:
        return "ERROR: Model not loaded. Please check the console."

    # Create a DataFrame from the user's inputs
    # Set amenities to False since they are removed from the UI
    input_data = pd.DataFrame({
        'sq_mt_built': [sq_mt_built],
        'n_rooms': [n_rooms],
        'n_bathrooms': [n_bathrooms],
        'is_new_development': [False],
        'built_year': [2000],
        'has_lift': [False],
        'is_exterior': [False],
        'has_garden': [False],
        'has_pool': [False],
        'has_terrace': [False],
        'has_balcony': [False],
        'has_storage_room': [False],
        'has_parking': [False],
        'neighborhood_id': [neighborhood],
        'poblacion_total': [DATA['poblacion_total'].median()],
        'ratio_extranjeros': [DATA['ratio_extranjeros'].median()]
    })

    # --- Preprocessing ---
    # One-hot encode the neighborhood
    encoded_neighborhood = ENCODER.transform(input_data[['neighborhood_id']])
    encoded_neighborhood_df = pd.DataFrame(
        encoded_neighborhood,
        columns=ENCODER.get_feature_names_out(['neighborhood_id'])
    )
    
    # Combine numerical and encoded features
    processed_input = pd.concat(
        [input_data.drop('neighborhood_id', axis=1), encoded_neighborhood_df],
        axis=1
    )
    
    # Ensure the order of columns matches the training data
    processed_input = processed_input[MODEL.get_booster().feature_names]

    # --- Prediction ---
    prediction = MODEL.predict(processed_input)[0]
    
    # Format the output
    return f"€{prediction:,.0f}"

# --- 3. Create the Gradio Interface ---
print("Creating Gradio interface...")
with gr.Blocks(theme=gr.themes.Soft(), title="Madrid HomeValuator") as app:
    gr.Markdown("# 🏡 Intelligent HomeValuator: Madrid")
    gr.Markdown("Enter the details of a property to get an estimated price from our AI model.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Core Features")
            sq_mt_input = gr.Slider(minimum=30, maximum=500, value=100, label="Square Meters (Built)")
            rooms_input = gr.Slider(minimum=1, maximum=10, step=1, value=3, label="Number of Rooms")
            bathrooms_input = gr.Slider(minimum=1, maximum=5, step=1, value=2, label="Number of Bathrooms")
            neighborhood_input = gr.Dropdown(choices=NEIGHBORHOODS, value=NEIGHBORHOODS[0], label="Neighborhood")
        
        # --- REMOVED Amenities Column ---

    predict_button = gr.Button("Predict Price", variant="primary")
    
    with gr.Row():
        output_price = gr.Textbox(label="Predicted Price", interactive=False)

    predict_button.click(
        fn=predict_price,
        inputs=[
            sq_mt_input,
            rooms_input,
            bathrooms_input,
            neighborhood_input
            # --- REMOVED Amenity Inputs ---
        ],
        outputs=output_price
    )

print("Launching application...")

# --- 4. Launch the App ---
# Share=True creates a public link for easy sharing
app.launch(share=False, server_name="0.0.0.0")

