# Intelligent HomeValuator: Madrid

This project is an end-to-end data science application that predicts real estate prices in Madrid, Spain, using a machine learning model. It serves as a portfolio piece demonstrating skills in data sourcing, cleaning, feature engineering, model training, and deployment.

The final output is an interactive web application, containerized with Docker, that allows users to get price predictions for hypothetical properties.

## Project Structure

* `app.py`: The Gradio web application script.
* `Dockerfile`: Recipe to build a Docker container for the application.
* `requirements.txt`: A list of all Python libraries required for the project.
* `data/`: Contains raw and processed datasets.
* `models/`: Contains the saved, trained machine learning models.
* `plots/`: Contains output plots from the analysis.
* `src/`: Contains all the Python scripts for the data processing and model training pipeline.
    * `data/`: Scripts for downloading and exploring data.
    * `processing/`: Scripts for cleaning, merging, and feature engineering.
    * `training/`: Scripts for training the machine learning models.
    * `analysis/`: Scripts for analyzing model performance and feature importance.

## How to Run

### Option 1: Using Docker (Recommended)

This is the easiest way to run the application.

1.  **Build the Docker image:**

    ```bash
    docker build -t homevaluator-madrid .
    ```

2.  **Run the Docker container:**

    ```bash
    docker run -p 7860:7860 homevaluator-madrid
    ```

3.  **Access the application:**
    Open your web browser and go to `http://localhost:7860`.

### Option 2: Running Locally

1.  **Set up the environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

    *Note: If you haven't run the full pipeline, you will first need to run the scripts in the `src` directory to generate the clean data and trained models.*

2.  **Run the application:**

    ```bash
    python app.py
    ```

3.  **Access the application:**
    Open your web browser and go to the local URL provided in the terminal (usually `http://12-7.0.0.1:7860`).
