# Dementia Prediction Project

## Overview
This project focuses on predicting dementia using a machine learning model. It utilizes the OASIS (Open Access Series of Imaging Studies) dataset to analyze patient demographics and cognitive decline patterns. The model is served via a Flask web application with a simple HTML/JavaScript frontend.

## Dataset Description
The project uses three datasets from OASIS:
- `oasis_longitudinal_demographics.xlsx`
- `oasis_cross-sectional.xlsx`
- `oasis_cross-sectional-reliability.xlsx`

These datasets contain demographic information, clinical assessments (like MMSE and CDR scores), and MRI-derived features for a cohort of subjects.

## Project Structure
- **`train.py`**: A script dedicated to the complete machine learning pipeline. It handles:
    - Loading and merging the raw datasets.
    - Preprocessing the data (imputation, encoding).
    - Training a Random Forest classifier.
    - Saving the trained model, scaler, label encoder, and feature selector to the `model/` directory using `joblib`.
- **`app.py`**: The core Flask application. It:
    - Loads the pre-trained artifacts from the `model/` directory.
    - Serves the main user interface.
    - Provides a `/predict` API endpoint that takes patient data, preprocesses it, and returns a dementia prediction with class probabilities.
- **`templates/index.html`**: A single-page frontend with a form for users to input patient data. It uses JavaScript to communicate with the Flask API and display the results dynamically.
- **`Dockerfile`**: A configuration file to build a Docker image for the application, making it portable and easy to deploy. It uses `gunicorn` as the production WSGI server.
- **`requirements.txt`**: A list of all Python dependencies required for the project.

## Methodology
### 1. **Data Preprocessing**
- **Merging**: The three source datasets are merged into a single DataFrame.
- **Handling Missing Values**: Missing data in key columns is imputed using the median.
- **Encoding Categorical Variables**: Variables like gender (`M/F`) are encoded numerically.

### 2. **Model Training (`train.py`)**
- **Feature & Target Split**: The data is split into features (X) and the target variable (y, which is the `CDR` score).
- **Encoding Target**: The `CDR` text labels are converted to numerical format using `LabelEncoder`.
- **Scaling**: Features are standardized using `StandardScaler`.
- **Feature Selection**: `RFE` (Recursive Feature Elimination) with a `RandomForestClassifier` is used to select the top 8 most important features.
- **Oversampling**: `RandomOverSampler` is applied to handle class imbalance in the training data.
- **Training**: A `RandomForestClassifier` is trained on the preprocessed, resampled data.
- **Serialization**: The trained model and all preprocessing objects (scaler, encoder, selector) are saved to disk.

### 3. **Deployment (`app.py`)**
- The Flask application provides a web interface for real-time predictions.
- It is containerized using Docker for consistent and isolated deployment environments.

## Running the Project
### Prerequisites:
- Python 3.9+
- Docker (for containerized deployment)

### Option 1: Running Locally
1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model**:
    Run the training script to generate the model artifacts. This step is only required once.
    ```bash
    python3 train.py
    ```
    This will create a `model/` directory with the necessary `.joblib` files.

3.  **Run the Flask Application**:
    ```bash
    python3 app.py
    ```
    The application will be available at **http://localhost:5000**.

### Option 2: Running with Docker
1.  **Build the Docker Image**:
    Make sure the Docker daemon is running.
    ```bash
    docker build -t dementia-prediction-app .
    ```

2.  **Run the Docker Container**:
    ```bash
    docker run -p 5000:5000 dementia-prediction-app
    ```
    The application will be available at **http://localhost:5000**.

## Future Enhancements
- Experiment with different models (XGBoost, SVM, etc.) and compare performance.
- Implement more advanced feature engineering techniques.
- Enhance the frontend for better visualization of results.
- Deploy the container to a cloud service like AWS App Runner or ECS.
- Incorporate Explainable AI (XAI) techniques (e.g., SHAP) to interpret model decisions.
