# Dementia Prediction Project

## Overview
This project focuses on predicting dementia using machine learning models. It utilizes the OASIS (Open Access Series of Imaging Studies) dataset, specifically its cross-sectional and longitudinal versions, to analyze patient demographics and cognitive decline patterns. The model is deployed using Streamlit for interactive user predictions.

## Dataset Description
The project uses three datasets from OASIS:

### 1. `oasis_longitudinal_demographics.xlsx`
- Contains demographic and cognitive assessment data of subjects over multiple visits.
- Important fields: `Subject ID`, `Visit`, `Age`, `MMSE` (Mini-Mental State Exam score), `CDR` (Clinical Dementia Rating), and MRI-derived features.
- Useful for tracking cognitive decline over time and understanding progression patterns.

### 2. `oasis_cross-sectional.xlsx`
- A cross-sectional dataset providing a single observation per subject.
- Key attributes include `Age`, `Gender`, `MMSE`, `CDR`, and `MRI measurements`.
- Helps in identifying key differentiating features between demented and non-demented individuals.

### 3. `oasis_cross-sectional-reliability.xlsx`
- Contains reliability assessments for features in the cross-sectional dataset.
- Helps in understanding data consistency and quality, ensuring robust model training.

## Methodology
### 1. **Data Preprocessing**
- **Handling Missing Values**: Missing data is imputed using median or mean values to maintain consistency.
- **Encoding Categorical Variables**: Variables like gender are encoded numerically to be used in machine learning models.
- **Feature Scaling & Normalization**: Standardization techniques are applied to ensure uniformity in data distribution.

### 2. **Exploratory Data Analysis (EDA)**
- **Statistical Analysis**: Distribution plots and summary statistics to understand data trends.
- **Feature Correlation**: Identifying relationships between MRI measures, MMSE, CDR, and other features.
- **Outlier Detection**: Using box plots and Z-score analysis to remove anomalies.

### 3. **Model Selection & Training**
- Implementing multiple models including **Random Forest** and **XGBoost** to predict dementia severity.
- Using **Grid Search & Cross-Validation** for hyperparameter tuning and optimizing performance.
- Evaluating models using **Accuracy, Precision, Recall, F1-score**, and **ROC-AUC**.

### 4. **Deployment Strategy**
- Deploying the trained model using **Streamlit** for real-time user interaction.
- Enabling user input for age, MMSE score, and other key parameters.
- Displaying predictions along with confidence scores and relevant data insights.

## Code Functionality
### `deployed_dementia.py`
This is the main script that:
- Loads the preprocessed dataset.
- Implements Random Forest and XGBoost classifiers.
- Trains models to predict dementia severity (`CDR` score).
- Deploys a Streamlit web application for user interaction.

### Key Features:
- **Data Preprocessing**: Handles missing values, encodes categorical variables, and normalizes numerical features.
- **Model Training**: Trains and evaluates multiple machine learning models.
- **Prediction & Visualization**: Provides insights through graphical representations and real-time user predictions.

## Running the Project
### Prerequisites:
- Python 3.x
- Required libraries: `pandas`, `numpy`, `sklearn`, `xgboost`, `streamlit`
- Install dependencies using:
  ```bash
  pip install -r requirements.txt

## Running the Streamlit App:

Execute the following command in the terminal:

```bash
streamlit run deployed_dementia.py
```
This will launch the interactive web application in the browser, allowing users to input patient details and get dementia severity predictions.

## Future Enhancements

    Improve feature selection and dimensionality reduction.

    Experiment with deep learning models.

    Integrate additional clinical datasets for better generalization.

    Incorporate Explainable AI (XAI) techniques to interpret model decisions better.
