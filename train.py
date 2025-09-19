import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from imblearn.over_sampling import RandomOverSampler
import joblib
import os

def train():
    print("Starting model training process...")

    # --- 1. Load and Merge Data ---
    print("Loading and merging datasets...")
    try:
        df_longitudinal = pd.read_excel("datasets/oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx", engine="openpyxl")
        df_cross_sectional = pd.read_excel("datasets/oasis_cross-sectional-5708aa0a98d82080.xlsx", engine="openpyxl")
        df_reliability = pd.read_excel("datasets/oasis_cross-sectional-reliability-063c8642b909ee76.xlsx", engine="openpyxl")
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. Make sure the 'datasets' folder is in the correct location. Details: {e}")
        return

    df_cross_sectional["Subject ID"] = df_cross_sectional["ID"].str.split("_MR").str[0]
    df_reliability["Subject ID"] = df_reliability["ID"].str.split("_MR").str[0]
    
    merged_df = df_longitudinal.merge(
        df_cross_sectional, on="Subject ID", how="outer", suffixes=("_long", "_cross")
    ).merge(
        df_reliability, on="Subject ID", how="outer", suffixes=("", "_reliability")
    )
    
    merged_df.drop(columns=["ID", "MRI ID"], errors="ignore", inplace=True)
    merged_df.rename(columns={"Educ": "EDUC", "Delay": "MR Delay"}, inplace=True)
    merged_df["CDR"] = merged_df["CDR_long"].combine_first(merged_df["CDR_cross"])
    merged_df.drop(columns=["CDR_long", "CDR_cross"], inplace=True)
    merged_df.dropna(subset=["CDR"], inplace=True)
    merged_df["M/F_long"] = merged_df["M/F_long"].map({"M": 0, "F": 1})
    merged_df["Hand_long"] = merged_df["Hand_long"].map({"R": 0, "L": 1})

    for col in ["EDUC", "SES_long", "MMSE_long", "eTIV_long", "nWBV_long", "ASF_long"]:
        merged_df[col] = merged_df[col].fillna(merged_df[col].median())

    final_columns = ["Age_long", "M/F_long", "EDUC", "SES_long", "MMSE_long", "CDR", "eTIV_long", "nWBV_long", "ASF_long"]
    df = merged_df[final_columns]
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"Data loaded and merged. Shape: {df.shape}")

    # --- 2. Preprocess Data ---
    print("Preprocessing data...")
    df = df.dropna()

    imputer = SimpleImputer(strategy="median")
    df[["SES_long", "MMSE_long"]] = imputer.fit_transform(df[["SES_long", "MMSE_long"]])

    cdr_mapping = {0: "No Dementia", 0.5: "Mild Cognitive Impairment", 1: "Mild Dementia", 2: "Moderate Dementia", 3: "Severe Dementia"}
    df["CDR"] = df["CDR"].map(cdr_mapping)
    df = df.dropna(subset=['CDR']) # Drop rows where CDR is NaN after mapping

    print("Data preprocessing complete.")

    # --- 3. Train Model ---
    print("Training Random Forest model...")
    X = df.drop(columns=["CDR"])
    y = df["CDR"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Using Random Forest for feature selection and as the main model
    rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    selector = RFE(rfe_estimator, n_features_to_select=8, step=1)
    X_selected = selector.fit_transform(X_scaled, y_encoded)
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_selected, y_encoded)

    model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    model.fit(X_resampled, y_resampled)
    print("Model training complete.")

    # --- 4. Save Artifacts ---
    print("Saving model and preprocessing artifacts...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, "model/dementia_model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump(label_encoder, "model/label_encoder.joblib")
    joblib.dump(selector, "model/selector.joblib")
    
    print("Artifacts saved successfully in the 'model' directory.")
    print("Training process finished.")

if __name__ == '__main__':
    train()
