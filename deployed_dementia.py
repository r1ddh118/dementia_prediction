from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st
from imblearn.over_sampling import RandomOverSampler

st.set_page_config(page_title="Dementia Prediction", layout="wide")
st.title("Dementia Prediction Using Machine Learning")
st.sidebar.header("Upload Dataset")

class DementiaPrediction:
    def __init__(self):
        if "df" not in st.session_state:
            st.session_state.df = None
        if "model" not in st.session_state:
            st.session_state.model = None
        if "label_encoder" not in st.session_state:
            st.session_state.label_encoder = None
        if "scaler" not in st.session_state:
            st.session_state.scaler = None
        if "selector" not in st.session_state:
            st.session_state.selector = None

    
    def load_and_merge_data(self, uploaded_files):
        try:
            if not all(uploaded_files.values()):
                st.error("Please upload all required datasets!")
                return

            df_longitudinal = pd.read_excel(uploaded_files["longitudinal"], engine="openpyxl")
            df_cross_sectional = pd.read_excel(uploaded_files["cross_sectional"], engine="openpyxl")
            df_reliability = pd.read_excel(uploaded_files["cross_sectional_reliability"], engine="openpyxl")

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

            final_columns = ["Subject ID", "Age_long", "M/F_long", "EDUC", "SES_long", "MMSE_long", "CDR", "eTIV_long", "nWBV_long", "ASF_long"]
            final_df = merged_df[final_columns]
            final_df = final_df.loc[:, ~final_df.columns.duplicated()]

            st.session_state.df = final_df
            st.success(f"Merged dataset created successfully! Shape: {st.session_state.df.shape}")
            st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Error merging data: {e}")
    
    def preprocess_data(self):
        try:
            if st.session_state.df is None:
                st.error("Dataset not loaded! Please load data before preprocessing.")
                return

            df = st.session_state.df.copy()
            selected_features = ["Age_long", "M/F_long", "EDUC", "SES_long", "MMSE_long", "CDR", "eTIV_long", "nWBV_long", "ASF_long"]
            df = df[selected_features].dropna()

            imputer = SimpleImputer(strategy="median")
            df[["SES_long", "MMSE_long"]] = imputer.fit_transform(df[["SES_long", "MMSE_long"]])

            cdr_mapping = {0: "No Dementia", 0.5: "Mild Cognitive Impairment", 1: "Mild Dementia", 2: "Moderate Dementia", 3: "Severe Dementia"}
            df["CDR"] = df["CDR"].map(cdr_mapping)

            st.session_state.df = df
            st.success("Data preprocessing completed successfully! (No features standardized)")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
    
    def perform_eda(self):
       if st.session_state.df is None:
           st.error("Dataset not available! Load and preprocess data before performing EDA.")
           return
      
       df = st.session_state.df.copy()
       st.subheader("Exploratory Data Analysis (EDA)")
      
       # Distribution of CDR classes
       st.write("### Distribution of Dementia Levels")
       fig, ax = plt.subplots()
       sns.countplot(x=df["CDR"], ax=ax, palette="coolwarm")
       ax.set_title("Distribution of Different Dementia Levels")
       st.pyplot(fig)
       st.write("This shows how dementia cases are distributed across severity levels.")
      
       # Age distribution
       st.write("### Age Distribution")
       fig, ax = plt.subplots()
       sns.histplot(df["Age_long"], bins=30, kde=True, ax=ax, color="blue")
       ax.set_title("Age Distribution in Dataset")
       st.pyplot(fig)
       st.write("Age distribution helps understand the age groups affected by dementia.")
      
       # MMSE vs CDR
       st.write("### MMSE Score vs Dementia Severity")
       fig, ax = plt.subplots()
       sns.boxplot(x=df["CDR"], y=df["MMSE_long"], ax=ax, palette="coolwarm")
       ax.set_title("Cognitive Decline (MMSE) Across Dementia Levels")
       st.pyplot(fig)
       st.write("Lower MMSE scores indicate more severe cognitive impairment.")
      
       # Brain volume vs CDR
       st.write("### Brain Volume vs Dementia Severity")
       fig, ax = plt.subplots()
       sns.boxplot(x=df["CDR"], y=df["nWBV_long"], ax=ax, palette="coolwarm")
       ax.set_title("Brain Volume Decrease Across Dementia Levels")
       st.pyplot(fig)
       st.write("Brain volume shrinkage correlates with increasing dementia severity.")
    
    def train_model(self, model_type):
        if st.session_state.df is None:
            st.error("Dataset not available! Load and preprocess data before training.")
            return

        df = st.session_state.df.copy()
        X = df.drop(columns=["CDR"])
        y = df["CDR"]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        selector = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=8)
        X = selector.fit_transform(X, y)

        try:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        except ValueError:
            st.warning("SMOTE failed due to insufficient samples. Using RandomOverSampler instead.")
            ros = RandomOverSampler(random_state=42)
            X, y = ros.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 6, 10],
                "gamma": [0, 0.1, 0.2, 0.3],
                "subsample": [0.6, 0.7, 0.8, 1.0]
            },
            "Decision Tree": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5]
            },
            "SVM": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }
        }

        if model_type == "Random Forest":
            model = RandomForestClassifier(class_weight="balanced", random_state=42)
        elif model_type == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
        elif model_type == "Decision Tree":
            model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
        elif model_type == "SVM":
            model = SVC(probability=True, class_weight="balanced", random_state=42)

        # Perform hyperparameter tuning
        if model_type in param_grids:
            st.write(f"Tuning hyperparameters for {model_type}...")
            search = RandomizedSearchCV(model, param_grids[model_type], cv=5, scoring="accuracy", n_jobs=-1, n_iter=20, random_state=42)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            st.write("Best Parameters:", search.best_params_)

        # Train best model
        model.fit(X_train, y_train)

        # Save trained model, label encoder, and scaler in session state
        st.session_state.model = model
        st.session_state.label_encoder = label_encoder
        st.session_state.scaler = scaler
        st.session_state.selector = selector  # Save feature selector

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        cm = confusion_matrix(y_test, y_pred)

        st.success(f"Model trained successfully using {model_type}! Accuracy: {acc:.2f}")
        st.text("Classification Report:")
        st.text(report)

        # Confusion Matrix Visualization
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Decision Tree Visualization (only for Decision Tree model)
        if model_type == "Decision Tree":
            st.write("### Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(15, 8))
            plot_tree(model, filled=True, fontsize=6, feature_names=df.drop(columns=["CDR"]).columns, class_names=label_encoder.classes_)
            st.pyplot(fig)

    # Ensemble Learning (Random Forest + XGBoost + SVM)
        if model_type == "Ensemble":
            ensemble_model = VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")),
            ("svm", SVC(probability=True, random_state=42))
        ], voting="soft", weights=[2, 3, 1])
            ensemble_model.fit(X_train, y_train)
            y_pred = ensemble_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Ensemble Model Accuracy: {acc:.2f}")
            
    def predict_custom_patient(self):
        st.sidebar.header("Test a New Patient")

        # Ensure the model and preprocessing components exist
        if "model" not in st.session_state or "label_encoder" not in st.session_state or "scaler" not in st.session_state:
            st.sidebar.warning("Train a model first before testing a new patient.")
            return

        # User Input Form
        age = st.sidebar.number_input("Age", min_value=50, max_value=100, step=1, value=70)
        sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
        education = st.sidebar.number_input("Years of Education", min_value=0, max_value=20, step=1, value=12)
        ses = st.sidebar.number_input("SES (Socioeconomic Status)", min_value=1, max_value=5, step=1, value=3)
        mmse = st.sidebar.number_input("MMSE Score", min_value=0, max_value=30, step=1, value=25)
        eTIV = st.sidebar.number_input("eTIV (Brain Volume)", min_value=1100, max_value=2000, step=10, value=1500)
        nWBV = st.sidebar.number_input("nWBV (Normalized Brain Volume)", min_value=0.60, max_value=0.90, step=0.01, value=0.75)
        asf = st.sidebar.number_input("ASF (Atlas Scaling Factor)", min_value=0.80, max_value=1.50, step=0.01, value=1.10)

        if st.sidebar.button("Predict Dementia Level"):
            sex_encoded = 0 if sex == "Male" else 1
            test_input = np.array([[age, sex_encoded, education, ses, mmse, eTIV, nWBV, asf]])
            test_input = st.session_state.scaler.transform(test_input)

            if "selector" in st.session_state:
                test_input = st.session_state.selector.transform(test_input)

            model = st.session_state.model
            label_encoder = st.session_state.label_encoder
            prediction = model.predict(test_input)
            predicted_class = label_encoder.inverse_transform(prediction)[0]

            probabilities = model.predict_proba(test_input)[0]
            class_labels = label_encoder.classes_

            st.write(f"### Prediction: **{predicted_class}**")
            fig, ax = plt.subplots()
            ax.pie(probabilities, labels=class_labels, autopct="%1.1f%%")
            st.pyplot(fig)
            
      
# Instantiate Model
dementia_model = DementiaPrediction()

uploaded_files = {
    "longitudinal": st.sidebar.file_uploader("Upload Longitudinal Data", type=["xlsx"]),
    "cross_sectional": st.sidebar.file_uploader("Upload Cross-Sectional Data", type=["xlsx"]),
    "cross_sectional_reliability": st.sidebar.file_uploader("Upload Reliability Data", type=["xlsx"]),
}
if st.sidebar.button("Load and Merge Data"):
    dementia_model.load_and_merge_data(uploaded_files)
if st.sidebar.button("Preprocess Data"):
    dementia_model.preprocess_data()
if st.sidebar.button("Perform EDA"):
    dementia_model.perform_eda()
selected_model = st.sidebar.selectbox("Select Model", ["Random Forest", "XGBoost", "Decision Tree", "SVM"])
if st.sidebar.button("Train Model"):
    dementia_model.train_model(selected_model)

dementia_model.predict_custom_patient()