"""Train, tune, and evaluate the CDR classifier without subject leakage."""
import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES = ["Age_long", "M/F_long", "EDUC", "SES_long", "MMSE_long",
            "eTIV_long", "nWBV_long", "ASF_long"]


def train(dataset_dir=None, model_dir=None):
    print("Starting model training process...")
    dataset_dir = Path(dataset_dir) if dataset_dir else BASE_DIR / "datasets"
    model_dir = Path(model_dir) if model_dir else BASE_DIR / "model"
    try:
        longitudinal = pd.read_excel(dataset_dir / "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx", engine="openpyxl")
        cross = pd.read_excel(dataset_dir / "oasis_cross-sectional-5708aa0a98d82080.xlsx", engine="openpyxl")
        reliability = pd.read_excel(dataset_dir / "oasis_cross-sectional-reliability-063c8642b909ee76.xlsx", engine="openpyxl")
    except FileNotFoundError as error:
        print(f"Error: Dataset file not found. Make sure the 'datasets' folder is in the correct location. Details: {error}")
        return

    cross["Subject ID"] = cross["ID"].str.split("_MR").str[0]
    reliability["Subject ID"] = reliability["ID"].str.split("_MR").str[0]
    # Keep the longitudinal education value used by FEATURES; renaming the
    # cross-sectional Educ column after the merge created a duplicate EDUC
    # column and silently expanded the scaler input from 8 to 9 features.
    cross = cross.rename(columns={"Educ": "EDUC_cross"})
    merged = longitudinal.merge(cross, on="Subject ID", how="outer", suffixes=("_long", "_cross"))
    merged = merged.merge(reliability, on="Subject ID", how="outer", suffixes=("", "_reliability"))
    merged.drop(columns=["ID", "MRI ID"], errors="ignore", inplace=True)
    merged.rename(columns={"Delay": "MR Delay"}, inplace=True)
    merged["CDR"] = merged["CDR_long"].combine_first(merged["CDR_cross"])
    merged.drop(columns=["CDR_long", "CDR_cross"], inplace=True)
    merged["M/F_long"] = merged["M/F_long"].map({"M": 0, "F": 1})
    merged["Hand_long"] = merged["Hand_long"].map({"R": 0, "L": 1})
    frame = merged[FEATURES + ["CDR", "Subject ID"]].dropna(subset=["CDR"]).copy()
    frame["CDR"] = frame["CDR"].map({0: "No Dementia", 0.5: "Mild Cognitive Impairment", 1: "Mild Dementia", 2: "Moderate Dementia", 3: "Severe Dementia"})
    frame.dropna(subset=["CDR"], inplace=True)
    if frame["Subject ID"].nunique() < 4 or frame["CDR"].nunique() < 2:
        raise ValueError("Need at least four subjects and two CDR classes for grouped evaluation.")

    X = frame[FEATURES].astype(float)
    y_text = frame["CDR"].astype(str)
    groups = frame["Subject ID"].astype(str)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    # Hold out subjects before any learned preprocessing, feature selection, or resampling.
    splitter = GroupShuffleSplit(n_splits=30, test_size=0.25, random_state=42)
    split = next((candidate for candidate in splitter.split(X, y, groups)
                  if len(np.unique(y[candidate[0]])) == len(encoder.classes_)), None)
    if split is None:
        raise ValueError("Could not form a subject-grouped holdout with every class represented in training.")
    train_idx, test_idx = split
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups.iloc[train_idx]
    n_splits = min(3, groups_train.nunique())
    if n_splits < 2:
        raise ValueError("Need at least two training subject groups for tuning.")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selector", RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=8)),
        ("oversample", RandomOverSampler(random_state=42)),
        ("classifier", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    search = GridSearchCV(
        pipe,
        {"classifier__n_estimators": [100, 200], "classifier__min_samples_leaf": [1, 2]},
        # Keep fitting in-process. Besides avoiding excess workers for this small
        # grid, this works around resource_tracker incompatibilities in some
        # Python 3.13 distributions.
        scoring="f1_macro", cv=GroupKFold(n_splits=n_splits), n_jobs=1,
        refit=True, error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups_train)
    best = search.best_estimator_
    predictions = best.predict(X_test)
    report = {
        "evaluation": "subject_group_holdout",
        "random_state": 42,
        "train_subjects": int(groups.iloc[train_idx].nunique()),
        "test_subjects": int(groups.iloc[test_idx].nunique()),
        "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
        "best_parameters": search.best_params_,
        "cross_validation_macro_f1": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "test_classification_report": classification_report(y_test, predictions, labels=np.arange(len(encoder.classes_)), target_names=encoder.classes_.tolist(), output_dict=True, zero_division=0),
        "test_confusion_matrix": confusion_matrix(y_test, predictions, labels=np.arange(len(encoder.classes_))).tolist(),
        "classes": encoder.classes_.tolist(),
        "dataset_sha256": hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest(),
        "note": "Single grouped holdout and grouped CV; metrics are not clinical validation.",
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.named_steps["classifier"], model_dir / "dementia_model.joblib")
    joblib.dump(best.named_steps["scaler"], model_dir / "scaler.joblib")
    joblib.dump(encoder, model_dir / "label_encoder.joblib")
    joblib.dump(best.named_steps["selector"], model_dir / "selector.joblib")
    (model_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    # Quantile bins are the privacy-preserving reference for input drift monitoring.
    baseline = {}
    for feature in FEATURES:
        values = X[feature].dropna().to_numpy(dtype=float)
        # An entirely missing feature has no reference distribution for drift
        # scoring. Omit it instead of failing while computing quantiles.
        if not len(values):
            continue
        edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))).tolist()
        if len(edges) < 2:
            edges = [float(values.min()) - 1, float(values.max()) + 1]
        baseline[feature] = {"edges": edges, "counts": np.histogram(values, bins=edges)[0].tolist(), "total": int(len(values))}
    (model_dir / "drift_baseline.json").write_text(json.dumps(baseline, indent=2))
    print(f"Training complete. Held-out macro F1: {report['test_macro_f1']:.3f}; report: {model_dir / 'evaluation.json'}")


if __name__ == "__main__":
    train()
