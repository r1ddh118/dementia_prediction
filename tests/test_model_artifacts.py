"""The committed model/ folder: this is what gets pushed to the registry."""
import joblib
import numpy as np
import pytest

FILES = ["dementia_model", "scaler", "label_encoder", "selector"]


@pytest.fixture(scope="module")
def artifacts(root):
    return {n: joblib.load(root / "model" / f"{n}.joblib") for n in FILES}


@pytest.mark.parametrize("name", FILES)
def test_artifact_file_exists_and_not_empty(root, name):
    f = root / "model" / f"{name}.joblib"
    assert f.is_file() and f.stat().st_size > 0


def test_artifact_types(artifacts):
    assert type(artifacts["dementia_model"]).__name__ == "RandomForestClassifier"
    assert type(artifacts["scaler"]).__name__ == "StandardScaler"
    assert type(artifacts["label_encoder"]).__name__ == "LabelEncoder"
    assert type(artifacts["selector"]).__name__ == "RFE"


def test_expected_classes(artifacts):
    assert set(artifacts["label_encoder"].classes_) == {
        "No Dementia", "Mild Cognitive Impairment", "Mild Dementia", "Moderate Dementia"}


def test_pipeline_shapes_are_consistent(artifacts):
    assert artifacts["scaler"].n_features_in_ == 8
    assert artifacts["selector"].n_features_in_ == 8
    assert artifacts["selector"].n_features_ == artifacts["dementia_model"].n_features_in_


def test_end_to_end_prediction(artifacts):
    x = np.array([[70, 0, 12, 3, 25, 1500, 0.75, 1.10]])
    x = artifacts["selector"].transform(artifacts["scaler"].transform(x))
    m = artifacts["dementia_model"]
    proba = m.predict_proba(x)[0]
    assert proba.shape[0] == len(artifacts["label_encoder"].classes_)
    assert proba.sum() == pytest.approx(1.0)
    assert artifacts["label_encoder"].inverse_transform(m.predict(x))[0] in artifacts["label_encoder"].classes_
