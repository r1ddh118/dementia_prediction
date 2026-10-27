import joblib
import pandas as pd
import pytest

import train as train_module


@pytest.fixture(scope="module")
def training_dataset(tmp_path_factory):
    dataset_dir = tmp_path_factory.mktemp("datasets")
    rows = []
    labels = [0, 0.5, 1, 2] * 3
    for index, cdr in enumerate(labels):
        subject = f"TEST_{index:04d}"
        rows.append({
            "Subject ID": subject,
            "MRI ID": f"{subject}_MR1",
            "Group": "Nondemented",
            "Visit": 1,
            "MR Delay": 0,
            "M/F": "M" if index % 2 == 0 else "F",
            "Hand": "R",
            "Age": 60 + index,
            "EDUC": 12 + index % 4,
            "SES": 1 + index % 4,
            "MMSE": 25 + index % 6,
            "CDR": cdr,
            "eTIV": 1300 + index * 10,
            "nWBV": 0.7 + index / 100,
            "ASF": 1.1 + index / 100,
        })

    longitudinal = pd.DataFrame(rows)
    cross_sectional = longitudinal.rename(columns={"Subject ID": "ID", "EDUC": "Educ"})
    cross_sectional["ID"] = cross_sectional["ID"] + "_MR1"
    reliability = pd.DataFrame([{
        "ID": "RELIABILITY_MR1",
        "Delay": 1,
        "eTIV": 1400,
        "nWBV": 0.75,
        "ASF": 1.0,
    }])

    longitudinal.to_excel(dataset_dir / "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx", index=False)
    cross_sectional.to_excel(dataset_dir / "oasis_cross-sectional-5708aa0a98d82080.xlsx", index=False)
    reliability.to_excel(dataset_dir / "oasis_cross-sectional-reliability-063c8642b909ee76.xlsx", index=False)
    return dataset_dir


@pytest.fixture(scope="module")
def trained_dir(training_dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    train_module.train(dataset_dir=str(training_dataset), model_dir=str(out))
    return out


def test_train_creates_all_artifacts(trained_dir):
    names = {p.name for p in trained_dir.iterdir()}
    assert {"dementia_model.joblib", "scaler.joblib", "label_encoder.joblib",
            "selector.joblib", "evaluation.json", "drift_baseline.json"} <= names


def test_trained_model_is_usable(trained_dir):
    m = joblib.load(trained_dir / "dementia_model.joblib")
    le = joblib.load(trained_dir / "label_encoder.joblib")
    sc = joblib.load(trained_dir / "scaler.joblib")
    sel = joblib.load(trained_dir / "selector.joblib")
    x = sel.transform(sc.transform([[70, 0, 12, 3, 25, 1500, 0.75, 1.1]]))
    assert le.inverse_transform(m.predict(x))[0] in le.classes_
    assert m.n_features_in_ == 8


def test_train_is_deterministic(trained_dir, training_dataset, tmp_path):
    train_module.train(dataset_dir=str(training_dataset), model_dir=str(tmp_path))
    a = joblib.load(trained_dir / "dementia_model.joblib")
    b = joblib.load(tmp_path / "dementia_model.joblib")
    x = [[0.1] * 8]
    assert (a.predict_proba(x) == b.predict_proba(x)).all()


def test_train_missing_datasets_does_not_write_model(tmp_path, capsys):
    out = tmp_path / "out"
    train_module.train(dataset_dir=str(tmp_path / "nope"), model_dir=str(out))
    assert not out.exists()
    assert "Dataset file not found" in capsys.readouterr().out


def test_dataset_files_present_and_readable(training_dataset):
    for f in training_dataset.glob("*.xlsx"):
        assert len(pd.read_excel(f, engine="openpyxl")) > 0
