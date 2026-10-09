import joblib
import pytest

import train as train_module


@pytest.fixture(scope="module")
def trained_dir(root, tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    train_module.train(dataset_dir=str(root / "datasets"), model_dir=str(out))
    return out


def test_train_creates_all_artifacts(trained_dir):
    names = {p.name for p in trained_dir.iterdir()}
    assert names == {"dementia_model.joblib", "scaler.joblib",
                     "label_encoder.joblib", "selector.joblib"}


def test_trained_model_is_usable(trained_dir):
    m = joblib.load(trained_dir / "dementia_model.joblib")
    le = joblib.load(trained_dir / "label_encoder.joblib")
    sc = joblib.load(trained_dir / "scaler.joblib")
    sel = joblib.load(trained_dir / "selector.joblib")
    x = sel.transform(sc.transform([[70, 0, 12, 3, 25, 1500, 0.75, 1.1]]))
    assert le.inverse_transform(m.predict(x))[0] in le.classes_
    assert m.n_features_in_ == 8


def test_train_is_deterministic(trained_dir, root, tmp_path):
    train_module.train(dataset_dir=str(root / "datasets"), model_dir=str(tmp_path))
    a = joblib.load(trained_dir / "dementia_model.joblib")
    b = joblib.load(tmp_path / "dementia_model.joblib")
    x = [[0.1] * 8]
    assert (a.predict_proba(x) == b.predict_proba(x)).all()


def test_train_missing_datasets_does_not_write_model(tmp_path, capsys):
    out = tmp_path / "out"
    train_module.train(dataset_dir=str(tmp_path / "nope"), model_dir=str(out))
    assert not out.exists()
    assert "Dataset file not found" in capsys.readouterr().out


def test_dataset_files_present_and_readable(root):
    import pandas as pd
    for f in (root / "datasets").glob("*.xlsx"):
        assert len(pd.read_excel(f, engine="openpyxl")) > 0
