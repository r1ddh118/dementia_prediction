from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os
import time
import json
import logging
import sqlite3
from collections import Counter
from threading import Lock

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "model")).resolve()
MONITORING_DB = Path(os.environ.get("MONITORING_DB", BASE_DIR / "monitoring" / "metrics.sqlite3"))
DRIFT_PSI_THRESHOLD = float(os.environ.get("DRIFT_PSI_THRESHOLD", "0.25"))
DRIFT_MIN_PREDICTIONS = int(os.environ.get("DRIFT_MIN_PREDICTIONS", "20"))
FEATURE_NAMES = ["Age_long", "M/F_long", "EDUC", "SES_long", "MMSE_long", "eTIV_long", "nWBV_long", "ASF_long"]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")
)
# Process-local operational metrics. Inputs are deliberately never recorded.
_metrics_lock = Lock()
_metrics = {
    'prediction_requests': 0,
    'prediction_errors': 0,
    'prediction_latency_seconds_total': 0.0,
    'predictions_by_class': Counter(),
}
try:
    MODEL_VERSION = (MODEL_DIR / "VERSION").read_text().strip()
except OSError:
    MODEL_VERSION = "local"
try:
    DRIFT_BASELINE = json.loads((MODEL_DIR / "drift_baseline.json").read_text())
except (OSError, json.JSONDecodeError):
    DRIFT_BASELINE = {}


def _db():
    MONITORING_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(MONITORING_DB, timeout=10)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE IF NOT EXISTS service_totals (key TEXT PRIMARY KEY, value REAL NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS prediction_classes (label TEXT PRIMARY KEY, count INTEGER NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS feature_bins (feature TEXT NOT NULL, bin_index INTEGER NOT NULL, count INTEGER NOT NULL, PRIMARY KEY(feature, bin_index))")
    connection.commit()
    return connection


def _psi(expected, observed):
    expected = np.asarray(expected, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if expected.sum() == 0 or observed.sum() == 0:
        return 0.0
    expected = np.clip(expected / expected.sum(), 1e-6, None)
    observed = np.clip(observed / observed.sum(), 1e-6, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _record_prediction(values, label, latency):
    """Persist aggregate counters and histogram bins, never raw feature values."""
    connection = _db()
    try:
        with connection:
            for key, value in (("prediction_requests", 1), ("prediction_latency_seconds_total", latency)):
                connection.execute("INSERT INTO service_totals(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=value+excluded.value", (key, value))
            connection.execute("INSERT INTO prediction_classes(label,count) VALUES(?,1) ON CONFLICT(label) DO UPDATE SET count=count+1", (str(label),))
            for name, value in zip(FEATURE_NAMES, values):
                baseline = DRIFT_BASELINE.get(name)
                if not baseline:
                    continue
                index = int(np.clip(np.searchsorted(baseline["edges"][1:-1], value, side="right"), 0, len(baseline["counts"]) - 1))
                connection.execute("INSERT INTO feature_bins(feature,bin_index,count) VALUES(?,?,1) ON CONFLICT(feature,bin_index) DO UPDATE SET count=count+1", (name, index))
    finally:
        connection.close()


def _record_error():
    connection = _db()
    try:
        with connection:
            connection.execute("INSERT INTO service_totals(key,value) VALUES('prediction_errors',1) ON CONFLICT(key) DO UPDATE SET value=value+1")
    finally:
        connection.close()


def _monitoring_snapshot():
    connection = _db()
    try:
        totals = dict(connection.execute("SELECT key,value FROM service_totals"))
        classes = dict(connection.execute("SELECT label,count FROM prediction_classes"))
        counts = connection.execute("SELECT feature,bin_index,count FROM feature_bins").fetchall()
    finally:
        connection.close()
    by_feature = {}
    for feature, index, count in counts:
        by_feature.setdefault(feature, {})[index] = count
    total = int(totals.get("prediction_requests", 0))
    drift = {}
    if DRIFT_BASELINE and total >= DRIFT_MIN_PREDICTIONS:
        for feature, baseline in DRIFT_BASELINE.items():
            observed = [by_feature.get(feature, {}).get(i, 0) for i in range(len(baseline["counts"]))]
            drift[feature] = round(_psi(baseline["counts"], observed), 6)
    max_psi = max(drift.values(), default=0.0)
    return {
        "model_version": MODEL_VERSION,
        "prediction_requests": total,
        "prediction_errors": int(totals.get("prediction_errors", 0)),
        "prediction_latency_seconds_total": round(totals.get("prediction_latency_seconds_total", 0.0), 6),
        "predictions_by_class": classes,
        "drift": {"available": bool(drift), "psi_by_feature": drift, "threshold": DRIFT_PSI_THRESHOLD,
                  "alert": bool(drift) and max_psi >= DRIFT_PSI_THRESHOLD,
                  "status": "alert" if drift and max_psi >= DRIFT_PSI_THRESHOLD else ("ok" if drift else "insufficient_data_or_baseline")},
    }
# --- Load Model and Preprocessing Artifacts ---
try:
    model = joblib.load(MODEL_DIR / "dementia_model.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    selector = joblib.load(MODEL_DIR / "selector.joblib")
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    print("Please run train.py to generate the model artifacts.")
    model = None # Set to None to handle gracefully

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    """Return readiness for container and deployment health checks."""
    if model is None:
        return jsonify({'status': 'error', 'model_loaded': False}), 503
    return jsonify({'status': 'ok', 'model_loaded': True}), 200


@app.route('/metrics', methods=['GET'])
def metrics():
    """Expose durable aggregate telemetry and feature-drift status."""
    with _metrics_lock:
        snapshot = _monitoring_snapshot()
    return jsonify(snapshot)


@app.route('/predict', methods=['POST'])
def predict():
    started = time.perf_counter()

    # --- Validate and parse input first (before checking model readiness) ---
    try:
        age = float(request.form['age'])
        sex = int(request.form['sex'])
        education = float(request.form['education'])
        ses = float(request.form['ses'])
        mmse = float(request.form['mmse'])
        etiv = float(request.form['etiv'])
        nwbv = float(request.form['nwbv'])
        asf = float(request.form['asf'])
    except (KeyError, ValueError) as e:
        _record_error()
        return jsonify({'error': f'Invalid or missing input field: {e}'}), 400

    input_data = np.array([[age, sex, education, ses, mmse, etiv, nwbv, asf]])
    if not np.isfinite(input_data).all():
        _record_error()
        return jsonify({'error': 'All input values must be finite numbers.'}), 400
    if sex not in (0, 1):
        _record_error()
        return jsonify({'error': 'Sex must be encoded as 0 or 1.'}), 400

    # --- Check model readiness after input is validated ---
    if model is None:
        with _metrics_lock:
            _metrics['prediction_errors'] += 1
        _record_error()
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        # --- Preprocess the input ---
        scaled_data = scaler.transform(input_data)
        selected_data = selector.transform(scaled_data)

        # --- Make prediction ---
        prediction_idx = model.predict(selected_data)[0]
        predicted_class = label_encoder.inverse_transform([prediction_idx])[0]

        probabilities = model.predict_proba(selected_data)[0]
        class_labels = label_encoder.classes_

        # Create a dictionary of class probabilities
        prob_dict = {label: f"{prob*100:.2f}%" for label, prob in zip(class_labels, probabilities)}

        with _metrics_lock:
            _metrics['prediction_requests'] += 1
            _metrics['predictions_by_class'][str(predicted_class)] += 1
            elapsed = time.perf_counter() - started
            _metrics['prediction_latency_seconds_total'] += elapsed
        _record_prediction(input_data[0], predicted_class, elapsed)
        state = _monitoring_snapshot()["drift"]
        if state["alert"]:
            logger.warning("Feature drift alert: PSI threshold exceeded (%s)", state["psi_by_feature"])

        return jsonify({
            'prediction': predicted_class,
            'probabilities': prob_dict
        })

    except Exception as e:
        with _metrics_lock:
            _metrics['prediction_errors'] += 1
        _record_error()
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 400

if __name__ == '__main__':
    # Before running for the first time, make sure to run train.py
    if not (MODEL_DIR / "dementia_model.joblib").exists():
        print("Model not found. Please run train.py first to train and save the model.")
    else:
        app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
