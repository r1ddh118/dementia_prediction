import app as app_module


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Dementia Prediction" in r.data


def test_health_reports_model_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_loaded": True}


def test_predict_valid(client, valid_form):
    r = client.post("/predict", data=valid_form)
    assert r.status_code == 200
    body = r.get_json()
    assert body["prediction"] in {
        "No Dementia", "Mild Cognitive Impairment", "Mild Dementia", "Moderate Dementia"}
    total = sum(float(v.rstrip("%")) for v in body["probabilities"].values())
    assert abs(total - 100) < 0.1


def test_predict_missing_field(client, valid_form):
    del valid_form["mmse"]
    r = client.post("/predict", data=valid_form)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_non_numeric(client, valid_form):
    valid_form["age"] = "abc"
    r = client.post("/predict", data=valid_form)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_rejects_non_finite_values(client, valid_form):
    valid_form["age"] = "nan"
    response = client.post("/predict", data=valid_form)
    assert response.status_code == 400
    assert "finite" in response.get_json()["error"]


def test_predict_model_not_loaded(client, valid_form, monkeypatch):
    monkeypatch.setattr(app_module, "model", None)
    r = client.post("/predict", data=valid_form)
    assert r.status_code == 500
    assert "Model not loaded" in r.get_json()["error"]


def test_predict_rejects_get(client):
    assert client.get("/predict").status_code == 405
