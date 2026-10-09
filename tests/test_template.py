"""index.html must send exactly the fields app.py reads."""
import re

FIELDS = ["age", "sex", "education", "ses", "mmse", "etiv", "nwbv", "asf"]


def _html(root):
    return (root / "templates" / "index.html").read_text()


def test_form_has_all_fields_expected_by_backend(root):
    names = set(re.findall(r'name="(\w+)"', _html(root)))
    assert set(FIELDS) <= names


def test_form_fields_match_app_py(root):
    src = (root / "src" / "app.py").read_text()
    used = set(re.findall(r"request\.form\['(\w+)'\]", src))
    assert used == set(FIELDS)


def test_js_posts_to_predict(root):
    assert "fetch('/predict'" in _html(root)


def test_every_form_default_is_accepted(root, client):
    html = _html(root)
    data = {n: v for n, v in re.findall(r'name="(\w+)" value="([\d.]+)"', html)}
    data["sex"] = "0"
    assert client.post("/predict", data=data).status_code == 200
