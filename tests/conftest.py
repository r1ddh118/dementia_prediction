import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture()
def client():
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture()
def valid_form():
    return {
        "age": "70", "sex": "0", "education": "12", "ses": "3",
        "mmse": "25", "etiv": "1500", "nwbv": "0.75", "asf": "1.10",
    }
