"""Static checks so a broken Dockerfile / workflow is caught in CI before deploy."""
import re

import yaml


def test_dockerfile_gunicorn_target_exists(root):
    df = (root / "Dockerfile").read_text()
    assert "gunicorn" in df
    assert "--chdir" in df and "src" in df and "app:app" in df
    assert (root / "src" / "app.py").is_file()


def test_dockerfile_copies_existing_paths(root):
    df = (root / "Dockerfile").read_text()
    for src in re.findall(r"^COPY (?!--from)(\S+)", df, flags=re.M):
        assert (root / src.rstrip("/")).exists(), f"COPY source {src} missing"


def test_requirements_pin_sklearn_to_model_version(root):
    reqs = (root / "requirements.txt").read_text()
    assert "scikit-learn==1.7.2" in reqs


def test_workflows_are_valid_yaml_and_separate(root):
    wf = root / ".github" / "workflows"
    for name in ("ci.yml", "cd.yml"):
        doc = yaml.safe_load((wf / name).read_text())
        assert "jobs" in doc


def test_cd_uses_required_secrets(root):
    cd = (root / ".github" / "workflows" / "cd.yml").read_text()
    for s in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
        assert f"secrets.{s}" in cd
