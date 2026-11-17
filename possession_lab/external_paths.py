"""Permit ignored experiment outputs while protecting the source tree."""
from pathlib import Path


def allowed_output(path, repo):
    path, repo = Path(path).resolve(), Path(repo).resolve()
    return not path.is_relative_to(repo) or path.is_relative_to(repo/'data/external/runs')
