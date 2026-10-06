"""Verify the delivered corpus using the real cached Dicta annotations."""

import shutil
from pathlib import Path

from app.analysis import run_analysis
from app.analysis.engine import ANALYSIS_KINDS
from app.nlp.cache import load_cache
from app.project import Project


def test_delivered_example_caches_and_all_analyses(tmp_path):
    source = Path(__file__).resolve().parents[1] / "example_project"
    shutil.copytree(source, tmp_path / "example")
    project = Project.open(tmp_path / "example")
    corpus = {}
    for record in project.texts:
        tokens = load_cache(project, record, project.settings["nlp"])
        assert tokens is not None and not tokens.empty
        corpus[record.filename] = tokens
    assert len(corpus) == 4
    params = {**project.settings["analysis"], "target": next(iter(corpus))}
    for kind in ANALYSIS_KINDS:
        result = run_analysis(corpus, kind, params)
        assert not result.table.empty, kind
