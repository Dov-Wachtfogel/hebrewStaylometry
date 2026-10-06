"""Worker lifecycle, background execution, errors, caching, and cancellation."""

import time

import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QThread, QTimer
from PySide6.QtWidgets import QApplication

from app.nlp.cache import load_cache, save_cache
from app.nlp.pipeline import ProcessingCancelled, TOKEN_COLUMNS
from app.nlp.workers import CacheLoadWorker, DictaWorker, ProcessingWorker
from app.project import Project, TextRecord


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def project_with_texts(tmp_path, count=2):
    project = Project.create(tmp_path / "corpus")
    for i in range(count):
        record = TextRecord(f"doc{i}", f"מחבר_ספר{i}_1900.txt")
        project.texts.append(record)
        (project.raw_dir / record.filename).write_text("שלום עולם", encoding="utf-8")
    project.save()
    return project


def frame():
    return pd.DataFrame([[1, "שלום", "שלום", "NOUN", "{}", 1]], columns=TOKEN_COLUMNS)


def run_worker(worker, cancel_after_ms=None):
    received = {"completed": [], "ready": [], "errors": [], "doc_errors": [],
                "cancelled": [], "progress": [], "started": [], "finished": []}
    loop = QEventLoop()
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    worker.completed.connect(received["completed"].append)
    worker.document_ready.connect(lambda identifier, data: received["ready"].append((identifier, data)))
    worker.error.connect(received["errors"].append)
    worker.document_error.connect(lambda identifier, message: received["doc_errors"].append((identifier, message)))
    worker.cancelled.connect(lambda: received["cancelled"].append(True))
    worker.progress.connect(received["progress"].append)
    worker.started.connect(lambda: received["started"].append(True))
    worker.finished.connect(lambda: received["finished"].append(True))
    worker.finished.connect(loop.quit)
    timer.start(5000)
    worker.start()
    if cancel_after_ms is not None:
        QTimer.singleShot(cancel_after_ms, worker.requestInterruption)
    loop.exec()
    timer.stop()
    assert worker.wait(1000), "Worker failed to terminate."
    QApplication.processEvents()
    assert received["started"] == [True] and received["finished"] == [True]
    return received


def test_model_load_and_processing_use_background_thread_and_cache(tmp_path, monkeypatch, application):
    project = project_with_texts(tmp_path)
    threads = []

    class FakePipeline:
        def __init__(self, settings):
            pass

        def load(self):
            threads.append(QThread.currentThread())

        def process(self, text, progress=None, is_cancelled=None):
            threads.append(QThread.currentThread())
            progress(100)
            return frame()

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", FakePipeline)
    received = run_worker(ProcessingWorker(project))
    assert len(threads) == 3 and all(thread is not application.thread() for thread in threads)
    assert len(received["ready"]) == 2
    assert set(received["completed"][0]) == {"doc0", "doc1"}
    assert received["progress"] == sorted(received["progress"])
    assert received["progress"][-1] == 100
    assert not received["errors"] and not received["doc_errors"]
    for record in project.texts:
        assert load_cache(project, record, project.settings["nlp"]) is not None


def test_fully_cached_corpus_never_constructs_model(tmp_path, monkeypatch, application):
    project = project_with_texts(tmp_path)
    for record in project.texts:
        save_cache(project, record, project.settings["nlp"], frame())

    def forbidden_pipeline(settings):
        pytest.fail("Model constructed despite a completely cached corpus.")

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", forbidden_pipeline)
    received = run_worker(ProcessingWorker(project))
    assert set(received["completed"][0]) == {"doc0", "doc1"}
    reopened = run_worker(CacheLoadWorker(project))
    assert set(reopened["completed"][0]) == {"doc0", "doc1"}


def test_document_error_continues_other_documents(tmp_path, monkeypatch, application):
    project = project_with_texts(tmp_path)

    class FakePipeline:
        def __init__(self, settings):
            self.calls = 0

        def load(self):
            pass

        def process(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("Invalid model output")
            return frame()

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", FakePipeline)
    received = run_worker(ProcessingWorker(project))
    assert received["doc_errors"][0][0] == "doc0"
    assert "Invalid model output" in received["doc_errors"][0][1]
    assert set(received["completed"][0]) == {"doc1"}
    assert not received["errors"]


def test_model_load_error_is_terminal(tmp_path, monkeypatch, application):
    project = project_with_texts(tmp_path)

    class BrokenPipeline:
        def __init__(self, settings):
            pass

        def load(self):
            raise RuntimeError("Model download unavailable")

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", BrokenPipeline)
    received = run_worker(ProcessingWorker(project))
    assert received["errors"] == ["Model download unavailable"]
    assert not received["completed"] and not received["cancelled"]


def test_processing_cooperative_cancellation_emits_no_completion(tmp_path, monkeypatch, application):
    project = project_with_texts(tmp_path)

    class SlowPipeline:
        def __init__(self, settings):
            pass

        def load(self):
            pass

        def process(self, text, progress=None, is_cancelled=None):
            for i in range(100):
                if is_cancelled():
                    raise ProcessingCancelled()
                time.sleep(0.002)
                progress(i)
            return frame()

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", SlowPipeline)
    received = run_worker(ProcessingWorker(project), cancel_after_ms=10)
    assert received["cancelled"] == [True]
    assert not received["completed"] and not received["errors"]
    assert not received["ready"]


def test_worker_settings_are_a_snapshot(tmp_path, application):
    project = project_with_texts(tmp_path)
    worker = CacheLoadWorker(project)
    project.settings["nlp"]["segmented"] = False
    project.texts.clear()
    assert worker.project.settings["nlp"]["segmented"] is True
    assert len(worker.project.texts) == 2


def test_source_edit_during_inference_never_creates_stale_cache(tmp_path, monkeypatch, application):
    project = project_with_texts(tmp_path, count=1)

    class EditingPipeline:
        def __init__(self, settings):
            pass

        def load(self):
            pass

        def process(self, *args, **kwargs):
            (project.raw_dir / project.texts[0].filename).write_text("טקסט חדש", encoding="utf-8")
            return frame()

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", EditingPipeline)
    received = run_worker(ProcessingWorker(project))
    assert received["completed"] == [{}]
    assert "changed during processing" in received["doc_errors"][0][1]
    assert load_cache(project, project.texts[0], project.settings["nlp"]) is None


def test_progress_is_coalesced_csvs_are_bulk_written_and_sources_rehashed_once(tmp_path, monkeypatch, application):
    from app.nlp import cache

    project = project_with_texts(tmp_path, count=3)
    writes, source_reads = [], []
    original_to_csv = pd.DataFrame.to_csv
    original_digest = cache._file_digest

    def to_csv(table, *args, **kwargs):
        writes.append(QThread.currentThread())
        return original_to_csv(table, *args, **kwargs)

    def digest(path):
        if path.parent == project.raw_dir:
            source_reads.append(path.name)
        return original_digest(path)

    class ChattyPipeline:
        def __init__(self, settings):
            pass

        def load(self):
            pass

        def process(self, text, progress=None, is_cancelled=None):
            for i in range(10000):
                progress(round(i / 100))
            return frame()

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", ChattyPipeline)
    monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
    monkeypatch.setattr(cache, "_file_digest", digest)
    received = run_worker(DictaWorker(project))
    assert not received["errors"] and not received["doc_errors"]
    assert len(writes) == len(project.texts)
    assert all(thread is not application.thread() for thread in writes)
    assert source_reads == [record.filename for record in project.texts]
    assert received["progress"] == sorted(set(received["progress"]))
    assert len(received["progress"]) <= 101
    assert received["progress"][-1] == 100
    assert len(received["ready"]) == 3
    assert ProcessingWorker is DictaWorker


@pytest.mark.parametrize("force", [False, True])
def test_missing_source_is_skipped_before_loading_model(tmp_path, monkeypatch, application, force):
    project = project_with_texts(tmp_path, count=1)
    (project.raw_dir / project.texts[0].filename).unlink()

    def forbidden_pipeline(settings):
        pytest.fail("Missing files must not load a model.")

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", forbidden_pipeline)
    received = run_worker(DictaWorker(project, force=force))
    assert received["completed"] == [{}]
    assert "source file is missing" in received["doc_errors"][0][1]
    assert not received["errors"]
