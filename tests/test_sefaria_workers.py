"""Sefaria jobs run off the GUI thread and publish coherent terminal results."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication

from app.project import CONFIG_FILENAME, Project
from app.services.sefaria import Book, BookDetails, DownloadBundle, DownloadedText, DownloadRequest, Edition, SefariaCancelled
from app.services.workers import SefariaImportResult, SefariaWorker


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def book():
    return Book("Genesis", "בראשית", ("Tanakh", "Torah"))


@pytest.fixture
def download_request(book):
    return DownloadRequest(book, Edition("Hebrew test edition", license="CC-BY"), split_sections=True)


@pytest.fixture
def bundle():
    return DownloadBundle([
        DownloadedText("Genesis 1.txt", "בְּרֵאשִׁית ברא אלהים\n", {"title": "בראשית", "ref": "Genesis 1", "source": "Sefaria"}),
        DownloadedText("Genesis 2.txt", "ויכלו השמים והארץ\n", {"title": "בראשית", "ref": "Genesis 2", "source": "Sefaria"}),
    ], ["Some sections contain no Hebrew text."], 3)


class Recorder(QObject):
    """Real QObject slots verify cross-thread signals arrive on the GUI thread."""

    def __init__(self):
        super().__init__()
        self.received = {
            "completed": [], "errors": [], "cancelled": [], "progress": [],
            "status": [], "started": [], "finished": [], "receiver_threads": [],
        }

    def observe(self, name, value):
        self.received[name].append(value)
        self.received["receiver_threads"].append(QThread.currentThread())

    @Slot(object)
    def completed(self, value):
        self.observe("completed", value)

    @Slot(str)
    def error(self, value):
        self.observe("errors", value)

    @Slot()
    def cancelled(self):
        self.observe("cancelled", True)

    @Slot(int)
    def progress(self, value):
        self.observe("progress", value)

    @Slot(str)
    def status(self, value):
        self.observe("status", value)

    @Slot()
    def started(self):
        self.observe("started", True)

    @Slot()
    def finished(self):
        self.observe("finished", True)


def run_worker(worker, application, *, cancel_after_ms=None):
    recorder = Recorder()
    for signal in ("completed", "cancelled", "progress", "status", "started", "finished"):
        getattr(worker, signal).connect(getattr(recorder, signal))
    worker.error.connect(recorder.error)
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    worker.finished.connect(loop.quit)
    timeout.start(3000)
    worker.start()
    if cancel_after_ms is not None:
        QTimer.singleShot(cancel_after_ms, worker.requestInterruption)
    loop.exec()
    timeout.stop()
    if worker.isRunning():
        worker.requestInterruption()
    assert worker.wait(2000), "Sefaria worker did not finish."
    application.processEvents()
    received = recorder.received
    assert received["started"] == [True]
    assert received["finished"] == [True]
    assert all(thread is application.thread() for thread in received["receiver_threads"])
    return received


class ClientBase:
    def __init__(self, *, cache_dir):
        self.cache_dir = cache_dir
        self.operations = [("construct", QThread.currentThread())]
        self.close_count = 0

    def observe(self, operation):
        self.operations.append((operation, QThread.currentThread()))

    def close(self):
        self.observe("close")
        self.close_count += 1


def client_factory(client_type):
    clients = []

    def construct(*, cache_dir):
        client = client_type(cache_dir=cache_dir)
        clients.append(client)
        return client

    return construct, clients


@pytest.mark.parametrize("task", ["catalog", "details"])
def test_catalog_and_editions_execute_and_close_in_background(tmp_path, application, book, download_request, task):
    details = BookDetails(book, [download_request.edition], {"title": book.title})
    expected = [book] if task == "catalog" else details

    class BrowsingClient(ClientBase):
        def catalog(self, *, refresh, cancelled):
            self.observe("catalog")
            assert refresh is True
            assert cancelled() is False
            return expected

        def details(self, selected_book, *, cancelled):
            self.observe("details")
            assert selected_book is book
            assert cancelled() is False
            return expected

    factory, clients = client_factory(BrowsingClient)
    cache = tmp_path / "cache"
    received = run_worker(SefariaWorker(task, cache, book=book, refresh=True, client_factory=factory), application)
    assert received["completed"] == [expected]
    assert received["completed"][0] is expected
    assert not received["errors"] and not received["cancelled"]
    assert received["progress"] == [0, 100]
    assert clients[0].cache_dir == cache
    assert clients[0].close_count == 1
    assert [name for name, _ in clients[0].operations] == ["construct", task, "close"]
    assert all(thread is not application.thread() for _, thread in clients[0].operations)


def test_service_error_is_terminal_and_closes_client(tmp_path, application):
    class UnavailableClient(ClientBase):
        def catalog(self, **kwargs):
            self.observe("catalog")
            raise RuntimeError("Sefaria is unavailable")

    factory, clients = client_factory(UnavailableClient)
    received = run_worker(SefariaWorker("catalog", tmp_path, client_factory=factory), application)
    assert received["errors"] == ["Sefaria is unavailable"]
    assert not received["completed"] and not received["cancelled"]
    assert clients[0].close_count == 1


def test_failed_client_construction_emits_error_and_finishes(tmp_path, application):
    threads = []

    def unavailable_client(*, cache_dir):
        threads.append(QThread.currentThread())
        raise OSError("Cannot initialize service")

    received = run_worker(SefariaWorker("catalog", tmp_path, client_factory=unavailable_client), application)
    assert received["errors"] == ["Cannot initialize service"]
    assert not received["completed"] and not received["cancelled"]
    assert threads[0] is not application.thread()


@pytest.mark.parametrize(
    ("task", "expected_error"),
    [("details", "Select a Sefaria book"), ("download", "Open a project"), ("unknown", "Unknown Sefaria task")],
)
def test_invalid_job_parameters_emit_error_and_release_client(tmp_path, application, task, expected_error):
    factory, clients = client_factory(ClientBase)
    received = run_worker(SefariaWorker(task, tmp_path, client_factory=factory), application)
    assert len(received["errors"]) == 1 and expected_error in received["errors"][0]
    assert not received["completed"] and not received["cancelled"]
    assert clients[0].close_count == 1


def test_client_cleanup_error_is_logged_without_losing_result(tmp_path, application, book, caplog):
    class BrokenCleanupClient(ClientBase):
        def catalog(self, **kwargs):
            return [book]

        def close(self):
            super().close()
            raise OSError("Cleanup failed")

    factory, clients = client_factory(BrokenCleanupClient)
    received = run_worker(SefariaWorker("catalog", tmp_path, client_factory=factory), application)
    assert received["completed"] == [[book]]
    assert not received["errors"] and not received["cancelled"]
    assert clients[0].close_count == 1
    assert "Cleanup failed" in caplog.text


def test_successful_import_publishes_snapshot_without_mutating_original(
    tmp_path, application, download_request, bundle
):
    project = Project.create(tmp_path / "project")
    [existing] = project.import_text_content([("existing.txt", "מקור", {"author": "Original"})])

    class DownloadClient(ClientBase):
        def download(self, selected_request, *, progress, status, cancelled):
            self.observe("download")
            assert selected_request is download_request
            assert cancelled() is False
            status("Fetching Hebrew chapters")
            for value in (0, 20, 100):
                progress(value)
            return bundle

    factory, clients = client_factory(DownloadClient)
    worker = SefariaWorker("download", tmp_path / "cache", request=download_request, project=project, client_factory=factory)
    # Mutable metadata/settings are independent after the job captures its input.
    existing.metadata["author"] = "Changed in original object"
    project.settings["analysis"]["stop_words"].append("של")
    received = run_worker(worker, application)
    [result] = received["completed"]
    assert isinstance(result, SefariaImportResult)
    assert result.project is not project
    assert project.texts == [existing]
    assert result.project.texts[0] is not existing
    assert result.project.texts[0].metadata["author"] == "Original"
    assert result.project.settings["analysis"]["stop_words"] == []
    assert len(result.records) == 2 and len(result.project.texts) == 3
    assert result.warnings == bundle.warnings
    assert result.sections_count == bundle.sections_count
    assert received["progress"] == sorted(received["progress"])
    assert received["progress"][-2:] == [90, 100]
    assert "Fetching Hebrew chapters" in received["status"]
    assert any("Importing 2 texts" in status for status in received["status"])
    assert not received["errors"] and not received["cancelled"]
    reopened = Project.open(project.root)
    assert [record.text_id for record in reopened.texts] == [record.text_id for record in result.project.texts]
    for document, record in zip(bundle.documents, result.records):
        assert (project.raw_dir / record.filename).read_text(encoding="utf-8") == document.text
    assert clients[0].close_count == 1
    assert all(thread is not application.thread() for _, thread in clients[0].operations)


def test_cooperative_download_cancellation_imports_nothing(tmp_path, application, download_request, bundle):
    project = Project.create(tmp_path / "project")
    previous_config = (project.root / CONFIG_FILENAME).read_bytes()

    class CancellableClient(ClientBase):
        def download(self, selected_request, *, progress, status, cancelled):
            self.observe("download")
            for _ in range(500):
                if cancelled():
                    raise SefariaCancelled("Cancelled")
                time.sleep(0.002)
            return bundle

    factory, clients = client_factory(CancellableClient)
    worker = SefariaWorker("download", tmp_path, request=download_request, project=project, client_factory=factory)
    received = run_worker(worker, application, cancel_after_ms=10)
    assert received["cancelled"] == [True]
    assert not received["completed"] and not received["errors"]
    assert project.texts == [] and worker.project.texts == []
    assert list(project.raw_dir.iterdir()) == []
    assert (project.root / CONFIG_FILENAME).read_bytes() == previous_config
    assert clients[0].close_count == 1


def test_cancellation_after_download_is_checked_before_import(tmp_path, application, download_request, bundle):
    project = Project.create(tmp_path / "project")

    class ReturningCancelledClient(ClientBase):
        def download(self, selected_request, **kwargs):
            QThread.currentThread().requestInterruption()
            return bundle

    factory, clients = client_factory(ReturningCancelledClient)
    worker = SefariaWorker("download", tmp_path, request=download_request, project=project, client_factory=factory)
    received = run_worker(worker, application)
    assert received["cancelled"] == [True]
    assert not received["completed"] and not received["errors"]
    assert worker.project.texts == []
    assert list(project.raw_dir.iterdir()) == []
    assert clients[0].close_count == 1


def test_cancellation_during_final_commit_publishes_coherent_import(
    tmp_path, application, download_request, bundle, monkeypatch
):
    project = Project.create(tmp_path / "project")
    original_import = Project.import_text_content
    import_threads = []

    def cancelling_import(project_snapshot, documents):
        import_threads.append(QThread.currentThread())
        QThread.currentThread().requestInterruption()
        return original_import(project_snapshot, documents)

    monkeypatch.setattr(Project, "import_text_content", cancelling_import)

    class DownloadClient(ClientBase):
        def download(self, selected_request, **kwargs):
            return bundle

    factory, clients = client_factory(DownloadClient)
    worker = SefariaWorker("download", tmp_path, request=download_request, project=project, client_factory=factory)
    received = run_worker(worker, application)
    [result] = received["completed"]
    assert len(result.records) == 2
    assert not received["errors"] and not received["cancelled"]
    assert len(Project.open(project.root).texts) == 2
    assert project.texts == []
    assert len(list(project.raw_dir.iterdir())) == 2
    assert received["progress"][-1] == 100
    assert import_threads[0] is not application.thread()
    assert clients[0].close_count == 1


def test_import_error_emits_no_completed_result_and_closes_client(
    tmp_path, application, download_request, bundle, monkeypatch
):
    project = Project.create(tmp_path / "project")

    def failing_import(project_snapshot, documents):
        raise OSError("Project import failed")

    monkeypatch.setattr(Project, "import_text_content", failing_import)

    class DownloadClient(ClientBase):
        def download(self, selected_request, **kwargs):
            return bundle

    factory, clients = client_factory(DownloadClient)
    received = run_worker(
        SefariaWorker("download", tmp_path, request=download_request, project=project, client_factory=factory), application,
    )
    assert received["errors"] == ["Project import failed"]
    assert not received["completed"] and not received["cancelled"]
    assert clients[0].close_count == 1
