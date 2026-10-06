"""Background Sefaria browsing/download jobs, including atomic project imports."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.project import Project, TextRecord
from app.services.sefaria import Book, DownloadRequest, SefariaCancelled, SefariaClient

logger = logging.getLogger(__name__)


@dataclass
class SefariaImportResult:
    project: Project
    records: list[TextRecord]
    warnings: list[str]
    sections_count: int


class SefariaWorker(QThread):
    """Only the main thread updates widgets; HTTP and disk I/O stay here."""

    progress = Signal(int)
    status = Signal(str)
    completed = Signal(object)
    error = Signal(str)
    cancelled = Signal()

    def __init__(
        self, task: str, cache_dir: Path, *, book: Book | None = None,
        request: DownloadRequest | None = None, project: Project | None = None,
        refresh: bool = False, parent=None, client_factory=None,
    ):
        super().__init__(parent)
        self.task = task
        self.cache_dir = cache_dir
        self.book = book
        self.request = request
        self.project = copy.deepcopy(project) if project is not None else None
        self.refresh = refresh
        self.client_factory = client_factory or SefariaClient

    def _check_cancelled(self):
        if self.isInterruptionRequested():
            raise SefariaCancelled()

    def run(self):
        client = None
        try:
            self._check_cancelled()
            client = self.client_factory(cache_dir=self.cache_dir)
            self.progress.emit(0)
            if self.task == "catalog":
                self.status.emit("Loading Sefaria library catalog…")
                result = client.catalog(refresh=self.refresh, cancelled=self.isInterruptionRequested)
            elif self.task == "details":
                if self.book is None:
                    raise ValueError("Select a Sefaria book first.")
                self.status.emit(f"Finding Hebrew editions: {self.book.title}")
                result = client.details(self.book, cancelled=self.isInterruptionRequested)
            elif self.task == "download":
                if self.request is None or self.project is None:
                    raise ValueError("Open a project and choose a Hebrew edition first.")
                bundle = client.download(
                    self.request,
                    progress=lambda value: self.progress.emit(round(value * 0.9)),
                    status=self.status.emit, cancelled=self.isInterruptionRequested,
                )
                self._check_cancelled()
                self.status.emit(f"Importing {len(bundle.documents)} texts into the project…")
                # Once the atomic import begins it finishes coherently, even if
                # cancellation is requested during the final disk write.
                records = self.project.import_text_content(
                    (item.filename, item.text, item.metadata) for item in bundle.documents
                )
                result = SefariaImportResult(self.project, records, bundle.warnings, bundle.sections_count)
            else:
                raise ValueError(f"Unknown Sefaria task: {self.task}")
            if self.task != "download":
                self._check_cancelled()
            self.progress.emit(100)
            self.completed.emit(result)
        except SefariaCancelled:
            self.status.emit("Sefaria task cancelled; no incomplete download was imported.")
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Sefaria task failed")
            self.error.emit(str(exc))
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    logger.exception("Could not close the Sefaria HTTP session")
