"""QThread workers: model loading, corpus processing, and cache reads off the UI."""

from __future__ import annotations

import copy
import hashlib
import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

from app.nlp.cache import cache_fingerprint, load_cache, save_cache
from app.nlp.pipeline import DictaPipeline, ProcessingCancelled
from app.utils.io import decode_hebrew_text

if TYPE_CHECKING:
    from app.project import Project

logger = logging.getLogger(__name__)


class CorpusWorker(QThread):
    """Common signals; started and finished are the built-in QThread signals."""

    progress = Signal(int)
    status = Signal(str)
    document_ready = Signal(str, object)
    document_error = Signal(str, str)
    error = Signal(str)
    completed = Signal(object)
    cancelled = Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        # Snapshot GUI-editable state: settings and metadata cannot race a job.
        self.project = copy.deepcopy(project)
        self._last_progress = -1

    def _report_progress(self, value: int) -> None:
        """Queue at most one signal per integer percentage across the whole job."""
        value = max(0, min(100, value))
        if value > self._last_progress:
            self._last_progress = value
            self.progress.emit(value)

    def _cancel_if_requested(self) -> None:
        if self.isInterruptionRequested():
            raise ProcessingCancelled()

    def _document_failed(self, record, exc: Exception) -> None:
        if isinstance(exc, FileNotFoundError):
            logger.error("Failed to process %s: %s", record.filename, exc)
        else:
            logger.exception("Failed to process %s", record.filename)
        self.document_error.emit(record.text_id, f"{record.filename}: {exc}")


class CacheLoadWorker(CorpusWorker):
    """Read validated CSVs on project open without importing model dependencies."""

    def run(self) -> None:
        results = {}
        try:
            texts = self.project.texts
            self._report_progress(0)
            for index, record in enumerate(texts):
                self._cancel_if_requested()
                self.status.emit(f"Checking cache: {record.filename}")
                try:
                    if not (self.project.raw_dir / record.filename).is_file():
                        raise FileNotFoundError(f"Project source file is missing: {record.filename}")
                    frame = load_cache(self.project, record, self.project.settings["nlp"])
                    self._cancel_if_requested()
                    if frame is not None:
                        results[record.text_id] = frame
                        self.document_ready.emit(record.text_id, frame)
                except ProcessingCancelled:
                    raise
                except Exception as exc:
                    self._document_failed(record, exc)
                self._report_progress(round(100 * (index + 1) / max(len(texts), 1)))
            self._cancel_if_requested()
            self._report_progress(100)
            self.completed.emit(results)
        except ProcessingCancelled:
            self.status.emit("Cache loading cancelled.")
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Cache loading failed")
            self.error.emit(str(exc))


class DictaWorker(CorpusWorker):
    """Resolve all caches before loading one Dicta model for uncached documents."""

    def __init__(self, project: Project, force: bool = False, parent=None):
        super().__init__(project, parent)
        self.force = force

    def run(self) -> None:
        results, pending = {}, []
        try:
            texts, settings = self.project.texts, self.project.settings["nlp"]
            self._report_progress(0)
            for index, record in enumerate(texts):
                self._cancel_if_requested()
                self.status.emit(f"Checking cache: {record.filename}")
                try:
                    if not (self.project.raw_dir / record.filename).is_file():
                        raise FileNotFoundError(f"Project source file is missing: {record.filename}")
                    frame = None if self.force else load_cache(self.project, record, settings)
                    if frame is None:
                        pending.append(record)
                    else:
                        results[record.text_id] = frame
                        self.document_ready.emit(record.text_id, frame)
                except Exception as exc:
                    self._document_failed(record, exc)
                self._report_progress(round(10 * (index + 1) / max(len(texts), 1)))
            self._cancel_if_requested()
            if pending:
                self.status.emit("Loading Dicta NLP model (the first run downloads model files)…")
                pipeline = DictaPipeline(settings)
                pipeline.load()
                self._cancel_if_requested()
                for index, record in enumerate(pending):
                    self._cancel_if_requested()
                    self.status.emit(f"Processing {index + 1}/{len(pending)}: {record.filename}")
                    try:
                        # Ensure a user/external edit during inference cannot stamp
                        # old tokens with the new source's cache fingerprint.
                        source = (self.project.raw_dir / record.filename).read_bytes()
                        before = cache_fingerprint(
                            self.project, record, settings,
                            source_sha256=hashlib.sha256(source).hexdigest(),
                        )
                        text = decode_hebrew_text(source)
                        del source
                        frame = pipeline.process(
                            text,
                            progress=lambda percent, i=index: self._report_progress(
                                min(99, round(10 + 90 * (i + percent / 100) / len(pending)))),
                            is_cancelled=self.isInterruptionRequested,
                        )
                        self._cancel_if_requested()
                        # save_cache rehashes once to check for external edits,
                        # then writes the completed DataFrame in one bulk call.
                        save_cache(self.project, record, settings, frame, expected_fingerprint=before)
                        results[record.text_id] = frame
                        self.document_ready.emit(record.text_id, frame)
                    except ProcessingCancelled:
                        raise
                    except Exception as exc:
                        self._document_failed(record, exc)
                    self._report_progress(min(99, round(10 + 90 * (index + 1) / len(pending))))
            self._cancel_if_requested()
            self._report_progress(100)
            self.status.emit(f"Finished: {len(results)}/{len(texts)} texts available.")
            self.completed.emit(results)
        except ProcessingCancelled:
            self.status.emit("Processing cancelled. Completed document caches are retained.")
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Dicta batch processing failed")
            self.error.emit(str(exc))


# Preserve the import used by existing scripts and integrations.
ProcessingWorker = DictaWorker
