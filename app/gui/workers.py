"""CPU analyses run outside Qt's event loop; figures are created on the UI thread."""

import logging

from PySide6.QtCore import QThread, Signal

from app.analysis import run_analysis
from app.analysis.common import AnalysisError

logger = logging.getLogger(__name__)


class AnalysisWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    completed = Signal(object)
    error = Signal(str)
    validation_error = Signal(str)

    def __init__(self, corpus, kind, params, parent=None):
        super().__init__(parent)
        self.corpus = dict(corpus)
        self.kind = kind
        self.params = dict(params)

    def run(self):
        try:
            self.status.emit(f"Calculating {self.kind}…")
            self.progress.emit(10)
            if self.isInterruptionRequested():
                return
            result = run_analysis(self.corpus, self.kind, self.params)
            if not self.isInterruptionRequested():
                self.progress.emit(100)
                self.completed.emit(result)
        except AnalysisError as exc:
            logger.info("%s cannot run with this corpus: %s", self.kind, exc)
            self.validation_error.emit(str(exc))
        except Exception as exc:
            logger.exception("Analysis failed")
            self.error.emit(str(exc))
