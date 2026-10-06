"""Main desktop workflow: project management, NLP jobs, analysis, and export."""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PySide6.QtCore import QStandardPaths, QTimer, Qt, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QProgressDialog, QPushButton, QSpinBox, QSplitter, QTableView, QTabWidget, QToolBar, QVBoxLayout, QWidget

from app.gui.dashboard import AnalysisDashboard
from app.gui.models import MetadataTableModel
from app.gui.sefaria import SefariaView
from app.gui.tables import DataTable
from app.gui.workers import AnalysisWorker
from app.nlp.workers import CacheLoadWorker, DictaWorker
from app.project import Project
from app.services.workers import SefariaImportResult, SefariaWorker
from app.utils.io import atomic_write_json, export_table
from app.utils.labels import document_groups

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.project: Project | None = None
        self.frames: dict[str, pd.DataFrame] = {}
        self.result = None
        self.result_params = {}
        self._worker = None
        self._job = ""
        self._pending_close = False
        self._progress_dialog = None
        self.sefaria_cache_dir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation)) / "sefaria"
        self.setWindowTitle("Hebrew Stylometry Studio")
        self.resize(1450, 940)
        self.setMinimumSize(1000, 700)
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.setStyleSheet("QToolBar { spacing: 8px; padding: 8px; } QPushButton { padding: 6px 12px; } QPushButton#primary { background: #2563eb; color: white; border-radius: 4px; } QTabWidget::pane { border: 1px solid #cbd5e1; } QProgressBar { min-height: 18px; }")
        self._build_toolbar()
        self._build_views()
        self._set_busy(False)
        self.statusBar().showMessage("Create or open a project to begin")

    def _build_toolbar(self):
        toolbar = QToolBar("Project")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        self.project_actions = []
        for text, shortcut, handler in [("New project…", "Ctrl+N", self.new_project), ("Open project…", "Ctrl+O", self.choose_project), ("Import texts…", "Ctrl+I", self.import_texts), ("Import metadata…", "", self.import_metadata)]:
            action = QAction(text, self)
            if shortcut:
                action.setShortcut(shortcut)
            action.triggered.connect(handler)
            toolbar.addAction(action)
            self.project_actions.append(action)
        self.sefaria_action = QAction("Download from Sefaria…", self)
        self.sefaria_action.triggered.connect(self.open_sefaria)
        toolbar.addAction(self.sefaria_action)
        self.project_label = QLabel("No project open")
        toolbar.addSeparator()
        toolbar.addWidget(self.project_label)

    def _build_views(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        heading = QLabel("PROJECT EXPLORER")
        sidebar_layout.addWidget(heading)
        note = QLabel("Checked texts form the analysis corpus.")
        note.setWordWrap(True)
        sidebar_layout.addWidget(note)
        self.explorer = QListWidget()
        self.explorer.currentItemChanged.connect(self._select_explorer)
        sidebar_layout.addWidget(self.explorer, 1)
        selection = QHBoxLayout()
        for caption, state in [("Select all", Qt.CheckState.Checked), ("Clear", Qt.CheckState.Unchecked)]:
            button = QPushButton(caption)
            button.clicked.connect(lambda checked=False, value=state: self._check_all(value))
            selection.addWidget(button)
        sidebar_layout.addLayout(selection)
        self.corpus_label = QLabel("0 texts · 0 processed")
        sidebar_layout.addWidget(self.corpus_label)
        sidebar.setMinimumWidth(200)
        splitter.addWidget(sidebar)
        self.tabs = QTabWidget()

        metadata_page = QWidget()
        metadata_layout = QVBoxLayout(metadata_page)
        help_label = QLabel("Double-click metadata cells to edit. Filename convention: Author_Title_Date.txt. Extra fields from metadata.json appear automatically.")
        help_label.setWordWrap(True)
        metadata_layout.addWidget(help_label)
        self.metadata_model = MetadataTableModel(self)
        self.metadata_model.error.connect(self._show_error)
        self.metadata = QTableView()
        self.metadata.setModel(self.metadata_model)
        self.metadata.setAlternatingRowColors(True)
        self.metadata.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.metadata.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.metadata.horizontalHeader().setDefaultSectionSize(150)
        metadata_layout.addWidget(self.metadata, 1)
        self.remove_text_button = QPushButton("Remove selected file from project…")
        self.remove_text_button.clicked.connect(self.remove_selected_text)
        metadata_layout.addWidget(self.remove_text_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.tabs.addTab(metadata_page, "Project & metadata")

        preprocessing = QWidget()
        preprocess_layout = QVBoxLayout(preprocessing)
        self.nlp_controls = QWidget()
        form = QFormLayout(self.nlp_controls)
        self.model_name = QComboBox()
        self.model_name.setEditable(True)
        self.model_name.addItems(["dicta-il/dictabert-tiny-joint", "dicta-il/dictabert-joint"])
        self.revision = QLineEdit("main")
        self.revision.setPlaceholderText("Model revision or pinned commit SHA")
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda", "mps"])
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(32)
        self.batch_size.setToolTip("Use 32 chunks per batch; choose 64 when memory permits. Smaller batches reduce memory use.")
        self.strip_niqqud = QCheckBox("Remove niqqud and cantillation before NLP")
        self.strip_niqqud.setChecked(True)
        self.segmented = QCheckBox("Segment prefixes/suffixes into individual UD tokens")
        self.segmented.setChecked(True)
        self.force = QCheckBox("Reprocess valid caches")
        form.addRow("Dicta model", self.model_name)
        form.addRow("Revision", self.revision)
        form.addRow("Device", self.device)
        form.addRow("Batch size", self.batch_size)
        form.addRow(self.strip_niqqud)
        form.addRow(self.segmented)
        form.addRow(self.force)
        preprocess_layout.addWidget(self.nlp_controls)
        information = QLabel("The model and its Python code are downloaded from Hugging Face on first use and run locally. Use a commit SHA for reproducible model loading. Valid caches are loaded without downloading the model; changed source files or NLP settings require processing again.")
        information.setWordWrap(True)
        preprocess_layout.addWidget(information)
        buttons = QHBoxLayout()
        self.process_button = QPushButton("Process corpus")
        self.process_button.setObjectName("primary")
        self.process_button.clicked.connect(self.process_corpus)
        self.cancel_button = QPushButton("Cancel task")
        self.cancel_button.clicked.connect(self.cancel_task)
        buttons.addWidget(self.process_button)
        buttons.addWidget(self.cancel_button)
        buttons.addStretch()
        preprocess_layout.addLayout(buttons)
        self.task_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        preprocess_layout.addWidget(self.task_label)
        preprocess_layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)
        preprocess_layout.addWidget(self.log, 1)
        self.tabs.addTab(preprocessing, "Preprocessing")

        self.dashboard = AnalysisDashboard()
        self.dashboard.color_by.currentTextChanged.connect(self._recolor_result)
        self.dashboard.run_requested.connect(self.run_analysis)
        self.dashboard.table_export_requested.connect(self.export_results)
        self.dashboard.chart_export_requested.connect(self.export_chart)
        self.dashboard.report_export_requested.connect(self.export_report)
        self.tabs.addTab(self.dashboard, "Analysis")
        tokens_page = QWidget()
        tokens_layout = QVBoxLayout(tokens_page)
        token_header = QHBoxLayout()
        self.token_document = QComboBox()
        self.token_document.currentIndexChanged.connect(self._show_tokens)
        self.token_export = QPushButton("Export tokens…")
        self.token_export.clicked.connect(self.export_tokens)
        token_header.addWidget(QLabel("Document"))
        token_header.addWidget(self.token_document, 1)
        token_header.addWidget(self.token_export)
        tokens_layout.addLayout(token_header)
        self.tokens = DataTable()
        # Token IDs are already in document order. Avoid an initial proxy sort
        # of every token when a large completed document is first displayed.
        self.tokens.view.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        tokens_layout.addWidget(self.tokens, 1)
        self.tabs.addTab(tokens_page, "Tokens")
        self.sefaria = SefariaView()
        self.sefaria.catalog_requested.connect(self.load_sefaria_catalog)
        self.sefaria.editions_requested.connect(self.load_sefaria_editions)
        self.sefaria.download_requested.connect(self.download_from_sefaria)
        self.sefaria.cancel_requested.connect(self.cancel_task)
        self.tabs.addTab(self.sefaria, "Sefaria")
        splitter.addWidget(self.tabs)
        splitter.setSizes([250, 1200])
        self.setCentralWidget(splitter)

    def _check_all(self, state):
        for row in range(self.explorer.count()):
            self.explorer.item(row).setCheckState(state)

    def _refresh_project(self):
        checked = {self.explorer.item(row).data(Qt.ItemDataRole.UserRole) for row in range(self.explorer.count()) if self.explorer.item(row).checkState() == Qt.CheckState.Checked}
        existing = {self.explorer.item(row).data(Qt.ItemDataRole.UserRole) for row in range(self.explorer.count())}
        old_tokens = self.token_document.currentData()
        self.explorer.clear()
        self.token_document.blockSignals(True)
        self.token_document.clear()
        if self.project:
            self.project_label.setText(self.project.name)
            self.project_label.setToolTip(str(self.project.root))
            self.setWindowTitle(f"{self.project.name} — Hebrew Stylometry Studio")
            for record in self.project.texts:
                item = QListWidgetItem(record.filename)
                item.setData(Qt.ItemDataRole.UserRole, record.text_id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if record.text_id not in existing or record.text_id in checked else Qt.CheckState.Unchecked)
                self.explorer.addItem(item)
                self.token_document.addItem(record.filename, record.text_id)
            self.metadata_model.set_project(self.project, self.metadata_model.statuses)
            labels = [record.filename for record in self.project.texts if record.text_id in self.frames]
            self.dashboard.set_targets(labels)
        self.token_document.blockSignals(False)
        index = self.token_document.findData(old_tokens)
        if index >= 0:
            self.token_document.setCurrentIndex(index)
        self._show_tokens()
        self._update_count()

    def _update_count(self):
        total = len(self.project.texts) if self.project else 0
        self.corpus_label.setText(f"{total} texts · {len(self.frames)} processed")

    def new_project(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a new or empty project folder")
        if folder:
            try:
                self._attach_project(Project.create(Path(folder)))
            except (OSError, ValueError) as exc:
                self._show_error(str(exc))

    def choose_project(self):
        folder = QFileDialog.getExistingDirectory(self, "Open project folder")
        if folder:
            self.open_project(Path(folder))

    def open_project(self, path: Path):
        if self._worker is not None:
            return
        try:
            self._attach_project(Project.open(path))
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))

    def _attach_project(self, project):
        self.project = project
        self.frames.clear()
        self.result = None
        self.result_params = {}
        self.result_corpus = []
        self.explorer.clear()
        self.metadata_model.set_project(project)
        self.dashboard.results.set_frame(pd.DataFrame())
        self.dashboard.chart.clear()
        self.dashboard.set_error("")
        for button in self.dashboard.export_buttons:
            button.setEnabled(False)
        self.dashboard.load_settings(project.settings.get("analysis", {}))
        nlp = project.settings["nlp"]
        self.model_name.setCurrentText(nlp["model_name"])
        self.revision.setText(nlp["revision"])
        self.device.setCurrentText(nlp["device"])
        self.batch_size.setValue(nlp["batch_size"])
        self.strip_niqqud.setChecked(nlp["strip_niqqud"])
        self.segmented.setChecked(nlp["segmented"])
        self.log.clear()
        self._refresh_project()
        self._start_job(CacheLoadWorker(project, parent=self), "Loading cached tokens")

    def import_texts(self):
        if self.project is None:
            return
        filenames, _ = QFileDialog.getOpenFileNames(self, "Import Hebrew texts", str(self.project.root), "Text files (*.txt)")
        if filenames:
            try:
                imported = self.project.import_texts(Path(filename) for filename in filenames)
                self._refresh_project()
                self._log(f"Imported {len(imported)} texts. Open Preprocessing to process them.")
            except (OSError, ValueError) as exc:
                self._show_error(str(exc))

    def import_metadata(self):
        if self.project is None:
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Import metadata JSON", str(self.project.root), "JSON files (*.json)")
        if filename:
            try:
                count = self.project.import_metadata(Path(filename))
                self._refresh_project()
                self._log(f"Updated metadata for {count} texts.")
            except (OSError, ValueError) as exc:
                self._show_error(str(exc))

    def remove_selected_text(self):
        if self.project is None or self._worker is not None:
            return
        rows = self.metadata.selectionModel().selectedRows()
        if not rows:
            self._show_error("Select a file in the project metadata table first.")
            return
        record = self.project.texts[rows[0].row()]
        answer = QMessageBox.question(
            self,
            "Remove file from project",
            f"Remove {record.filename} from this project? Its project copy and processed token cache will also be deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self.project.remove_text(record.text_id)
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))
            return
        self.frames.pop(removed.text_id, None)
        self.metadata_model.statuses.pop(removed.text_id, None)
        self.result = None
        self.result_params = {}
        self.result_corpus = []
        self.dashboard.results.set_frame(pd.DataFrame())
        self.dashboard.chart.clear()
        self.dashboard.set_error("")
        for button in self.dashboard.export_buttons:
            button.setEnabled(False)
        self._refresh_project()
        self._log(f"Removed {removed.filename} and its project data.")

    def process_corpus(self):
        if self.project is None or self._worker is not None:
            return
        if not self.project.texts:
            self._show_error("Import at least one .txt file first.")
            return
        settings = {**self.project.settings["nlp"], "backend": "dicta", "model_name": self.model_name.currentText().strip(), "revision": self.revision.text().strip() or "main", "device": self.device.currentText(), "batch_size": self.batch_size.value(), "strip_niqqud": self.strip_niqqud.isChecked(), "segmented": self.segmented.isChecked()}
        if not settings["model_name"]:
            self._show_error("Choose a Dicta model name.")
            return
        previous = self.project.settings["nlp"]
        try:
            self.project.settings["nlp"] = settings
            self.project.save()
        except (OSError, ValueError) as exc:
            self.project.settings["nlp"] = previous
            self._show_error(str(exc))
            return
        # Clear old data before processing so failed/reconfigured documents cannot remain analyzable.
        self.frames.clear()
        self.result = None
        self.result_params = {}
        self.dashboard.results.set_frame(pd.DataFrame())
        self.dashboard.chart.clear()
        for button in self.dashboard.export_buttons:
            button.setEnabled(False)
        self.metadata_model.set_project(self.project)
        self._show_tokens()
        self.tabs.setCurrentIndex(1)
        self._start_job(DictaWorker(self.project, force=self.force.isChecked(), parent=self), "Processing corpus")

    def open_sefaria(self):
        self.tabs.setCurrentWidget(self.sefaria)
        if not self.sefaria.model.books and self._worker is None:
            self.load_sefaria_catalog()

    def load_sefaria_catalog(self, refresh=False):
        if self._worker is not None:
            return
        self._start_job(
            SefariaWorker("catalog", self.sefaria_cache_dir, refresh=refresh, parent=self),
            "Loading Sefaria library", self._sefaria_catalog_ready,
        )

    def _sefaria_catalog_ready(self, books):
        self.sefaria.set_catalog(books)
        self._task_status(f"Sefaria library ready: {len(books):,} books. Search and load Hebrew editions.")

    def load_sefaria_editions(self, book):
        if self._worker is not None:
            return
        self._start_job(
            SefariaWorker("details", self.sefaria_cache_dir, book=book, parent=self),
            "Loading Hebrew editions", self._sefaria_editions_ready,
        )

    def _sefaria_editions_ready(self, details):
        self.sefaria.set_details(details)
        self._task_status(f"{details.book.title}: {len(details.editions)} Hebrew editions available.")

    def download_from_sefaria(self, request):
        if self.project is None or self._worker is not None:
            return
        self._start_job(
            SefariaWorker("download", self.sefaria_cache_dir, request=request, project=self.project, parent=self),
            f"Downloading {request.book.title}", self._sefaria_import_complete,
        )

    def _sefaria_import_complete(self, result: SefariaImportResult):
        # Keep existing token frames and analysis results; only the corpus grows.
        self.project = result.project
        self._refresh_project()
        for warning in result.warnings:
            self._log(f"Sefaria: {warning}")
        message = f"Imported {len(result.records)} texts ({result.sections_count:,} sections) from Sefaria. Open Preprocessing to process them."
        if result.warnings:
            message += " " + " ".join(result.warnings)
        self._task_status(message)

    def _start_job(self, worker, title, on_completed=None):
        self._worker = worker
        self._job = title
        self._set_busy(True)
        self.progress.setValue(0)
        self.sefaria.progress.setValue(0)
        self.task_label.setText(title)
        self.sefaria.status.setText(title)
        self._progress_dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        self._progress_dialog.setWindowTitle(title)
        self._progress_dialog.setWindowModality(Qt.WindowModality.NonModal)
        self._progress_dialog.setMinimumDuration(500)
        self._progress_dialog.setAutoClose(False)
        self._progress_dialog.canceled.connect(self.cancel_task)
        worker.progress.connect(self._set_progress, Qt.ConnectionType.QueuedConnection)
        worker.status.connect(self._task_status, Qt.ConnectionType.QueuedConnection)
        worker.error.connect(self._worker_error)
        if isinstance(worker, AnalysisWorker):
            worker.validation_error.connect(self._analysis_error)
        worker.finished.connect(self._finish_job)
        if on_completed is not None:
            worker.completed.connect(on_completed)
        elif hasattr(worker, "document_ready"):
            worker.document_ready.connect(self._document_ready)
            worker.document_error.connect(self._document_error)
            worker.completed.connect(self._processing_complete)
            worker.cancelled.connect(lambda: self._task_status("Cancelled; completed document caches are preserved."))
        else:
            worker.completed.connect(self._analysis_complete)
        worker.finished.connect(worker.deleteLater)
        worker.start()

    @Slot(int)
    def _set_progress(self, value):
        self.progress.setRange(0, 100)
        self.progress.setValue(value)
        self.sefaria.progress.setValue(value)
        if self._progress_dialog:
            self._progress_dialog.setRange(0, 100)
            self._progress_dialog.setValue(value)

    @Slot(str)
    def _task_status(self, message):
        self.task_label.setText(message)
        self.sefaria.status.setText(message)
        self.statusBar().showMessage(message)
        if self._progress_dialog:
            self._progress_dialog.setLabelText(message)
        self._log(message)

    @Slot(str, object)
    def _document_ready(self, text_id, frame):
        self.frames[text_id] = frame
        self.metadata_model.set_status(text_id, f"Ready · {len(frame):,} tokens")
        self._update_count()
        # Updating a large sortable token table can dominate the GUI thread.
        # Display the completed corpus once at the end of the job instead.
        if self._worker is None and self.token_document.currentData() == text_id:
            self._show_tokens()

    @Slot(str, str)
    def _document_error(self, text_id, message):
        self.frames.pop(text_id, None)
        self.metadata_model.set_status(text_id, "Failed")
        record = self.project.get_record(text_id)
        self._log(f"{record.filename}: {message}")

    @Slot(object)
    def _processing_complete(self, frames):
        self.frames = dict(frames)
        self._refresh_project()
        self._task_status(f"Ready: {len(self.frames)} of {len(self.project.texts)} texts have valid cached tokens.")

    @Slot(str)
    def _worker_error(self, message):
        if self._job == "Running analysis":
            self.dashboard.set_error(message)
        self._task_status(f"Task failed: {message}")

    @Slot(str)
    def _analysis_error(self, message):
        self.dashboard.set_error(message)
        self._task_status(f"Analysis needs attention: {message}")

    @Slot()
    def _finish_job(self):
        self._worker = None
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog.deleteLater()
            self._progress_dialog = None
        self._set_busy(False)
        self.dashboard.set_targets([record.filename for record in self.project.texts if record.text_id in self.frames] if self.project else [])
        self._update_count()
        if self._pending_close:
            QTimer.singleShot(0, self.close)

    def _set_busy(self, busy):
        for index, action in enumerate(self.project_actions):
            action.setEnabled(not busy and (index < 2 or self.project is not None))
        self.sefaria_action.setEnabled(not busy)
        self.sefaria.set_project_available(self.project is not None)
        self.sefaria.set_busy(busy)
        self.metadata.setEnabled(not busy)
        self.remove_text_button.setEnabled(not busy and self.project is not None)
        self.nlp_controls.setEnabled(not busy)
        self.process_button.setEnabled(not busy and self.project is not None)
        self.cancel_button.setEnabled(busy)
        self.explorer.setEnabled(not busy)
        self.dashboard.set_busy(busy)
        self.dashboard.run_button.setEnabled(not busy and bool(self.frames))
        for button in self.dashboard.export_buttons:
            button.setEnabled(not busy and self.result is not None)

    def cancel_task(self):
        if self._worker:
            self._worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.sefaria.cancel_button.setEnabled(False)
            self._task_status("Cancellation requested; waiting for the current operation to finish…")

    def _select_explorer(self, item, previous):
        if item:
            index = self.token_document.findData(item.data(Qt.ItemDataRole.UserRole))
            if index >= 0:
                self.token_document.setCurrentIndex(index)

    def _show_tokens(self, index=None):
        frame = self.frames.get(self.token_document.currentData(), pd.DataFrame())
        if self.tokens.model.frame is not frame:
            self.tokens.set_frame(frame)
        self.token_export.setEnabled(not frame.empty)

    def run_analysis(self):
        if self.project is None or self._worker is not None:
            return
        self.dashboard.set_error("")
        selected = {self.explorer.item(row).data(Qt.ItemDataRole.UserRole) for row in range(self.explorer.count()) if self.explorer.item(row).checkState() == Qt.CheckState.Checked}
        missing = [record.filename for record in self.project.texts if record.text_id in selected and record.text_id not in self.frames]
        if missing:
            self._show_error("Preprocess or uncheck these texts first: " + ", ".join(missing))
            return
        corpus = {record.filename: self.frames[record.text_id] for record in self.project.texts if record.text_id in selected and record.text_id in self.frames}
        if not corpus:
            self._show_error("Check at least one processed text in the explorer.")
            return
        kind = self.dashboard.kind.currentText()
        required = {"PCA 2D": 3, "PCA 3D": 4}.get(kind)
        if required is not None and len(corpus) < required:
            self._analysis_error(
                f"{kind} needs at least {required} processed texts; {len(corpus)} are selected. "
                "Download chapters as separate texts, preprocess them, and check them in the explorer. "
                "You can run MFW or POS distribution with one text."
            )
            return
        params = self.dashboard.params()
        previous = copy.deepcopy(self.project.settings.get("analysis", {}))
        try:
            self.project.settings["analysis"] = params
            self.project.save()
        except (OSError, ValueError) as exc:
            self.project.settings["analysis"] = previous
            self._show_error(str(exc))
            return
        self._active_params = params
        self._active_corpus = list(corpus)
        self._active_metadata = copy.deepcopy([{"filename": record.filename, "metadata": record.metadata} for record in self.project.texts if record.filename in corpus])
        self._active_nlp_settings = copy.deepcopy(self.project.settings["nlp"])
        self._start_job(AnalysisWorker(corpus, kind, params, parent=self), "Running analysis")

    @Slot(object)
    def _analysis_complete(self, result):
        self.result = result
        self.result_params = dict(self._active_params)
        self.result_corpus = list(self._active_corpus)
        self.result_metadata = copy.deepcopy(self._active_metadata)
        self.result_nlp_settings = copy.deepcopy(self._active_nlp_settings)
        self.dashboard.results.set_frame(result.table)
        self.dashboard.description.setText(result.description)
        self._render_result()
        for button in self.dashboard.export_buttons:
            button.setEnabled(True)
        self._task_status(f"Completed {result.kind}: {len(result.table):,} result rows.")

    def _recolor_result(self, field):
        if self.result is not None and self.result.kind in {"PCA 2D", "PCA 3D", "t-SNE"}:
            self.result_params["color_by"] = field
            self._render_result()

    def _render_result(self):
        metadata = {item["filename"]: item["metadata"] for item in self.result_metadata}
        groups, field = document_groups(self.result_corpus, metadata, self.result_params.get("color_by", "auto"))
        self.result_params["resolved_color_by"] = field or "none"
        try:
            self.dashboard.chart.plot(self.result, groups=groups, metadata=metadata, group_title=field)
        except Exception as exc:
            logger.exception("Chart rendering failed")
            self._log(f"Chart could not be rendered: {exc}")

    def _export_table(self, frame, stem, include_index=True):
        if self.project is None:
            return
        filename, file_filter = QFileDialog.getSaveFileName(self, "Export table", str(self.project.outputs_dir / f"{stem}.csv"), "CSV (*.csv);;Excel (*.xlsx)")
        if filename:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix(".xlsx" if "xlsx" in file_filter else ".csv")
            try:
                export_table(frame, path, index=include_index)
                self.statusBar().showMessage(f"Exported {path}")
            except (OSError, ValueError, ImportError) as exc:
                self._show_error(str(exc))

    def export_results(self):
        if self.result is not None:
            self._export_table(self.dashboard.results.visible_frame(), self.result.kind.replace(" ", "_"))

    def export_tokens(self):
        label = self.token_document.currentText()
        self._export_table(self.tokens.visible_frame(), f"{Path(label).stem}_tokens", include_index=False)

    def export_chart(self):
        if self.result is None or self.project is None:
            return
        filename, file_filter = QFileDialog.getSaveFileName(self, "Export chart", str(self.project.outputs_dir / "chart.png"), "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if filename:
            path = Path(filename)
            if not path.suffix:
                path = path.with_suffix(".svg" if "svg" in file_filter else ".pdf" if "pdf" in file_filter else ".png")
            try:
                self.dashboard.chart.export(path)
            except (OSError, ValueError) as exc:
                self._show_error(str(exc))

    def export_report(self):
        if self.result is None or self.project is None:
            return
        folder = self.project.outputs_dir / datetime.now(timezone.utc).strftime("report_%Y%m%d_%H%M%S_%f")
        try:
            folder.mkdir()
            export_table(self.result.table, folder / "results.csv")
            self.dashboard.chart.export(folder / "chart.png")
            atomic_write_json(folder / "analysis_settings.json", {"analysis": self.result.kind, "description": self.result.description, "parameters": self.result_params, "project": self.project.name, "texts": self.result_metadata, "nlp_settings": self.result_nlp_settings})
            (folder / "report.md").write_text(f"# {self.result.kind}\n\nProject: {self.project.name}\n\n{self.result.description}\n\nFull results: results.csv. Settings and metadata: analysis_settings.json.\n\n![Chart](chart.png)\n", encoding="utf-8")
            self.statusBar().showMessage(f"Report exported to {folder}")
            self._log(f"Report exported to {folder}")
        except (OSError, ValueError) as exc:
            self._show_error(str(exc))

    def _log(self, message):
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {message}")

    def _show_error(self, message):
        self._log(message)
        QMessageBox.warning(self, "Hebrew Stylometry Studio", message)

    def closeEvent(self, event):
        if self._worker is not None:
            self._pending_close = True
            self.cancel_task()
            event.ignore()
        else:
            event.accept()
