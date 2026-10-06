"""Analysis controls, parameter persistence, charts, and result tables."""

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget

from app.analysis.engine import ANALYSIS_KINDS
from app.gui.tables import DataTable
from app.gui.visualization import VisualizationPanel


def spin(minimum, maximum, value):
    widget = QSpinBox()
    widget.setRange(minimum, maximum)
    widget.setValue(value)
    return widget


class AnalysisDashboard(QWidget):
    run_requested = Signal()
    table_export_requested = Signal()
    chart_export_requested = Signal()
    report_export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.kind = QComboBox()
        self.kind.addItems(ANALYSIS_KINDS)
        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self.run_requested)
        top.addWidget(QLabel("Analysis"))
        top.addWidget(self.kind, 1)
        top.addWidget(self.run_button)
        layout.addLayout(top)
        self.validation_message = QLabel()
        self.validation_message.setWordWrap(True)
        self.validation_message.setTextFormat(Qt.TextFormat.PlainText)
        self.validation_message.setStyleSheet("color: #d97706; padding: 6px;")
        self.validation_message.hide()
        layout.addWidget(self.validation_message)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.controls = QTabWidget()
        self.controls.setMinimumWidth(270)
        features, feature_form = self._page("Features")
        self.field = QComboBox()
        self.field.addItems(["lemma", "word"])
        self.basis = QComboBox()
        self.basis.addItems(["mfw", "pos"])
        self.n_mfw = spin(1, 10000, 200)
        self.mattr = spin(1, 100000, 50)
        self.ngram = spin(1, 8, 2)
        self.stop_words = QPlainTextEdit()
        self.stop_words.setPlaceholderText("One stop word per line")
        self.stop_words.setMaximumHeight(170)
        self.stop_words.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        feature_form.addRow("Token representation", self.field)
        feature_form.addRow("Multivariate features", self.basis)
        feature_form.addRow("Top N features", self.n_mfw)
        feature_form.addRow("MATTR window", self.mattr)
        feature_form.addRow("N-gram size", self.ngram)
        feature_form.addRow("Stop words", self.stop_words)
        note = QLabel("Stop words affect lexical analyses. POS includes punctuation. PCA needs 3 texts for 2D and 4 for 3D, with enough variable features.")
        note.setWordWrap(True)
        feature_form.addRow(note)

        _, clustering_form = self._page("Clustering")
        self.distance = QComboBox()
        self.distance.addItems(["delta", "euclidean"])
        self.linkage = QComboBox()
        self.linkage.addItems(["average", "complete", "single", "ward"])
        self.perplexity = QDoubleSpinBox()
        self.perplexity.setRange(0.1, 1000)
        self.perplexity.setValue(30)
        self.seed = spin(0, 2147483647, 42)
        self.color_by = QComboBox()
        self.color_by.addItems(["auto", "book", "author", "genre", "period", "date", "chapter", "none"])
        self.color_by.setToolTip("Auto chooses populated metadata. Changing the group recolors the current scatter plot.")
        clustering_form.addRow("HCA distance", self.distance)
        clustering_form.addRow("HCA linkage", self.linkage)
        clustering_form.addRow("t-SNE perplexity", self.perplexity)
        clustering_form.addRow("Random seed", self.seed)
        clustering_form.addRow("Scatter color by", self.color_by)
        note = QLabel("Ward requires Euclidean distance. t-SNE perplexity is reduced automatically for small corpora.")
        note.setWordWrap(True)
        clustering_form.addRow(note)

        _, windows_form = self._page("Rolling")
        self.target = QComboBox()
        self.window = spin(2, 1000000, 500)
        self.step = spin(1, 1000000, 250)
        windows_form.addRow("Target document", self.target)
        windows_form.addRow("Window (lexical tokens)", self.window)
        windows_form.addRow("Step (lexical tokens)", self.step)
        note = QLabel("Check the target and at least two reference texts in the explorer. Rolling Delta fits vocabulary and scaling on references only; stylistic changes require interpretation.")
        note.setWordWrap(True)
        windows_form.addRow(note)
        splitter.addWidget(self.controls)

        right = QSplitter(Qt.Orientation.Vertical)
        self.chart = VisualizationPanel()
        self.results = DataTable()
        right.addWidget(self.chart)
        right.addWidget(self.results)
        right.setSizes([430, 220])
        splitter.addWidget(right)
        splitter.setSizes([280, 900])
        layout.addWidget(splitter, 1)
        self.description = QLabel("Select processed texts using the checkboxes in the explorer.")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        exports = QHBoxLayout()
        self.export_buttons = []
        for caption, signal in [("Export table…", self.table_export_requested), ("Export chart…", self.chart_export_requested), ("Export report", self.report_export_requested)]:
            button = QPushButton(caption)
            button.clicked.connect(signal)
            button.setEnabled(False)
            self.export_buttons.append(button)
            exports.addWidget(button)
        exports.addStretch()
        layout.addLayout(exports)

    def _page(self, caption):
        page = QWidget()
        form = QFormLayout(page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.controls.addTab(scroll, caption)
        return page, form

    def params(self):
        return {"field": self.field.currentText(), "feature_type": self.basis.currentText(), "n_mfw": self.n_mfw.value(), "mattr_window": self.mattr.value(), "ngram_n": self.ngram.value(), "stop_words": self.stop_words.toPlainText().splitlines(), "distance": self.distance.currentText(), "linkage": self.linkage.currentText(), "perplexity": self.perplexity.value(), "random_state": self.seed.value(), "target": self.target.currentText(), "window_size": self.window.value(), "step_size": self.step.value(), "color_by": self.color_by.currentText(), "kind": self.kind.currentText()}

    def load_settings(self, values):
        for widget, key in [(self.field, "field"), (self.basis, "feature_type"), (self.distance, "distance"), (self.linkage, "linkage"), (self.color_by, "color_by"), (self.kind, "kind")]:
            value = values.get(key, values.get("feature") if key == "field" else None)
            if value is not None and widget.findText(str(value)) >= 0:
                widget.setCurrentText(str(value))
        for widget, key in [(self.n_mfw, "n_mfw"), (self.mattr, "mattr_window"), (self.ngram, "ngram_n"), (self.window, "window_size"), (self.step, "step_size"), (self.seed, "random_state"), (self.perplexity, "perplexity")]:
            if key in values:
                try:
                    widget.setValue(float(values[key]) if isinstance(widget, QDoubleSpinBox) else int(values[key]))
                except (ValueError, TypeError, OverflowError):
                    pass
        words = values.get("stop_words", [])
        self.stop_words.setPlainText("\n".join(map(str, words)) if isinstance(words, list) else "")

    def set_targets(self, labels):
        old = self.target.currentText()
        self.target.clear()
        self.target.addItems(labels)
        if old in labels:
            self.target.setCurrentText(old)

    def set_busy(self, busy):
        self.controls.setEnabled(not busy)
        self.kind.setEnabled(not busy)
        self.run_button.setEnabled(not busy)

    def set_error(self, message):
        self.validation_message.setText(message)
        self.validation_message.setVisible(bool(message))
