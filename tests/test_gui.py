"""Headless Qt integration checks for the desktop workflow and chart exports."""

import json
import time

import pandas as pd
import pytest
from PySide6.QtCore import QThread, QTimer, Qt

from app.analysis import run_analysis
from app.analysis.engine import ANALYSIS_KINDS
from app.gui.main_window import MainWindow
from app.gui.models import MetadataTableModel, PandasTableModel
from app.gui.visualization import VisualizationPanel
from app.nlp.cache import save_cache
from app.project import Project
from app.utils.labels import document_display_name


def frame(words, tags=None):
    return pd.DataFrame({"token_id": range(1, len(words) + 1), "word": words, "lemma": words, "pos_tag": tags or ["NOUN"] * len(words), "morph_features": ["{}"] * len(words), "sentence_id": [1 + index // 4 for index in range(len(words))]})


@pytest.fixture
def corpus():
    return {
        "א_סיפור_1900.txt": frame(["אדם", "אדם", "אדם", "בית", "כתב", "הלך", "בית", "כתב"], ["NOUN", "NOUN", "NOUN", "NOUN", "VERB", "VERB", "NOUN", "VERB"]),
        "ב_סיפור_1910.txt": frame(["אדם", "בית", "בית", "בית", "בית", "כתב", "הלך", "הלך"], ["NOUN", "NOUN", "NOUN", "NOUN", "NOUN", "VERB", "VERB", "VERB"]),
        "ג_סיפור_1920.txt": frame(["אדם", "בית", "כתב", "כתב", "כתב", "כתב", "הלך", "הלך"], ["NOUN", "NOUN", "VERB", "VERB", "VERB", "VERB", "VERB", "VERB"]),
        "ד_סיפור_1930.txt": frame(["אדם", "בית", "כתב", "הלך", "הלך", "הלך", "הלך", "הלך"], ["NOUN", "NOUN", "VERB", "VERB", "VERB", "VERB", "VERB", "VERB"]),
    }


@pytest.fixture
def cached_project(tmp_path, corpus):
    project = Project.create(tmp_path / "project")
    sources = tmp_path / "sources"
    sources.mkdir()
    for filename, tokens in corpus.items():
        source = sources / filename
        source.write_text(" ".join(tokens.word), encoding="utf-8")
        record = project.import_texts([source])[0]
        save_cache(project, record, project.settings["nlp"], tokens)
    return project


@pytest.mark.parametrize("kind", ANALYSIS_KINDS)
def test_every_analysis_renders_and_exports(qtbot, corpus, tmp_path, kind):
    params = {"target": next(iter(corpus)), "window_size": 4, "step_size": 2, "ngram_n": 2}
    result = run_analysis(corpus, kind, params)
    panel = VisualizationPanel()
    qtbot.addWidget(panel)
    panel.plot(result)
    panel.canvas.draw()
    output = tmp_path / "chart.png"
    panel.export(output)
    assert output.stat().st_size > 1000
    if kind == "Rolling Delta":
        assert len(panel.figure.axes[0].lines) == len(result.payload["distance_columns"])


def test_main_window_cached_project_analysis_and_report(qtbot, cached_project, tmp_path):
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.open_project(cached_project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    assert len(window.frames) == 4
    assert window.metadata_model.data(window.metadata_model.index(0, 1)).startswith("Ready")
    assert window.tokens.model.rowCount() == 8
    window.dashboard.kind.setCurrentText("Burrows Delta")
    window.run_analysis()
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    assert window.result.kind == "Burrows Delta"
    assert window.result.table.shape == (4, 4)
    window.tabs.setCurrentIndex(2)
    window.dashboard.chart.canvas.draw()
    qtbot.wait(100)
    assert window.grab().save(str(tmp_path / "studio.png"))
    window.export_report()
    report = next(cached_project.outputs_dir.glob("report_*"))
    assert {"results.csv", "chart.png", "analysis_settings.json", "report.md"} <= {path.name for path in report.iterdir()}
    settings = json.loads((report / "analysis_settings.json").read_text())
    assert len(settings["texts"]) == 4
    window.close()


def test_metadata_edit_persists_and_hebrew_stays_logical(qtbot, cached_project):
    model = MetadataTableModel()
    model.set_project(cached_project)
    author_index = model.index(0, 2)
    assert model.setData(author_index, "מחבר עברי")
    assert Project.open(cached_project.root).texts[0].metadata["author"] == "מחבר עברי"
    tokens_model = PandasTableModel(frame(["שלום"]))
    assert tokens_model.data(tokens_model.index(0, 1)) == "שלום"


def test_refresh_preserves_unchecked_corpus(qtbot, cached_project):
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_project(cached_project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    window._check_all(Qt.CheckState.Unchecked)
    window._refresh_project()
    assert all(window.explorer.item(index).checkState() == Qt.CheckState.Unchecked for index in range(4))
    window.close()


def test_preprocessing_keeps_gui_responsive_and_refreshes_token_table_once(qtbot, cached_project, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_project(cached_project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    threads, ticks, displayed = [], [], []

    class BatchedPipeline:
        def __init__(self, settings):
            pass

        def load(self):
            threads.append(QThread.currentThread())

        def process(self, text, progress=None, is_cancelled=None):
            threads.append(QThread.currentThread())
            for value in range(0, 101, 10):
                time.sleep(0.005)
                progress(value)
            return frame(["שלום", "עולם"])

    original_set_frame = window.tokens.set_frame

    def display(table):
        if not table.empty:
            displayed.append(table)
        original_set_frame(table)

    monkeypatch.setattr("app.nlp.workers.DictaPipeline", BatchedPipeline)
    monkeypatch.setattr(window.tokens, "set_frame", display)
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(True))
    window.force.setChecked(True)
    window.process_corpus()
    timer.start()
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    timer.stop()
    assert len(ticks) >= 5
    assert all(thread is not window.thread() for thread in threads)
    assert len(displayed) == 1
    assert len(window.frames) == len(cached_project.texts)
    assert window.tokens.proxy.sortColumn() == -1
    assert window.tokens.model.frame.word.tolist() == ["שלום", "עולם"]
    assert window.progress.value() == 100
    assert window.process_button.isEnabled()
    window.close()


def test_pca_document_count_is_explained_before_starting_worker(qtbot, cached_project):
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_project(cached_project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    for row in range(2, 4):
        window.explorer.item(row).setCheckState(Qt.CheckState.Unchecked)
    config_path = cached_project.root / "project_config.json"
    saved_config = config_path.read_bytes()
    window.dashboard.kind.setCurrentText("PCA 2D")
    window.run_analysis()
    assert window._worker is None
    assert window.result is None
    assert "3 processed texts; 2 are selected" in window.dashboard.validation_message.text()
    assert "chapters as separate texts" in window.dashboard.validation_message.text()
    assert config_path.read_bytes() == saved_config
    # Lexical analyses remain available for the same small corpus.
    window.dashboard.kind.setCurrentText("MFW")
    window.run_analysis()
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    assert window.result.kind == "MFW"
    assert not window.dashboard.validation_message.text()
    window.close()


def test_pca_feature_limit_is_visible_and_recovers_controls(qtbot, cached_project, caplog):
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_project(cached_project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    window.dashboard.kind.setCurrentText("PCA 2D")
    window.dashboard.n_mfw.setValue(1)
    window.run_analysis()
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    assert "2 variable features" in window.dashboard.validation_message.text()
    assert "Analysis needs attention" in window.task_label.text()
    assert window.dashboard.run_button.isEnabled()
    assert window.result is None
    assert not any(record.exc_info for record in caplog.records)
    window.close()


def test_numeric_sort_and_hebrew_filter(qtbot):
    from app.gui.tables import DataTable

    table = DataTable()
    qtbot.addWidget(table)
    table.set_frame(pd.DataFrame({"word": ["שלום", "אדם", "בית"], "count": [2, 10, 1]}))
    table.view.sortByColumn(1, Qt.SortOrder.AscendingOrder)
    assert table.visible_frame()["count"].tolist() == [1, 2, 10]
    table.search.setText("שלום")
    assert table.visible_frame().word.tolist() == ["שלום"]


def test_chart_svg_pdf_and_excel_exports(qtbot, corpus, tmp_path):
    from app.utils.io import export_table

    result = run_analysis(corpus, "MFW", {"n_mfw": 4})
    panel = VisualizationPanel()
    qtbot.addWidget(panel)
    panel.plot(result)
    for extension in ("svg", "pdf"):
        path = tmp_path / f"chart.{extension}"
        panel.export(path)
        assert path.stat().st_size > 1000
    excel = tmp_path / "words.xlsx"
    export_table(result.table, excel)
    imported = pd.read_excel(excel, index_col=0)
    assert imported.index.tolist() == result.table.index.tolist()
    assert imported.token_count.tolist() == result.table.token_count.tolist()


def test_hebrew_scatter_labels_render_once_and_fit_canvas(qtbot, corpus, tmp_path):
    from packaging.version import Version
    import matplotlib

    result = run_analysis(corpus, "PCA 2D", {})
    panel = VisualizationPanel()
    qtbot.addWidget(panel)
    panel.resize(900, 550)
    panel.show()
    qtbot.wait(100)
    groups = dict(zip(corpus, ["כהן", "לוי", "רחל", "דוד"]))
    panel.plot(result, groups=groups)
    panel.canvas.draw()
    axes = panel.figure.axes[0]
    labels = panel._labels
    if Version(matplotlib.__version__).release >= (3, 11):
        assert {label.get_text() for label in labels} == {document_display_name(name) for name in corpus}
        assert {text.get_text() for text in axes.get_legend().get_texts()} == set(groups.values())
    renderer = panel.canvas.get_renderer()
    bounds = panel.figure.bbox
    for label in labels:
        box = label.get_window_extent(renderer)
        assert box.x0 >= bounds.x0 and box.x1 <= bounds.x1
        assert box.y0 >= bounds.y0 and box.y1 <= bounds.y1
    for extension in ("png", "svg", "pdf"):
        output = tmp_path / f"rtl_pca.{extension}"
        panel.export(output)
        assert output.stat().st_size > 1000
