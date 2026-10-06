"""Scatter label geometry, document lookup, and metadata coloring regressions."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.backend_bases import MouseEvent
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolTip

from app.gui.main_window import MainWindow
from app.gui.visualization import VisualizationPanel
from app.nlp.cache import save_cache
from app.nlp.pipeline import TOKEN_COLUMNS
from app.project import Project


def scatter_result(count, *, clustered=False, three_d=False):
    labels = [f"II Samuel {i + 1} - Tanach with Text Only_{i:010x}.txt" for i in range(count)]
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    x, y = np.cos(angles), np.sin(angles)
    if clustered:
        x[:-2], y[:-2] = np.arange(count - 2) * 1e-5, np.arange(count - 2) * 1e-5
        x[-2:], y[-2:] = [-1, 1], [-1, 1]
    table = pd.DataFrame({"PC1": x, "PC2": y}, index=labels)
    if three_d:
        table["PC3"] = np.linspace(-0.5, 0.5, count)
    return SimpleNamespace(kind="PCA 3D" if three_d else "PCA 2D", table=table,
                           payload={"explained_variance": [0.6, 0.3, 0.1]})


def visible_panel(qtbot):
    panel = VisualizationPanel()
    qtbot.addWidget(panel)
    panel.resize(1100, 650)
    panel.show()
    qtbot.wait(30)
    return panel


def test_clean_titles_full_hover_details_and_visibility_toggle(qtbot, monkeypatch, tmp_path):
    panel = visible_panel(qtbot)
    result = scatter_result(4)
    filenames = result.table.index.tolist()
    metadata = {name: {"book": "II Samuel", "author": "מחבר", "period": "biblical"} for name in filenames}
    panel.plot(result, groups=dict.fromkeys(filenames, "II Samuel"), metadata=metadata, group_title="book")
    qtbot.wait(200)
    panel.canvas.draw()
    assert {text.get_text() for text in panel._labels} == {f"II Samuel {i}" for i in range(1, 5)}
    assert result.table.index.tolist() == filenames
    assert panel.documents.count() == 4
    tooltip = panel.documents.item(0).toolTip()
    assert filenames[0] in tooltip and "מחבר" in tooltip and "biblical" in tooltip
    calls = []
    monkeypatch.setattr(QToolTip, "showText", lambda position, text, *args: calls.append(text))
    x, y = panel._point_pixels()[0]
    panel._hover(MouseEvent("motion_notify_event", panel.canvas, x, y))
    assert panel._hovered_document == filenames[0] and filenames[0] in calls[-1]
    panel.show_labels.setChecked(False)
    assert not any(artist.get_visible() for artist in panel._labels + panel._label_arrows)
    assert panel.documents.count() == 4
    panel.show_labels.setChecked(True)
    assert all(text.get_visible() for text in panel._labels)
    for suffix in ["png", "svg", "pdf"]:
        path = tmp_path / f"clean.{suffix}"
        panel.export(path)
        assert path.stat().st_size > 1000


def test_clustered_labels_do_not_overlap_and_remain_inside_axes(qtbot):
    panel = visible_panel(qtbot)
    panel.plot(scatter_result(8, clustered=True))
    qtbot.wait(200)
    panel.canvas.draw()
    renderer = panel.canvas.get_renderer()
    boxes = [label.get_window_extent(renderer) for label in panel._labels]
    assert not any(a.overlaps(b) for index, a in enumerate(boxes) for b in boxes[index + 1:])
    bounds = panel._axes.get_window_extent(renderer)
    for box in boxes:
        assert bounds.x0 - 1 <= box.x0 <= box.x1 <= bounds.x1 + 1
        assert bounds.y0 - 1 <= box.y0 <= box.y1 <= bounds.y1 + 1
    assert panel._label_arrows


def test_larger_scatter_uses_numbered_key_and_selection_highlights_original_point(qtbot):
    panel = visible_panel(qtbot)
    result = scatter_result(20)
    panel.plot(result)
    assert panel._annotation_mode == "Numbers"
    assert [text.get_text() for text in panel._labels] == list(map(str, range(1, 21)))
    item = panel.documents.item(8)
    assert item.text().startswith("9. II Samuel 9")
    assert item.data(Qt.ItemDataRole.UserRole) == result.table.index[8]
    panel.documents.setCurrentItem(item)
    np.testing.assert_allclose(panel._selected_marker.get_offsets()[0], result.table.iloc[8].to_numpy())
    panel.label_mode.setCurrentText("Hover only")
    assert not panel._labels and panel.documents.count() == 20
    panel.clear()
    assert panel._coordinates is None and not panel._labels and panel.documents.count() == 0


@pytest.mark.parametrize("three_d,count", [(True, 4), (False, 81)])
def test_3d_and_dense_plots_keep_document_details_without_canvas_clutter(qtbot, three_d, count):
    panel = visible_panel(qtbot)
    result = scatter_result(count, three_d=three_d)
    panel.plot(result)
    panel.canvas.draw()
    assert not panel._labels
    assert panel.documents.count() == count
    assert panel._point_pixels().shape == (count, 2)
    assert result.table.index[0] in panel.documents.item(0).toolTip()


def test_missing_authors_are_colored_by_book_and_recoloring_needs_no_analysis(qtbot, tmp_path):
    project = Project.create(tmp_path / "project")
    records = project.import_text_content([
        (f"doc{i}.txt", "שלום עולם", {"book": "II Samuel" if i < 2 else "I Kings", "author": ""})
        for i in range(4)
    ])
    for i, record in enumerate(records):
        counts = [1] * 4
        counts[i] = 4
        words = [word for word, count in zip(["שלום", "בית", "אדם", "הלך"], counts) for _ in range(count)]
        tokens = pd.DataFrame([[index, word, word, "NOUN", "{}", 1] for index, word in enumerate(words, 1)], columns=TOKEN_COLUMNS)
        save_cache(project, record, project.settings["nlp"], tokens)
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_project(project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    window.dashboard.kind.setCurrentText("PCA 2D")
    window.run_analysis()
    qtbot.waitUntil(lambda: window._worker is None, timeout=10000)
    assert window.result is not None, window.task_label.text()
    assert window.result_params["resolved_color_by"] == "book"
    legend = window.dashboard.chart._axes.get_legend()
    assert {text.get_text() for text in legend.get_texts()} == {"I Kings", "II Samuel"}
    coordinates = window.result.table.copy()
    window.dashboard.color_by.setCurrentText("none")
    assert window._worker is None
    assert window.result_params["resolved_color_by"] == "none"
    assert len(window.dashboard.chart._axes.collections) == 1
    pd.testing.assert_frame_equal(window.result.table, coordinates)
    window.close()
