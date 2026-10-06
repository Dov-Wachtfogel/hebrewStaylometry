"""Embedded charts with renderer-aware Hebrew labels."""

from collections import Counter
from html import escape
from pathlib import Path

import numpy as np
from adjustText import adjust_text
from matplotlib import colormaps
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import proj3d
from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QSplitter, QToolTip, QVBoxLayout, QWidget
from scipy.cluster.hierarchy import dendrogram

from app.utils.hebrew import display_hebrew
from app.utils.labels import document_display_name


class VisualizationPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 5), layout="constrained", facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.scatter_controls = QWidget()
        controls = QHBoxLayout(self.scatter_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        self.show_labels = QCheckBox("Show labels")
        self.show_labels.setChecked(True)
        self.label_mode = QComboBox()
        self.label_mode.addItems(["Auto", "Titles", "Numbers", "Hover only"])
        self.label_mode.setToolTip("Auto uses short titles for up to 15 points, then numbers and the document list.")
        self.label_hint = QLabel("Hover or select a document for full details.")
        self.label_hint.setWordWrap(True)
        controls.addWidget(self.show_labels)
        controls.addWidget(self.label_mode)
        controls.addWidget(self.label_hint, 1)
        body = QSplitter(Qt.Orientation.Horizontal)
        self.documents = QListWidget()
        self.documents.setMinimumWidth(180)
        self.documents.setMaximumWidth(300)
        body.addWidget(self.canvas)
        body.addWidget(self.documents)
        body.setStretchFactor(0, 1)
        body.setSizes([800, 230])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.scatter_controls)
        layout.addWidget(body, 1)
        self._result = None
        self._metadata = {}
        self._groups = None
        self._group_title = ""
        self._labels = []
        self._label_arrows = []
        self._coordinates = None
        self._axes = None
        self._selected_marker = None
        self._hovered_document = None
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.setInterval(150)
        self._layout_timer.timeout.connect(self._layout_labels)
        self.show_labels.toggled.connect(self._toggle_labels)
        self.label_mode.currentTextChanged.connect(self._change_label_mode)
        self.documents.currentItemChanged.connect(self._select_document)
        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self.canvas.mpl_connect("figure_leave_event", lambda event: self._hide_tooltip())
        self.canvas.mpl_connect("resize_event", lambda event: self._layout_timer.start() if self._labels else None)
        self.clear()

    def clear(self):
        self._reset_scatter()
        self._result = None
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.text(0.5, 0.5, "Choose an analysis and run it to explore your corpus", ha="center", va="center", color="#64748b", transform=axes.transAxes)
        axes.set_axis_off()
        self.canvas.draw_idle()

    def _reset_scatter(self):
        self._layout_timer.stop()
        self._hide_tooltip()
        self._labels = []
        self._label_arrows = []
        self._coordinates = None
        self._selected_marker = None
        self.documents.clear()
        self.documents.hide()
        self.scatter_controls.hide()

    def plot(self, result, groups=None, metadata=None, group_title=""):
        self._reset_scatter()
        self._result, self._groups = result, groups
        self._metadata = dict(metadata or {})
        self._group_title = group_title
        self.figure.clear()
        table, kind, payload = result.table, result.kind, result.payload
        axes = self.figure.add_subplot(111, projection="3d" if kind == "PCA 3D" else None)
        self._axes = axes
        axes.set_title(kind, fontfamily="DejaVu Sans")
        if kind in {"PCA 2D", "PCA 3D", "t-SNE"}:
            coordinates = table.select_dtypes(include="number")
            self._coordinates = coordinates.iloc[:, :3 if kind == "PCA 3D" else 2]
            group_labels = {label: (groups or {}).get(label, "Corpus") for label in table.index}
            categories = sorted(set(group_labels.values()))
            palette = colormaps["tab10" if len(categories) <= 10 else "tab20" if len(categories) <= 20 else "turbo"]
            colors = {category: palette(index if len(categories) <= 20 else index / max(1, len(categories) - 1))
                      for index, category in enumerate(categories)}
            for category in categories:
                subset = coordinates.loc[[label for label in table.index if group_labels[label] == category]]
                axes.scatter(*[subset.iloc[:, dimension] for dimension in range(3 if kind == "PCA 3D" else 2)], s=55, alpha=0.85, color=colors[category], label=display_hebrew(category))
            axes.margins(0.18)
            self._display_names = {name: document_display_name(name, self._metadata.get(name)) for name in table.index}
            duplicates = Counter(self._display_names.values())
            for number, label in enumerate(table.index, 1):
                name = self._display_names[label]
                if duplicates[name] > 1:
                    self._display_names[label] = f"{name} · {number}"
                item = QListWidgetItem(f"{number}. {self._display_names[label]} — {group_labels[label]}")
                item.setData(Qt.ItemDataRole.UserRole, label)
                item.setToolTip(self._tooltip(label))
                swatch = QPixmap(12, 12)
                swatch.fill(QColor.fromRgbF(*colors[group_labels[label]][:3]))
                item.setIcon(QIcon(swatch))
                self.documents.addItem(item)
            self.documents.show()
            self.scatter_controls.show()
            can_label = kind != "PCA 3D" and len(coordinates) <= 80
            self.show_labels.setEnabled(can_label)
            self.label_mode.setEnabled(can_label)
            mode = self.label_mode.currentText()
            if mode == "Auto":
                mode = "Titles" if len(coordinates) <= 15 else "Numbers"
            if not can_label:
                mode = "Hover only"
            self._annotation_mode = mode
            if mode != "Hover only":
                for number, (label, row) in enumerate(coordinates.iterrows(), 1):
                    caption = self._display_names[label] if mode == "Titles" else str(number)
                    self._labels.append(axes.annotate(
                        display_hebrew(caption), (row.iloc[0], row.iloc[1]),
                        xytext=(row.iloc[0], row.iloc[1]), textcoords="data",
                        fontsize=8, ha="center", va="center", fontfamily="DejaVu Sans",
                        visible=self.show_labels.isChecked(),
                    ))
            variance = payload.get("explained_variance", [])
            for dimension, setter in enumerate([axes.set_xlabel, axes.set_ylabel] + ([axes.set_zlabel] if kind == "PCA 3D" else [])):
                caption = str(coordinates.columns[dimension])
                if dimension < len(variance):
                    caption += f" ({float(variance[dimension]):.1%})"
                setter(caption)
            if len(categories) <= 12:
                axes.legend(fontsize=8, title=display_hebrew(group_title.title()) if group_title else None)
        elif kind == "Burrows Delta":
            image = axes.imshow(table.to_numpy(dtype=float), cmap="magma_r", interpolation="nearest", aspect="auto")
            self.figure.colorbar(image, ax=axes, label="Delta distance")
            labels = [self._document_label(label) for label in table.index]
            positions = np.arange(len(labels))
            axes.set_xticks(positions, labels, rotation=60, ha="right", fontsize=8)
            axes.set_yticks(positions, labels, fontsize=8)
        elif kind == "Hierarchical clustering":
            dendrogram(payload["linkage"], labels=[self._document_label(label) for label in payload["labels"]], ax=axes, leaf_rotation=60, leaf_font_size=8)
            axes.set_ylabel("Linkage distance")
        elif kind == "POS distribution":
            positions = np.arange(len(table))
            bottoms = np.zeros(len(table))
            for category in table.columns:
                values = table[category].to_numpy(dtype=float)
                axes.bar(positions, values, bottom=bottoms, label=str(category))
                bottoms += values
            axes.set_xticks(positions, [self._document_label(label) for label in table.index], rotation=55, ha="right", fontsize=8)
            axes.set_ylabel("Relative frequency")
            axes.legend(fontsize=8, ncol=3)
        elif kind == "Sentence / word length":
            self.figure.clear()
            axes_word = self.figure.add_subplot(121)
            axes_sentence = self.figure.add_subplot(122)
            for key, axes_hist, label in [("word_lengths", axes_word, "Characters per token"), ("sentence_lengths", axes_sentence, "Tokens per sentence")]:
                values = payload.get(key, {})
                if isinstance(values, dict):
                    for document, lengths in values.items():
                        if len(lengths):
                            axes_hist.hist(lengths, bins="auto", histtype="step", label=self._document_label(document))
                else:
                    axes_hist.hist(values, bins="auto", histtype="step")
                axes_hist.set_xlabel(label)
                axes_hist.set_ylabel("Count")
                axes_hist.legend(fontsize=7)
            axes_word.set_title("Word lengths")
            axes_sentence.set_title("Sentence lengths")
        elif kind == "Rolling Delta":
            numeric = table.select_dtypes(include="number")
            x = table["center"] if "center" in table else (table["start"] if "start" in table else np.arange(len(table)))
            for column in payload["distance_columns"]:
                axes.plot(x, numeric[column], label=self._document_label(column))
            axes.set_xlabel("Token position in target document")
            axes.set_ylabel("Delta to reference")
            axes.legend(fontsize=8)
        elif kind in {"MFW", "Word n-grams", "POS n-grams"}:
            subset = table.head(30)
            count_column = next((name for name in ["token_count", "count", "frequency"] if name in subset), subset.select_dtypes(include="number").columns[0])
            positions = np.arange(len(subset))
            axes.barh(positions, subset[count_column].to_numpy(dtype=float), color="#2563eb")
            axes.set_yticks(positions, [display_hebrew(label) for label in subset.index], fontsize=8)
            axes.invert_yaxis()
            axes.set_xlabel(str(count_column).replace("_", " ").title())
            axes.set_title(f"{kind} — top {len(subset)}")
        else:
            numeric = table.select_dtypes(include="number")
            columns = [column for column in numeric if str(column).casefold() in {"ttr", "mattr"}]
            if not columns:
                columns = list(numeric.columns[:2])
            positions = np.arange(len(table))
            width = 0.8 / max(1, len(columns))
            for offset, column in enumerate(columns):
                axes.bar(positions + offset * width, numeric[column], width=width, label=str(column))
            axes.set_xticks(positions, [self._document_label(label) for label in table.index], rotation=55, ha="right", fontsize=8)
            axes.legend(fontsize=8)
        # DejaVu Sans includes Hebrew glyphs. Qt always keeps logical strings;
        # display_hebrew also leaves them logical for Matplotlib 3.11+.
        for text in self.figure.findobj(match=lambda artist: hasattr(artist, "set_fontfamily")):
            text.set_fontfamily("DejaVu Sans")
        if self._labels and self.show_labels.isChecked():
            self._layout_labels()
        else:
            self.canvas.draw_idle()

    def _document_label(self, filename):
        return display_hebrew(document_display_name(filename, self._metadata.get(filename)))

    def _change_label_mode(self, mode):
        if self._coordinates is not None:
            self.plot(self._result, self._groups, self._metadata, self._group_title)

    def _toggle_labels(self, visible):
        for artist in self._labels + self._label_arrows:
            artist.set_visible(visible)
        if visible:
            self._layout_labels()
        else:
            self.canvas.draw_idle()

    def _layout_labels(self):
        """Fit text after axes layout, with bounded work on the GUI thread."""
        if not self._labels or not self.show_labels.isChecked():
            return
        for arrow in self._label_arrows:
            arrow.remove()
        self._label_arrows = []
        self.canvas.draw()
        positions = self._coordinates.iloc[:, :2].to_numpy(dtype=float)

        def adjust():
            # Distinct initial offsets avoid random jitter for coincident points.
            pixels = self._axes.transData.transform(positions)
            angles = np.arange(len(positions)) * 2.399963229728653
            offsets = np.column_stack((np.cos(angles), np.sin(angles))) * 12
            starts = self._axes.transData.inverted().transform(pixels + offsets)
            for text, start in zip(self._labels, starts):
                text.set_position(start)
            legend = self._axes.get_legend()
            _, arrows = adjust_text(
                self._labels, x=positions[:, 0], y=positions[:, 1],
                target_x=positions[:, 0], target_y=positions[:, 1],
                objects=[legend] if legend is not None else None,
                ax=self._axes, ensure_inside_axes=True, time_lim=0.35,
                force_text=(0.3, 0.5), expand=(1.1, 1.2), min_arrow_len=4,
                arrowprops={"arrowstyle": "-", "color": "#94a3b8", "lw": 0.65, "alpha": 0.75},
            )
            return arrows

        self._label_arrows = adjust()
        self.canvas.draw()
        boxes = [text.get_window_extent(self.canvas.get_renderer()) for text in self._labels]
        overlap = any(first.overlaps(second) for index, first in enumerate(boxes) for second in boxes[index + 1:])
        if overlap and self.label_mode.currentText() == "Auto" and self._annotation_mode == "Titles":
            # A tight cluster can be unreadable even with short titles. Its
            # numbered key keeps every document available without wide labels.
            self._annotation_mode = "Numbers"
            for number, text in enumerate(self._labels, 1):
                text.set_text(str(number))
            for arrow in self._label_arrows:
                arrow.remove()
            self._label_arrows = adjust()
        self.canvas.draw_idle()

    def _tooltip(self, filename):
        title = document_display_name(filename, self._metadata.get(filename), max_length=None)
        lines = [f"<b>{escape(title)}</b>", f"File: {escape(str(filename))}"]
        lines.extend(f"{escape(str(key).replace('_', ' ').title())}: {escape(str(value))}"
                     for key, value in self._metadata.get(filename, {}).items() if value)
        return "<br>".join(lines)

    def _point_pixels(self):
        values = self._coordinates.to_numpy(dtype=float)
        if values.shape[1] == 3:
            x, y, _ = proj3d.proj_transform(*values.T, self._axes.get_proj())
            values = np.column_stack((x, y))
        return self._axes.transData.transform(values)

    def _hover(self, event):
        if self._coordinates is None or event.inaxes is not self._axes or self.toolbar.mode or event.button:
            self._hide_tooltip()
            return
        distances = np.linalg.norm(self._point_pixels() - [event.x, event.y], axis=1)
        index = int(np.argmin(distances))
        if distances[index] > 10:
            self._hide_tooltip()
            return
        filename = self._coordinates.index[index]
        if filename != self._hovered_document or not QToolTip.isVisible():
            self._hovered_document = filename
            ratio = self.canvas.device_pixel_ratio
            position = QPoint(int(event.x / ratio), int(self.canvas.height() - event.y / ratio))
            QToolTip.showText(self.canvas.mapToGlobal(position), self._tooltip(filename), self.canvas)

    def _hide_tooltip(self):
        self._hovered_document = None
        QToolTip.hideText()

    def _select_document(self, item, previous):
        if self._coordinates is None:
            return
        if self._selected_marker is not None:
            self._selected_marker.remove()
            self._selected_marker = None
        if item is not None:
            position = self._coordinates.loc[item.data(Qt.ItemDataRole.UserRole)]
            self._selected_marker = self._axes.scatter(
                *[[value] for value in position], s=160, facecolors="none",
                edgecolors="#0f172a", linewidths=1.6, zorder=8,
            )
        self.canvas.draw_idle()

    def export(self, path: Path):
        if path.suffix.casefold() not in {".png", ".svg", ".pdf"}:
            raise ValueError("Charts must be exported as PNG, SVG, or PDF.")
        self.figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
