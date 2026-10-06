"""Reusable sortable, filterable pandas table view."""

import pandas as pd
from PySide6.QtCore import QRegularExpression, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QLineEdit, QTableView, QVBoxLayout, QWidget

from app.gui.models import PandasTableModel


class DataTable(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter all columns…")
        self.model = PandasTableModel(parent=self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setFilterKeyColumn(-1)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)
        self.view = QTableView()
        self.view.setModel(self.proxy)
        self.view.setSortingEnabled(True)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.view.horizontalHeader().setDefaultSectionSize(140)
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)
        layout.addWidget(self.view)

    def _filter(self, text):
        expression = QRegularExpression(QRegularExpression.escape(text))
        expression.setPatternOptions(QRegularExpression.PatternOption.CaseInsensitiveOption)
        self.proxy.setFilterRegularExpression(expression)

    def set_frame(self, frame):
        self.model.set_frame(frame)

    def visible_frame(self) -> pd.DataFrame:
        """Export the filtered and sorted rows visible to the user."""
        positions = [self.proxy.mapToSource(self.proxy.index(row, 0)).row() for row in range(self.proxy.rowCount())]
        return self.model.frame.iloc[positions]
