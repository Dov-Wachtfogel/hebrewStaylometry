"""Logical Unicode stays untouched in Qt: bidi conversion is chart-only."""

import numbers
import re
from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

from app.project import Project

HEBREW = re.compile(r"[\u0590-\u05ff]")


class PandasTableModel(QAbstractTableModel):
    """Read-only DataFrame with numeric sorting and native Hebrew rendering."""

    def __init__(self, frame: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self.frame = frame if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame) -> None:
        self.beginResetModel()
        self.frame = frame
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.frame.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        value = self.frame.iat[index.row(), index.column()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole):
            if pd.isna(value):
                return ""
            return f"{value:.6g}" if isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral) else str(value)
        if role == Qt.ItemDataRole.UserRole:
            if pd.isna(value):
                return ""
            return value.item() if hasattr(value, "item") else value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            side = Qt.AlignmentFlag.AlignRight if HEBREW.search(str(value)) or isinstance(value, numbers.Number) else Qt.AlignmentFlag.AlignLeft
            return int(side | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            values = self.frame.columns if orientation == Qt.Orientation.Horizontal else self.frame.index
            return str(values[section])
        return None


class MetadataTableModel(QAbstractTableModel):
    """Editable metadata; persistence errors propagate as signals, never Qt exceptions."""

    metadata_changed = Signal()
    error = Signal(str)
    BASE_FIELDS = ["author", "title", "genre", "period", "date", "chapter"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project: Project | None = None
        self.statuses: dict[str, str] = {}
        self.fields = list(self.BASE_FIELDS)

    def set_project(self, project: Project | None, statuses=None):
        self.beginResetModel()
        self.project = project
        self.statuses = dict(statuses or {})
        extra = {key for record in project.texts for key in record.metadata} if project else set()
        self.fields = self.BASE_FIELDS + sorted(extra.difference(self.BASE_FIELDS))
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() or self.project is None else len(self.project.texts)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2 + len(self.fields)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.project is None:
            return None
        record = self.project.texts[index.row()]
        if index.column() == 0:
            value = record.filename
        elif index.column() == 1:
            value = self.statuses.get(record.text_id, "Not processed")
        else:
            value = record.metadata.get(self.fields[index.column() - 2], "")
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole, Qt.ItemDataRole.ToolTipRole):
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int((Qt.AlignmentFlag.AlignRight if HEBREW.search(value) else Qt.AlignmentFlag.AlignLeft) | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return (["File", "Status"] + [field.title() for field in self.fields])[section] if orientation == Qt.Orientation.Horizontal else str(section + 1)
        return None

    def flags(self, index):
        base = super().flags(index)
        return base | Qt.ItemFlag.ItemIsEditable if index.isValid() and index.column() >= 2 else base

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if self.project is None or not index.isValid() or index.column() < 2 or role != Qt.ItemDataRole.EditRole:
            return False
        try:
            self.project.update_metadata(self.project.texts[index.row()].text_id, self.fields[index.column() - 2], str(value))
        except (ValueError, OSError) as exc:
            self.error.emit(str(exc))
            return False
        self.dataChanged.emit(index, index, [role, Qt.ItemDataRole.DisplayRole])
        self.metadata_changed.emit()
        return True

    def set_status(self, text_id: str, status: str):
        self.statuses[text_id] = status
        if self.project:
            for row, record in enumerate(self.project.texts):
                if record.text_id == text_id:
                    index = self.index(row, 1)
                    self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole])
                    break
