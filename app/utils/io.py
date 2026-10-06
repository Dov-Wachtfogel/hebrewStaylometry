"""Atomic persistence and interoperable Hebrew text/table exports."""

from __future__ import annotations

import codecs
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def atomic_write_json(path: Path, data: Any) -> None:
    """Write UTF-8 JSON by replacing a complete, flushed temporary file."""
    path = Path(path)
    # Serialize first so an unsupported value never leaves a partial file.
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """Store a token table atomically, without a synthetic index column."""
    _write_csv(Path(path), frame, index=False)


def _write_csv(path: Path, frame: pd.DataFrame, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            frame.to_csv(handle, index=index)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_hebrew_text(path: Path) -> str:
    """Read UTF-8 (with an optional BOM) or legacy Windows Hebrew CP1255.

    Decoding is strict: invalid bytes are never silently dropped/replaced.
    A UTF-8 BOM is authoritative, so malformed BOM-marked UTF-8 raises.
    """
    return decode_hebrew_text(Path(path).read_bytes())


def decode_hebrew_text(data: bytes) -> str:
    """Decode source bytes already read for hashing, without another file read."""
    if data.startswith(codecs.BOM_UTF8):
        text = data.decode("utf-8-sig", errors="strict")
    else:
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            text = data.decode("cp1255", errors="strict")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def export_table(
    frame: pd.DataFrame, path: Path, *, index: bool = True, sheet_name: str = "Data"
) -> None:
    """Export CSV or XLSX; retain analysis row labels unless explicitly disabled."""
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        _write_csv(path, frame, index=index)
        return
    if suffix != ".xlsx":
        raise ValueError("Table exports require a .csv or .xlsx filename.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.stem}.", suffix=".xlsx", delete=False
        ) as handle:
            temporary = Path(handle.name)
        frame.to_excel(temporary, index=index, sheet_name=sheet_name, engine="openpyxl")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
