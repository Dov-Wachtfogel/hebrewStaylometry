"""Portable folder-based projects with validated metadata and atomic saves."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from app.utils.io import atomic_write_json

SCHEMA_VERSION = 1
CONFIG_FILENAME = "project_config.json"
DEFAULT_SETTINGS: dict[str, Any] = {
    "nlp": {
        "backend": "dicta",
        "model_name": "dicta-il/dictabert-tiny-joint",
        "revision": "main",
        "device": "cpu",
        "strip_niqqud": True,
        "segmented": True,
        "batch_size": 32,
    },
    "analysis": {
        "feature": "lemma",
        "n_mfw": 200,
        "stop_words": [],
        "mattr_window": 50,
        "ngram_size": 2,
        "window_size": 500,
        "step_size": 250,
        "distance": "delta",
        "linkage": "average",
        "reduction": "tsne",
        "random_state": 42,
    },
}


@dataclass
class TextRecord:
    text_id: str
    filename: str
    metadata: dict[str, str] = field(default_factory=dict)


def _safe_filename(filename: object) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or ":" in filename
        or any(ord(character) < 32 for character in filename)
        or Path(filename).suffix.casefold() != ".txt"
    ):
        raise ValueError(f"Invalid text filename: {filename!r}. Use a .txt basename.")
    return filename


def _metadata(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        raise ValueError("Text metadata must be a JSON object.")
    result: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Metadata keys must be nonempty strings.")
        if isinstance(value, (dict, list)):
            raise ValueError(f"Metadata field {key!r} must contain a scalar value.")
        result[key.strip().casefold().replace(" ", "_")] = "" if value is None else str(value)
    return result


def parse_filename_metadata(filename: str) -> dict[str, str]:
    """Parse Author_Title_Date.txt; additional underscores belong to the title."""
    parts = Path(_safe_filename(filename)).stem.split("_")
    result = {"author": "", "title": parts[0], "date": "", "genre": "", "period": "", "chapter": ""}
    if len(parts) == 2:
        result.update(author=parts[0], title=parts[1])
    elif len(parts) >= 3:
        result.update(author=parts[0], title="_".join(parts[1:-1]), date=parts[-1])
    return result


def _validate_nlp_settings(settings: dict[str, Any]) -> None:
    """Reject malformed persisted options before the GUI reads them.

    Model downloads and construction belong to the processing worker. The
    project layer only validates JSON types, so opening a project remains
    independent of NLP dependencies and cannot trigger model loading.
    Unknown options are retained for future pipeline extensions.
    """
    if settings["backend"] != "dicta":
        raise ValueError("NLP backend must be 'dicta'.")
    batch_size = settings["batch_size"]
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("NLP batch_size must be a positive integer.")
    for name in ("model_name", "revision", "device"):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f"NLP {name} must be a nonempty string.")
    for name in ("strip_niqqud", "segmented", "local_files_only"):
        if name in settings and not isinstance(settings[name], bool):
            raise ValueError(f"NLP {name} must be a boolean.")
    if "code_revision" in settings and not isinstance(settings["code_revision"], str):
        raise ValueError("NLP code_revision must be a string.")
    max_length = settings.get("max_sequence_length")
    if max_length is not None and (type(max_length) is not int or max_length < 4):
        raise ValueError("NLP max_sequence_length must be an integer of at least 4.")


def _merge_settings(settings: object) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("Project settings must be a JSON object.")
    result = copy.deepcopy(DEFAULT_SETTINGS)
    for key, value in settings.items():
        if key in result and isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"The {key!r} settings must be a JSON object.")
            result[key].update(copy.deepcopy(value))
        else:
            result[key] = copy.deepcopy(value)
    _validate_nlp_settings(result["nlp"])
    return result


@dataclass
class Project:
    root: Path
    name: str
    texts: list[TextRecord] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_SETTINGS))

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw_texts"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed_data"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @classmethod
    def create(cls, root: Path, name: str | None = None) -> Project:
        root = Path(root).expanduser().resolve()
        config = root / CONFIG_FILENAME
        if config.exists() or config.is_symlink():
            raise FileExistsError(f"A project already exists at {root}.")
        project = cls(root=root, name=name if name is not None else root.name)
        project._validate()
        root.mkdir(parents=True, exist_ok=True)
        project._ensure_directories()
        project.save()
        return project

    @classmethod
    def open(cls, root: Path) -> Project:
        root = Path(root).expanduser().resolve()
        config = root / CONFIG_FILENAME
        if config.is_symlink():
            raise ValueError("Project configuration must not be a symbolic link.")
        with config.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("Unsupported or missing project schema_version.")
        if not isinstance(data.get("texts"), list):
            raise ValueError("Project texts must be a JSON list.")
        records: list[TextRecord] = []
        for item in data["texts"]:
            if not isinstance(item, dict) or not {"text_id", "filename"}.issubset(item):
                raise ValueError("Each text needs text_id and filename fields.")
            records.append(TextRecord(item["text_id"], _safe_filename(item["filename"]), _metadata(item.get("metadata", {}))))
        project = cls(root, data.get("name"), records, _merge_settings(data.get("settings", {})))
        project._validate()
        project._ensure_directories()
        return project

    def _ensure_directories(self) -> None:
        for directory in (self.raw_dir, self.processed_dir, self.outputs_dir):
            if directory.is_symlink():
                raise ValueError(f"Project directory must not be a symbolic link: {directory.name}")
            directory.mkdir(parents=True, exist_ok=True)

    def _validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Project name must be a nonempty string.")
        _merge_settings(self.settings)
        if (self.root / CONFIG_FILENAME).is_symlink():
            raise ValueError("Project configuration must not be a symbolic link.")
        for directory in (self.raw_dir, self.processed_dir, self.outputs_dir):
            if directory.is_symlink():
                raise ValueError(f"Project directory must not be a symbolic link: {directory.name}")
            if directory.exists() and not directory.is_dir():
                raise ValueError(f"Expected a project directory: {directory.name}")
        identifiers: set[str] = set()
        filenames: set[str] = set()
        for record in self.texts:
            if (
                not isinstance(record.text_id, str)
                or re.fullmatch(r"[A-Za-z0-9_-]+", record.text_id) is None
                or record.text_id in identifiers
            ):
                raise ValueError("Text identifiers must be safe, nonempty, and unique.")
            filename = _safe_filename(record.filename)
            if filename.casefold() in filenames:
                raise ValueError("Text filenames must be unique, ignoring case.")
            if (self.raw_dir / filename).is_symlink():
                raise ValueError(f"Raw text must not be a symbolic link: {filename}")
            _metadata(record.metadata)
            identifiers.add(record.text_id)
            filenames.add(filename.casefold())

    def save(self) -> None:
        self._validate()
        atomic_write_json(self.root / CONFIG_FILENAME, {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "texts": [{"text_id": item.text_id, "filename": item.filename, "metadata": item.metadata} for item in self.texts],
            "settings": self.settings,
        })

    def get_record(self, text_id: str) -> TextRecord:
        for record in self.texts:
            if record.text_id == text_id:
                return record
        raise KeyError(f"Unknown text identifier: {text_id}")

    def import_texts(self, paths: Iterable[Path]) -> list[TextRecord]:
        self._validate()
        self._ensure_directories()
        sources = [Path(path).expanduser() for path in paths]
        for source in sources:
            _safe_filename(source.name)
            if not source.is_file():
                raise FileNotFoundError(f"Text file does not exist: {source}")
        occupied = {path.name.casefold() for path in self.raw_dir.iterdir()}
        occupied.update(record.filename.casefold() for record in self.texts)
        imported: list[TextRecord] = []
        created: list[Path] = []
        original_count = len(self.texts)
        try:
            for source in sources:
                filename = source.name
                sequence = 2
                while filename.casefold() in occupied:
                    filename = f"{source.stem}_{sequence}{source.suffix}"
                    sequence += 1
                destination = self.raw_dir / filename
                with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
                    created.append(destination)
                    shutil.copyfileobj(source_handle, destination_handle)
                record = TextRecord(uuid.uuid4().hex, filename, parse_filename_metadata(source.name))
                self.texts.append(record)
                imported.append(record)
                occupied.add(filename.casefold())
            if imported:
                self.save()
        except Exception:
            del self.texts[original_count:]
            for destination in created:
                destination.unlink(missing_ok=True)
            raise
        return imported

    def import_text_content(
        self, documents: Iterable[tuple[str, str, dict[str, str]]]
    ) -> list[TextRecord]:
        """Import a complete batch of downloaded text and its metadata.

        Text is stored in its original logical Unicode order as UTF-8. Validate
        the entire batch before creating files; persist metadata only after all
        files have been written and flushed. Existing filenames are never
        overwritten, including names that differ only by case. A failed write
        or configuration save removes only files and records from this batch.

        Services can supply fields such as ``title``, ``author``, and ``ref``;
        these override filename-derived defaults. Call this from a background
        worker for large downloads, using the worker's own Project instance.
        """
        self._validate()
        prepared: list[tuple[str, bytes, dict[str, str]]] = []
        for document in documents:
            try:
                filename, text, values = document
            except (TypeError, ValueError) as error:
                raise ValueError("Each text document needs filename, text, and metadata.") from error
            filename = _safe_filename(filename)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Text content for {filename!r} must be a nonempty string.")
            metadata = parse_filename_metadata(filename)
            metadata.update(_metadata(values))
            # Encoding belongs to validation, so an invalid Unicode string in
            # a later document cannot leave any earlier document on disk.
            prepared.append((filename, text.encode("utf-8"), metadata))
        if not prepared:
            return []

        self._ensure_directories()
        occupied = {path.name.casefold() for path in self.raw_dir.iterdir()}
        occupied.update(record.filename.casefold() for record in self.texts)
        imported: list[TextRecord] = []
        created: list[Path] = []
        original_count = len(self.texts)
        try:
            for requested_name, content, metadata in prepared:
                source_name = Path(requested_name)
                filename = requested_name
                sequence = 2
                while filename.casefold() in occupied:
                    filename = f"{source_name.stem}_{sequence}{source_name.suffix}"
                    sequence += 1
                destination = self.raw_dir / filename
                # Exclusive creation also protects against a concurrent writer
                # claiming this destination after the directory was scanned.
                with destination.open("xb") as handle:
                    created.append(destination)
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                imported.append(TextRecord(uuid.uuid4().hex, filename, metadata))
                occupied.add(filename.casefold())
            self.texts.extend(imported)
            self.save()
        except Exception:
            del self.texts[original_count:]
            for destination in created:
                destination.unlink(missing_ok=True)
            raise
        return imported

    def update_metadata(self, text_id: str, key: str, value: str) -> None:
        record = self.get_record(text_id)
        fields = _metadata({key: value})
        previous = record.metadata.copy()
        record.metadata.update(fields)
        try:
            self.save()
        except Exception:
            record.metadata = previous
            raise

    def remove_text(self, text_id: str) -> TextRecord:
        """Remove a text record, its project copy, and its token cache."""
        record = self.get_record(text_id)
        raw_path = self.raw_dir / record.filename
        if raw_path.is_symlink():
            raise ValueError(f"Raw text must not be a symbolic link: {record.filename}")

        index = self.texts.index(record)
        self.texts.pop(index)
        try:
            self.save()
        except Exception:
            self.texts.insert(index, record)
            raise

        raw_path.unlink(missing_ok=True)
        stem = Path(record.filename).stem
        cache_path = self.processed_dir / f"{stem}_tokens.csv"
        cache_path.unlink(missing_ok=True)
        cache_path.with_suffix(".meta.json").unlink(missing_ok=True)
        return record

    def import_metadata(self, path: Path) -> int:
        """Merge filename-keyed JSON or a list of records containing filename.

        Unknown filenames are ignored. Scalar custom fields such as chapter,
        genre, and period are supported alongside author, title, and date.
        The returned count is the number of matched texts.
        """
        with Path(path).open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            rows = list(data.items())
        elif isinstance(data, list):
            rows = []
            for item in data:
                if not isinstance(item, dict) or "filename" not in item:
                    raise ValueError("Metadata list entries must contain filename.")
                values = {key: value for key, value in item.items() if key not in {"filename", "metadata"}}
                if "metadata" in item:
                    values.update(_metadata(item["metadata"]))
                rows.append((item["filename"], values))
        else:
            raise ValueError("Metadata JSON must be a filename mapping or a list.")
        updates: dict[str, dict[str, str]] = {}
        records_by_filename = {record.filename.casefold(): record for record in self.texts}
        for filename, values in rows:
            safe_filename = _safe_filename(filename)
            normalized = _metadata(values)
            record = records_by_filename.get(safe_filename.casefold())
            if record is not None:
                updates.setdefault(record.text_id, {}).update(normalized)
        snapshots = {text_id: self.get_record(text_id).metadata.copy() for text_id in updates}
        try:
            for text_id, values in updates.items():
                self.get_record(text_id).metadata.update(values)
            if updates:
                self.save()
        except Exception:
            for text_id, values in snapshots.items():
                self.get_record(text_id).metadata = values
            raise
        return len(updates)
