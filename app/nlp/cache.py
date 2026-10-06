"""CSV token caches verified against source, settings, schema, and CSV digest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from app.nlp.pipeline import DEFAULT_NLP_SETTINGS, TOKEN_COLUMNS
from app.utils.io import atomic_write_csv, atomic_write_json

if TYPE_CHECKING:
    from app.project import Project, TextRecord

CACHE_SCHEMA_VERSION = 2


def cache_paths(project: Project, record: TextRecord) -> tuple[Path, Path]:
    stem = Path(record.filename).stem
    csv_path = project.processed_dir / f"{stem}_tokens.csv"
    return csv_path, csv_path.with_suffix(".meta.json")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_fingerprint(
    project: Project, record: TextRecord, settings: dict[str, Any],
    *, source_sha256: str | None = None,
) -> str:
    """Fingerprint exact source bytes; callers may reuse a digest from their read."""
    normalized_settings = {**DEFAULT_NLP_SETTINGS, **settings}
    payload = {"source_sha256": source_sha256 if source_sha256 is not None else _file_digest(project.raw_dir / record.filename),
               "settings": normalized_settings, "schema_version": CACHE_SCHEMA_VERSION}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_token_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Require sequential token IDs and valid sentence IDs; preserve NA strings."""
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise ValueError("Token data must be a DataFrame with unique columns.")
    missing = set(TOKEN_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Token data is missing columns: {', '.join(sorted(missing))}.")
    result = frame.loc[:, TOKEN_COLUMNS].copy()
    for name in ("token_id", "sentence_id"):
        numeric = pd.to_numeric(result[name], errors="raise")
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values % 1) != 0).any() or (values < 1).any():
            raise ValueError(f"{name} must contain positive integer IDs.")
        if (values >= np.iinfo(np.int64).max).any():
            raise ValueError(f"{name} exceeds the supported integer range.")
        result[name] = numeric.astype("int64")
    if not np.array_equal(result["token_id"].to_numpy(), np.arange(1, len(result) + 1)):
        raise ValueError("token_id must be a consecutive sequence beginning at 1.")
    if (result["sentence_id"].diff().dropna() < 0).any():
        raise ValueError("sentence_id must follow document order.")
    for name in ("word", "lemma", "pos_tag", "morph_features"):
        result[name] = result[name].fillna("").astype(str)
    if result["word"].str.strip().eq("").any() or result["pos_tag"].str.strip().eq("").any():
        raise ValueError("Tokens must have nonempty words and POS tags.")
    return result


def load_cache(project: Project, record: TextRecord, settings: dict[str, Any]) -> pd.DataFrame | None:
    """Load only matching, uncorrupted caches; legacy files require processing.

    A CSV without a provenance sidecar cannot establish which source/settings
    generated it. It remains on disk, but is deliberately not trusted as NLP
    output. Malformed, stale, and partially written caches return None.
    """
    csv_path, metadata_path = cache_paths(project, record)
    if not csv_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        if metadata.get("fingerprint") != cache_fingerprint(project, record, settings):
            return None
        if metadata.get("csv_sha256") != _file_digest(csv_path):
            return None
        frame = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        return validate_token_frame(frame)
    except (ValueError, TypeError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, OverflowError):
        return None


def save_cache(
    project: Project, record: TextRecord, settings: dict[str, Any], frame: pd.DataFrame,
    *, expected_fingerprint: str | None = None,
) -> None:
    """Write CSV first and provenance second; an interrupted pair is untrusted."""
    validated = validate_token_frame(frame)
    fingerprint = cache_fingerprint(project, record, settings)
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise RuntimeError("The source file changed during processing. Run preprocessing again.")
    csv_path, metadata_path = cache_paths(project, record)
    atomic_write_csv(csv_path, validated)
    atomic_write_json(metadata_path, {
        "schema_version": CACHE_SCHEMA_VERSION, "fingerprint": fingerprint,
        "csv_sha256": _file_digest(csv_path), "text_id": record.text_id,
        "filename": record.filename, "settings": {**DEFAULT_NLP_SETTINGS, **settings},
        "token_count": len(validated),
    })
