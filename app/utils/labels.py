"""Readable document names and metadata groups for chart presentation."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

_HASH_TITLE = re.compile(r"[0-9a-f]{8,64}(?:_\d+)?", re.IGNORECASE)


def document_display_name(
    filename: str, metadata: Mapping[str, str] | None = None, *, max_length: int | None = 42,
) -> str:
    """Prefer a reference/title; remove filename extensions, hashes, and editions."""
    values = metadata or {}
    title = str(values.get("ref") or values.get("title") or filename).strip()
    # The generic Author_Title_Date importer can mistake a Sefaria filename's
    # hash for its title. Such metadata must not override the readable filename.
    if _HASH_TITLE.fullmatch(title):
        title = str(filename)
    title = re.sub(r"\.txt$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"_[0-9a-f]{8,64}(?:_\d+)?$", "", title, flags=re.IGNORECASE)
    title = title.split(" - ", 1)[0]
    title = re.sub(r"\s+", " ", title.replace("_", " ")).strip() or str(filename)
    if max_length is not None and len(title) > max_length:
        title = title[:max_length - 1].rstrip() + "…"
    return title


def _group_value(filename: str, metadata: Mapping[str, str], field: str) -> str:
    value = str(metadata.get(field) or "").strip()
    if value.casefold() in {"unspecified", "unknown", "none"}:
        value = ""
    if field == "author" and not metadata.get("ref") and _HASH_TITLE.fullmatch(str(metadata.get("title") or "")):
        inferred_author = re.sub(r"\.txt$", "", filename, flags=re.IGNORECASE).split("_", 1)[0]
        if value == inferred_author:
            value = ""
    if field == "book" and not value:
        title = document_display_name(filename, metadata, max_length=None)
        # A section reference such as II Samuel 9:1-2 belongs to II Samuel.
        value = re.sub(r"(?:\s+|,\s*)\d+(?::\d+)*(?:[-–]\d+(?::\d+)*)?$", "", title)
    return value


def document_groups(
    filenames: Sequence[str], metadata: Mapping[str, Mapping[str, str]], field: str = "auto",
) -> tuple[dict[str, str] | None, str]:
    """Choose a meaningful group, falling back when the requested field is empty.

    Explicit fields with partial coverage retain an 'Unknown <field>' category.
    Auto prefers a populated attribute with variation; missing authors are
    never guessed from a filename. Book names can be derived from references.
    """
    if field == "none":
        return None, ""
    candidates = ["author", "period", "genre"]
    if any((metadata.get(name) or {}).get("book") for name in filenames):
        candidates.insert(0, "book")
    else:
        candidates.append("book")

    def values_for(key):
        return {name: _group_value(name, metadata.get(name, {}), key) for name in filenames}

    values = values_for(field) if field != "auto" else {}
    if not any(values.values()):
        choices = [(key, values_for(key)) for key in candidates]
        populated = [(key, values) for key, values in choices if any(values.values())]
        varied = [(key, values) for key, values in populated if len(set(filter(None, values.values()))) > 1]
        if not populated:
            return None, ""
        field, values = (varied or populated)[0]
    return {name: value or f"Unknown {field}" for name, value in values.items()}, field
