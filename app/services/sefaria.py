"""Download explicitly selected Hebrew editions from Sefaria's public API.

The client is synchronous so it can run entirely in a Qt worker thread. It never
writes project texts: callers can import a completed bundle as one transaction.
Complex books are fetched by schema leaf, including introductions and appendices;
an API response resolving a book to its first chapter is never accepted as a book.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from app.utils.io import atomic_write_json

CancelCheck = Callable[[], bool] | None


class SefariaError(RuntimeError):
    """A network, reference, edition, or library-response error."""


class SefariaCancelled(SefariaError):
    """The user cancelled before the completed result could be imported."""


@dataclass(frozen=True)
class Book:
    title: str
    he_title: str = ""
    categories: tuple[str, ...] = ()
    author: str = ""


@dataclass(frozen=True)
class Edition:
    version_title: str
    he_title: str = ""
    source: str = ""
    license: str = ""
    language: str = "he"
    first_section_ref: str = ""


@dataclass(frozen=True)
class BookDetails:
    book: Book
    editions: list[Edition]
    index: dict


@dataclass(frozen=True)
class DownloadRequest:
    book: Book
    edition: Edition
    reference: str = ""
    split_sections: bool = False
    strip_niqqud: bool = False


@dataclass(frozen=True)
class DownloadedText:
    filename: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class DownloadBundle:
    documents: list[DownloadedText]
    warnings: list[str] = field(default_factory=list)
    sections_count: int = 0


def _check_cancelled(cancelled: CancelCheck) -> None:
    if cancelled is not None and cancelled():
        raise SefariaCancelled("Sefaria download cancelled.")


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(filter(None, (_scalar(item) for item in value)))
    if isinstance(value, dict):
        return _scalar(value.get("en") or value.get("he") or value.get("name"))
    return ""


def _hebrew_version(version: dict) -> bool:
    # `language=he` is an RTL bucket and includes Yiddish and Arabic editions.
    actual = str(version.get("actualLanguage") or "").casefold()
    family = str(version.get("languageFamilyName") or "").casefold()
    if actual:
        return actual in {"he", "heb", "hebrew"}
    return family == "hebrew"


def _normalized_ref(reference: str) -> str:
    return " ".join(reference.replace("_", " ").split()).casefold()


class _PlainTextParser(HTMLParser):
    _void = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _breaks = {"p", "div", "br", "hr", "li"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []

    @property
    def skipping(self) -> bool:
        return bool(self.stack and self.stack[-1][1])

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = str(attributes.get("class") or "").casefold()
        skip = self.skipping or tag in {"script", "style", "sup"} or "footnote" in classes
        if tag in self._breaks and not skip:
            self.parts.append("\n")
        if tag not in self._void:
            self.stack.append((tag, skip))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._breaks and not self.skipping:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._breaks and not self.skipping:
            self.parts.append("\n")
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        if not self.skipping:
            self.parts.append(data)


def normalize_text(text: str, *, strip_niqqud: bool = False) -> str:
    """Remove source markup/footnotes without reshaping or reversing Hebrew."""
    parser = _PlainTextParser()
    parser.feed(text)
    parser.close()
    result = unicodedata.normalize("NFC", "".join(parser.parts))
    result = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", result)
    if strip_niqqud:
        result = re.sub(r"[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7]", "", result)
    return "\n".join(line for raw in result.splitlines() if (line := " ".join(raw.split())))


def _filename(book: Book, edition: Edition, reference: str, *, strip_niqqud: bool) -> str:
    label = f"{reference} - {edition.version_title}"
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label).strip(" .")
    safe = re.sub(r"\s+", " ", safe)[:110].rstrip(" .") or "sefaria_text"
    if safe.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        safe = "_" + safe
    identity = f"{book.title}\0{edition.version_title}\0{reference}\0{strip_niqqud}"
    return f"{safe}_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:10]}.txt"


def _schema_leaves(index: dict, book_title: str) -> list[str]:
    schema = index.get("schema")
    if not isinstance(schema, dict):
        raise SefariaError("Sefaria did not return a book schema; completeness cannot be verified.")
    leaves: list[str] = []

    def visit(node: dict, parent: str, root: bool = False) -> None:
        if not root and not node.get("default"):
            title = node.get("title") or next((item.get("text") for item in node.get("titles", []) if item.get("lang") == "en" and item.get("primary")), "")
            if not title:
                raise SefariaError("An unnamed Sefaria schema section cannot be downloaded reliably.")
            parent += ", " + str(title)
        children = node.get("nodes")
        if isinstance(children, list) and children:
            for child in children:
                if not isinstance(child, dict):
                    raise SefariaError("Sefaria returned an invalid book schema.")
                visit(child, parent)
        else:
            leaves.append(parent)

    visit(schema, book_title, root=True)
    return list(dict.fromkeys(leaves))


def _inside_leaf(reference: str, leaf: str) -> bool:
    ref, root = _normalized_ref(reference), _normalized_ref(leaf)
    # A named sibling (book, Introduction) does not belong to the default leaf.
    return ref == root or re.match(re.escape(root) + r"[ :]+\d", ref) is not None


def _address(number: int, address_type: str) -> str:
    if address_type == "Talmud":
        return f"{(number + 1) // 2}{'a' if number % 2 else 'b'}"
    return str(number)


def _section_groups(payload: dict, text: Any) -> list[tuple[str, list[str]]]:
    """Split a jagged array into its bottom-level sections, retaining API refs."""
    ref = str(payload.get("ref") or "")
    book_ref = str(payload.get("book") or payload.get("indexTitle") or ref)
    if isinstance(text, str):
        return [(ref, [text])]
    if not isinstance(text, list):
        raise SefariaError("Sefaria returned an unsupported text structure.")
    if not text or all(isinstance(item, str) for item in text):
        return [(ref, text)]
    addresses = payload.get("addressTypes") or []
    start_sections = payload.get("sections") or []
    offsets = payload.get("index_offsets_by_depth") or {}
    groups: list[tuple[str, list[str]]] = []

    def array_depth(value: Any) -> int:
        return 0 if isinstance(value, str) else 1 + max((array_depth(item) for item in value), default=0)

    nesting = array_depth(text)
    depth = int(payload.get("textDepth") or nesting)
    start_level = max(0, depth - nesting)
    initial_path = list(start_sections[:start_level])

    def visit(value: Any, path: list[int], level: int) -> None:
        if not isinstance(value, list):
            raise SefariaError("Sefaria returned a malformed jagged text array.")
        if not value or all(isinstance(item, str) for item in value):
            suffix = ":".join(_address(number, str(addresses[i]) if i < len(addresses) else "Integer") for i, number in enumerate(path))
            groups.append((book_ref + (" " + suffix if suffix else ""), value))
            return
        for position, child in enumerate(value):
            # Range refs can begin at chapter >1; roots and ordinary sections
            # begin at 1. Offsets represent unusual index numbering schemes.
            start = int(start_sections[level]) if level < len(start_sections) else 1
            offset = offsets.get(str(level + 1), 0)
            if not isinstance(offset, int):
                offset = 0
            visit(child, path + [start + position + offset], level + 1)

    visit(text, initial_path, start_level)
    return groups


class SefariaClient:
    """Small, cancellable Sefaria API client; create one per background worker."""

    BASE_URL = "https://www.sefaria.org"
    CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
    MAX_RESPONSE_BYTES = 64 * 1024 * 1024
    MAX_SECTIONS = 20_000
    MAX_RETRIES = 2

    def __init__(self, cache_dir: Path | None = None, session: Any = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._session = session if session is not None else requests.Session()
        if hasattr(self._session, "headers"):
            self._session.headers.update({"User-Agent": "HebrewStylometry/1.0 (Sefaria text-library importer)", "Accept": "application/json"})
        self._details: dict[str, BookDetails] = {}

    def close(self) -> None:
        self._session.close()

    def _get(self, path: str, *, params: dict | None = None, cancelled: CancelCheck = None) -> Any:
        url = self.BASE_URL + path
        for attempt in range(self.MAX_RETRIES + 1):
            _check_cancelled(cancelled)
            try:
                response = self._session.get(url, params=params, timeout=(10, 30))
            except requests.RequestException as exc:
                _check_cancelled(cancelled)
                raise SefariaError(f"Could not connect to Sefaria: {exc}") from exc
            _check_cancelled(cancelled)
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt < self.MAX_RETRIES:
                    try:
                        retry_after = min(5.0, max(0.0, float(response.headers.get("Retry-After", 0))))
                    except (ValueError, TypeError):
                        retry_after = 0.0
                    delay = max(retry_after, 0.5 * (2 ** attempt))
                    deadline = time.monotonic() + delay
                    while time.monotonic() < deadline:
                        _check_cancelled(cancelled)
                        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                    continue
            if not 200 <= response.status_code < 300:
                detail = ""
                try:
                    error = response.json()
                    detail = ": " + _scalar(error.get("error")) if isinstance(error, dict) and error.get("error") else ""
                except (ValueError, requests.RequestException):
                    pass
                raise SefariaError(f"Sefaria returned HTTP {response.status_code}{detail}.")
            if len(getattr(response, "content", b"")) > self.MAX_RESPONSE_BYTES:
                raise SefariaError("This Sefaria response is too large; download a smaller reference range.")
            try:
                result = response.json()
            except (ValueError, requests.RequestException) as exc:
                raise SefariaError("Sefaria returned invalid JSON. Please try again.") from exc
            if isinstance(result, dict) and result.get("error"):
                raise SefariaError("Sefaria: " + _scalar(result["error"]))
            _check_cancelled(cancelled)
            return result
        raise SefariaError("Sefaria request failed after retries.")  # defensive

    @staticmethod
    def _books(payload: Any) -> list[Book]:
        if not isinstance(payload, list):
            raise SefariaError("Sefaria returned an invalid library catalog.")
        books: dict[str, Book] = {}

        def visit(items: list, categories: tuple[str, ...] = ()) -> None:
            for item in items:
                if not isinstance(item, dict):
                    raise SefariaError("Sefaria returned an invalid catalog entry.")
                if "contents" in item:
                    contents = item["contents"]
                    if not isinstance(contents, list):
                        raise SefariaError("Sefaria returned an invalid category.")
                    category = _scalar(item.get("category"))
                    visit(contents, categories + ((category,) if category else ()))
                elif item.get("title"):
                    title = str(item["title"])
                    values = item.get("categories") or categories
                    if not isinstance(values, (list, tuple)):
                        raise SefariaError("Sefaria returned invalid book categories.")
                    books[title] = Book(title, _scalar(item.get("heTitle")), tuple(map(str, values)), _scalar(item.get("authors") or item.get("author")))

        visit(payload)
        if not books:
            raise SefariaError("Sefaria returned an empty library catalog.")
        return sorted(books.values(), key=lambda book: book.title.casefold())

    def catalog(self, refresh: bool = False, cancelled: CancelCheck = None) -> list[Book]:
        _check_cancelled(cancelled)
        cache = self.cache_dir / "sefaria_catalog.json" if self.cache_dir else None
        if cache and cache.is_file() and not refresh:
            try:
                data = json.loads(cache.read_text(encoding="utf-8"))
                age = time.time() - float(data["fetched_at"])
                if 0 <= age < self.CACHE_TTL_SECONDS:
                    books = self._books(data["catalog"])
                    _check_cancelled(cancelled)
                    return books
            except (OSError, ValueError, TypeError, KeyError, SefariaError):
                _check_cancelled(cancelled)
        payload = self._get("/api/index", params={"include_authors": 1}, cancelled=cancelled)
        books = self._books(payload)
        _check_cancelled(cancelled)
        if cache:
            try:
                atomic_write_json(cache, {"fetched_at": time.time(), "catalog": payload})
            except OSError:
                pass  # A read-only cache must not prevent downloading books.
        return books

    def details(self, book: Book, cancelled: CancelCheck = None) -> BookDetails:
        _check_cancelled(cancelled)
        if book.title in self._details:
            return self._details[book.title]
        encoded = quote(book.title, safe="")
        index = self._get("/api/v2/index/" + encoded, cancelled=cancelled)
        versions = self._get("/api/texts/versions/" + encoded, cancelled=cancelled)
        if not isinstance(index, dict) or not isinstance(versions, list) or any(not isinstance(item, dict) for item in versions):
            raise SefariaError("Sefaria returned invalid book or edition metadata.")
        if _normalized_ref(str(index.get("title") or "")) != _normalized_ref(book.title):
            raise SefariaError("Sefaria returned metadata for a different book.")
        categories = index.get("categories") or book.categories
        resolved = Book(book.title, _scalar(index.get("heTitle")) or book.he_title, tuple(map(str, categories)), _scalar(index.get("authors")) or book.author)
        editions = [Edition(str(item["versionTitle"]), _scalar(item.get("versionTitleInHebrew")), _scalar(item.get("versionSource")), _scalar(item.get("license")), "he", _scalar(item.get("firstSectionRef"))) for item in versions if item.get("versionTitle") and _hebrew_version(item)]
        details = BookDetails(resolved, editions, index)
        _check_cancelled(cancelled)
        self._details[book.title] = details
        return details

    def _text(self, reference: str, book: Book, edition: Edition, cancelled: CancelCheck) -> tuple[dict, Any]:
        payload = self._get("/api/v3/texts/" + quote(reference, safe=""), params={"version": "hebrew|" + edition.version_title, "fill_in_missing_segments": 0, "return_format": "text_only"}, cancelled=cancelled)
        if not isinstance(payload, dict):
            raise SefariaError("Sefaria returned an invalid text response.")
        identity = payload.get("indexTitle") or payload.get("title")
        if not identity or _normalized_ref(str(identity)) != _normalized_ref(book.title):
            raise SefariaError("The reference resolves to a different book. Select its book in the library first.")
        if not payload.get("ref"):
            raise SefariaError("Sefaria returned text without a reference.")
        versions = payload.get("versions")
        if not isinstance(versions, list):
            raise SefariaError("Sefaria did not return the selected edition.")
        selected = next((version for version in versions if isinstance(version, dict) and version.get("versionTitle") == edition.version_title and _hebrew_version(version)), None)
        if selected is None:
            raise SefariaError(f"The selected Hebrew edition '{edition.version_title}' is unavailable for {reference}. No replacement edition was downloaded.")
        if "text" not in selected:
            raise SefariaError("Sefaria returned an edition without text data.")
        return payload, selected["text"]

    def _leaf(self, leaf: str, book: Book, edition: Edition, cancelled: CancelCheck, status: Callable[[str], None] | None) -> list[tuple[str, list[str]]]:
        payload, text = self._text(leaf, book, edition, cancelled)
        if _normalized_ref(str(payload["ref"])) == _normalized_ref(leaf) and payload.get("sections") == []:
            return _section_groups(payload, text)
        # The documented v3 behavior may resolve a bare title to its first
        # section. Follow sections, with explicit completeness/cycle checks.
        first = edition.first_section_ref
        if first and _inside_leaf(first, leaf) and _normalized_ref(str(payload["ref"])) != _normalized_ref(first):
            # Root resolution must not discard early chapters of the selected
            # edition. Its firstSectionRef is edition-specific metadata, unlike
            # firstAvailableSectionRef which can describe a different edition.
            if status:
                status(f"Starting at the selected edition's first section: {first}")
            payload, text = self._text(first, book, edition, cancelled)
            if _normalized_ref(str(payload["ref"])) != _normalized_ref(first):
                raise SefariaError("Sefaria did not return the selected edition's first section; the incomplete book was not imported.")
        elif not first or not _inside_leaf(first, leaf):
            # Named schema leaves do not have individual edition firstSectionRef
            # records. A root may only resolve to its first numbered section.
            resolved_sections = payload.get("sections")
            if not isinstance(resolved_sections, list) or not resolved_sections or any(number != 1 for number in resolved_sections):
                raise SefariaError(f"The first section of {leaf} cannot be verified; the incomplete book was not imported.")
        groups: list[tuple[str, list[str]]] = []
        seen: set[str] = set()
        while True:
            _check_cancelled(cancelled)
            current = str(payload["ref"])
            if not _inside_leaf(current, leaf):
                raise SefariaError(f"Sefaria did not resolve the expected section of {leaf}.")
            normalized = _normalized_ref(current)
            if normalized in seen:
                raise SefariaError("Sefaria repeated a section; the incomplete book was not imported.")
            seen.add(normalized)
            if len(seen) > self.MAX_SECTIONS:
                raise SefariaError("This book exceeds the section download limit. Download a smaller reference.")
            if status:
                status(f"Downloading {current} ({len(seen)} sections)")
            groups.extend(_section_groups(payload, text))
            if "next" not in payload:
                raise SefariaError("Sefaria omitted section navigation; a complete book cannot be verified.")
            next_ref = payload.get("next")
            if not next_ref or not _inside_leaf(str(next_ref), leaf):
                return groups
            payload, text = self._text(str(next_ref), book, edition, cancelled)

    def download(self, request: DownloadRequest, progress: Callable[[int], None] | None = None, status: Callable[[str], None] | None = None, cancelled: CancelCheck = None) -> DownloadBundle:
        _check_cancelled(cancelled)
        if progress:
            progress(0)
        details = self.details(request.book, cancelled=cancelled)
        edition = next((item for item in details.editions if item.version_title == request.edition.version_title), None)
        if edition is None:
            raise SefariaError("The selected Hebrew edition is no longer available. Refresh the book's editions.")
        book = details.book
        requested_ref = request.reference.strip()
        aliases = [book.title, book.he_title, *(details.index.get("titleVariants") or []), *(details.index.get("heTitleVariants") or [])]
        whole_book = not requested_ref or _normalized_ref(requested_ref) in {_normalized_ref(str(alias)) for alias in aliases if alias}
        roots = _schema_leaves(details.index, book.title) if whole_book else [requested_ref]
        if len(roots) > self.MAX_SECTIONS:
            raise SefariaError("This book exceeds the section download limit.")
        groups: list[tuple[str, list[str]]] = []
        warnings: list[str] = []
        for position, root in enumerate(roots):
            _check_cancelled(cancelled)
            if status:
                status(f"Downloading {root} — {edition.version_title}")
            if whole_book:
                current_groups = self._leaf(root, book, edition, cancelled, status)
            else:
                payload, text = self._text(root, book, edition, cancelled)
                current_groups = _section_groups(payload, text)
            if not any(segment.strip() for _, segments in current_groups for segment in segments):
                warnings.append(f"{root}: the selected edition has no text in this section.")
            groups.extend(current_groups)
            if len(groups) > self.MAX_SECTIONS:
                raise SefariaError("This book exceeds the section download limit. Download a smaller reference.")
            if progress:
                progress(min(90, int(90 * (position + 1) / len(roots))))
        normalized: list[tuple[str, str]] = []
        missing = 0
        for reference, segments in groups:
            _check_cancelled(cancelled)
            clean_segments = [normalize_text(segment, strip_niqqud=request.strip_niqqud) for segment in segments]
            missing += sum(not item.strip() for item in clean_segments)
            clean = "\n".join(filter(None, clean_segments))
            if clean:
                normalized.append((reference, clean))
        if missing:
            warnings.append(f"{missing} empty or markup-only segments in the selected edition were omitted; no other edition was substituted.")
        if not normalized:
            raise SefariaError("The selected edition contains no text for this book or reference.")
        output = normalized if request.split_sections else [(book.title if whole_book else requested_ref, "\n\n".join(text for _, text in normalized))]
        documents: list[DownloadedText] = []
        for reference, text in output:
            _check_cancelled(cancelled)
            metadata = self._metadata(details, edition, reference, whole_book=whole_book and not request.split_sections)
            metadata["niqqud_removed"] = str(request.strip_niqqud).lower()
            documents.append(DownloadedText(_filename(book, edition, reference, strip_niqqud=request.strip_niqqud), text + "\n", metadata))
        _check_cancelled(cancelled)
        if progress:
            progress(100)
        _check_cancelled(cancelled)
        return DownloadBundle(documents, list(dict.fromkeys(warnings)), len(normalized))

    def _metadata(self, details: BookDetails, edition: Edition, reference: str, *, whole_book: bool) -> dict[str, str]:
        book, index = details.book, details.index
        comp_date = index.get("compDate")
        date = "–".join(map(str, comp_date)) if isinstance(comp_date, list) else _scalar(comp_date)
        chapter = reference[len(book.title):].lstrip(" ,") if reference.startswith(book.title) and not whole_book else ""
        return {
            "title": book.he_title if whole_book and book.he_title else reference,
            "book": book.title,
            "he_title": book.he_title,
            "ref": reference,
            "source": "Sefaria",
            "source_url": self.BASE_URL + "/" + quote(reference.replace(" ", "_"), safe=""),
            "version_title": edition.version_title,
            "version_source": edition.source,
            "license": edition.license,
            "language": "he",
            "author": book.author,
            "genre": book.categories[0] if book.categories else "",
            "categories": " / ".join(book.categories),
            "period": _scalar(index.get("era")),
            "date": date,
            "composition_date": date,
            # The index's first publication date is not this edition's date.
            "first_publication_date": _scalar(index.get("pubDate")),
            "chapter": chapter,
        }
