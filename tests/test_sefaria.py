"""Deterministic Sefaria fixtures; no live network is used by this test suite."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote

import pytest
import requests

from app.services.sefaria import (
    Book,
    DownloadRequest,
    Edition,
    SefariaCancelled,
    SefariaClient,
    SefariaError,
    normalize_text,
)


BOOK = Book("Example Book", "ספר לדוגמה", ("Mussar",), "Author")
VERSION = {
    "versionTitle": "Chosen Hebrew Edition",
    "versionTitleInHebrew": "מהדורה עברית",
    "versionSource": "https://example.org/source",
    "license": "CC-BY",
    "language": "he",
    "actualLanguage": "he",
    "languageFamilyName": "hebrew",
    "firstSectionRef": "Example Book 1",
}
EDITION = Edition(VERSION["versionTitle"])
INDEX = {
    "title": BOOK.title,
    "heTitle": BOOK.he_title,
    "categories": ["Mussar"],
    "authors": [{"en": "Author", "he": "מחבר"}],
    "era": "RI",
    "compDate": [1140, 1200],
    "pubDate": [1487],
    "schema": {"nodeType": "JaggedArrayNode", "depth": 2, "addressTypes": ["Integer", "Integer"], "title": BOOK.title},
}


class Response:
    def __init__(self, payload=None, status=200, headers=None, invalid_json=False):
        self.payload = payload
        self.status_code = status
        self.headers = headers or {}
        self.content = b"fixture"
        self.invalid_json = invalid_json

    def json(self):
        if self.invalid_json:
            raise ValueError("Invalid JSON")
        return deepcopy(self.payload)


class Session:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.headers = {}
        self.closed = False

    def get(self, url, params=None, timeout=None):
        path = unquote(url.removeprefix(SefariaClient.BASE_URL))
        self.calls.append((path, params, timeout))
        assert timeout == (10, 30)
        if path not in self.routes:
            raise AssertionError(f"Unexpected fixture request: {path}")
        route = self.routes[path]
        if isinstance(route, list) and route and isinstance(route[0], (Response, Exception)):
            route = route.pop(0)
        if isinstance(route, Exception):
            raise route
        if callable(route):
            route = route(params)
        return route if isinstance(route, Response) else Response(route)

    def close(self):
        self.closed = True


def text_response(text, *, reference=BOOK.title, sections=None, next_ref=None, **extra):
    return {
        "ref": reference,
        "book": BOOK.title,
        "indexTitle": BOOK.title,
        "sections": [] if sections is None else sections,
        "textDepth": 2,
        "addressTypes": ["Integer", "Integer"],
        "next": next_ref,
        "versions": [{**VERSION, "text": text}],
        **extra,
    }


def client_for(text, *, index=None, versions=None, extra_routes=None, cache_dir=None):
    routes = {
        "/api/v2/index/" + BOOK.title: INDEX if index is None else index,
        "/api/texts/versions/" + BOOK.title: [VERSION] if versions is None else versions,
        "/api/v3/texts/" + BOOK.title: text,
    }
    routes.update(extra_routes or {})
    session = Session(routes)
    return SefariaClient(cache_dir=cache_dir, session=session), session


def test_whole_simple_book_retains_every_chapter_exact_edition_and_metadata():
    client, session = client_for(text_response([["ראשון", "פסוק"], ["שני"], ["שלישי"], ["רביעי"]]))
    progress, status = [], []
    bundle = client.download(DownloadRequest(BOOK, EDITION), progress.append, status.append)
    assert len(bundle.documents) == 1
    assert bundle.sections_count == 4
    assert bundle.documents[0].text == "ראשון\nפסוק\n\nשני\n\nשלישי\n\nרביעי\n"
    metadata = bundle.documents[0].metadata
    assert metadata["ref"] == BOOK.title
    assert metadata["version_source"] == VERSION["versionSource"]
    assert metadata["license"] == "CC-BY"
    assert metadata["date"] == "1140–1200"
    assert metadata["first_publication_date"] == "1487"
    assert metadata["chapter"] == ""
    assert all(isinstance(value, str) for value in metadata.values())
    assert progress[0] == 0 and progress[-1] == 100
    assert progress == sorted(progress)
    assert status
    _, params, _ = session.calls[-1]
    assert params == {"version": "hebrew|Chosen Hebrew Edition", "fill_in_missing_segments": 0, "return_format": "text_only"}


def test_split_sections_preserves_labels_and_filename_uniqueness():
    client, _ = client_for(text_response([["אחד"], ["שניים"]]))
    bundle = client.download(DownloadRequest(BOOK, EDITION, split_sections=True))
    assert [doc.metadata["ref"] for doc in bundle.documents] == ["Example Book 1", "Example Book 2"]
    assert [doc.metadata["chapter"] for doc in bundle.documents] == ["1", "2"]
    assert len({doc.filename for doc in bundle.documents}) == 2
    assert all(doc.filename.endswith(".txt") for doc in bundle.documents)


def test_complex_book_fetches_introduction_default_and_appendix_in_schema_order():
    index = deepcopy(INDEX)
    index["schema"] = {"nodes": [
        {"title": "Introduction", "depth": 1},
        {"default": True, "depth": 2},
        {"title": "Appendix", "depth": 1},
    ]}
    intro = text_response(["פתיחה"], reference=BOOK.title + ", Introduction", book=BOOK.title + ", Introduction", textDepth=1)
    appendix = text_response(["סוף"], reference=BOOK.title + ", Appendix", book=BOOK.title + ", Appendix", textDepth=1)
    client, session = client_for(text_response([["תוכן"], ["עוד תוכן"]]), index=index, extra_routes={
        "/api/v3/texts/" + BOOK.title + ", Introduction": intro,
        "/api/v3/texts/" + BOOK.title + ", Appendix": appendix,
    })
    bundle = client.download(DownloadRequest(BOOK, EDITION, split_sections=True))
    assert bundle.sections_count == 4
    assert [doc.metadata["ref"] for doc in bundle.documents] == [BOOK.title + ", Introduction", BOOK.title + " 1", BOOK.title + " 2", BOOK.title + ", Appendix"]
    assert len(session.calls) == 5


def test_first_section_only_api_response_is_followed_to_last_section():
    first = text_response(["אחד"], reference=BOOK.title + " 1", sections=[1], next_ref=BOOK.title + " 2")
    second = text_response(["שניים"], reference=BOOK.title + " 2", sections=[2], next_ref=BOOK.title + " 3")
    third = text_response(["שלוש"], reference=BOOK.title + " 3", sections=[3])
    client, _ = client_for(first, extra_routes={"/api/v3/texts/" + BOOK.title + " 2": second, "/api/v3/texts/" + BOOK.title + " 3": third})
    bundle = client.download(DownloadRequest(BOOK, EDITION))
    assert bundle.sections_count == 3
    assert "שלוש" in bundle.documents[0].text


def test_root_resolving_to_later_chapter_starts_at_selected_editions_first_section():
    first = text_response(["ראשון"], reference=BOOK.title + " 1", sections=[1], next_ref=BOOK.title + " 2")
    later = text_response(["שני"], reference=BOOK.title + " 2", sections=[2])
    client, _ = client_for(later, extra_routes={"/api/v3/texts/" + BOOK.title + " 1": first, "/api/v3/texts/" + BOOK.title + " 2": later})
    bundle = client.download(DownloadRequest(BOOK, EDITION))
    assert bundle.sections_count == 2
    assert bundle.documents[0].text == "ראשון\n\nשני\n"


def test_root_resolving_to_later_chapter_without_edition_first_section_is_rejected():
    later = text_response(["שני"], reference=BOOK.title + " 2", sections=[2])
    client, _ = client_for(later, versions=[{**VERSION, "firstSectionRef": ""}])
    with pytest.raises(SefariaError, match="first section.*cannot be verified"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_hebrew_bare_title_downloads_all_schema_leaves():
    index = deepcopy(INDEX)
    index["schema"] = {"nodes": [{"title": "Introduction", "depth": 1}, {"default": True, "depth": 2}]}
    intro = text_response(["פתיחה"], reference=BOOK.title + ", Introduction", book=BOOK.title + ", Introduction", textDepth=1)
    client, _ = client_for(text_response([["תוכן"]]), index=index, extra_routes={"/api/v3/texts/" + BOOK.title + ", Introduction": intro})
    bundle = client.download(DownloadRequest(BOOK, EDITION, reference=BOOK.he_title))
    assert bundle.sections_count == 2
    assert bundle.documents[0].text == "פתיחה\n\nתוכן\n"
    assert bundle.documents[0].metadata["ref"] == BOOK.title


def test_navigation_never_downloads_next_book_or_named_sibling():
    first = text_response(["אחד"], reference=BOOK.title + " 1", sections=[1], next_ref="Example Book Two 1")
    client, session = client_for(first)
    assert client.download(DownloadRequest(BOOK, EDITION)).sections_count == 1
    assert len(session.calls) == 3


def test_navigation_cycle_fails_without_returning_partial_book():
    first = text_response(["אחד"], reference=BOOK.title + " 1", sections=[1], next_ref=BOOK.title + " 1")
    client, _ = client_for(first, extra_routes={"/api/v3/texts/" + BOOK.title + " 1": first})
    with pytest.raises(SefariaError, match="repeated a section"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_omitted_navigation_fails_for_first_chapter_response():
    first = text_response(["אחד"], reference=BOOK.title + " 1", sections=[1])
    del first["next"]
    client, _ = client_for(first)
    with pytest.raises(SefariaError, match="complete book cannot be verified"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_explicit_reference_is_not_expanded_to_rest_of_book():
    chapter = text_response(["שניים"], reference=BOOK.title + " 2", sections=[2], next_ref=BOOK.title + " 3")
    client, session = client_for({}, extra_routes={"/api/v3/texts/" + BOOK.title + " 2": chapter})
    bundle = client.download(DownloadRequest(BOOK, EDITION, reference=BOOK.title + " 2"))
    assert bundle.documents[0].text == "שניים\n"
    assert bundle.documents[0].metadata["ref"] == BOOK.title + " 2"
    assert len(session.calls) == 3


def test_explicit_range_labels_begin_at_requested_chapter():
    chapter = text_response([["שני"], ["שלישי"]], reference=BOOK.title + " 2-3", sections=[2])
    client, _ = client_for({}, extra_routes={"/api/v3/texts/" + BOOK.title + " 2-3": chapter})
    bundle = client.download(DownloadRequest(BOOK, EDITION, reference=BOOK.title + " 2-3", split_sections=True))
    assert [doc.metadata["ref"] for doc in bundle.documents] == [BOOK.title + " 2", BOOK.title + " 3"]


def test_talmud_addresses_use_daf_and_side():
    response = text_response([[], [], ["ב א"], ["ב ב"], ["ג א"]], addressTypes=["Talmud", "Integer"])
    client, _ = client_for(response)
    bundle = client.download(DownloadRequest(BOOK, EDITION, split_sections=True))
    assert [doc.metadata["ref"] for doc in bundle.documents] == [BOOK.title + " 2a", BOOK.title + " 2b", BOOK.title + " 3a"]


@pytest.mark.parametrize("foreign", [
    {"actualLanguage": "yi", "languageFamilyName": "yiddish"},
    {"actualLanguage": "ar", "languageFamilyName": "arabic"},
    {"actualLanguage": "en", "languageFamilyName": "english"},
    {"actualLanguage": "yi", "languageFamilyName": "hebrew"},
    {"actualLanguage": "", "languageFamilyName": ""},
])
def test_rtl_bucket_does_not_admit_non_hebrew_editions(foreign):
    client, _ = client_for({}, versions=[VERSION, {**VERSION, "versionTitle": "Foreign", **foreign}])
    assert [item.version_title for item in client.details(BOOK).editions] == [EDITION.version_title]


def test_hebrew_family_without_actual_language_is_accepted():
    version = {**VERSION, "actualLanguage": ""}
    client, _ = client_for({}, versions=[version])
    assert len(client.details(BOOK).editions) == 1


def test_substituted_edition_is_rejected_even_if_api_returns_text():
    response = text_response([["תוכן"]])
    response["versions"][0]["versionTitle"] = "Different Hebrew Edition"
    response["warnings"] = ["Requested edition not found"]
    client, _ = client_for(response)
    with pytest.raises(SefariaError, match="No replacement edition"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_selected_edition_removed_from_catalog_is_rejected():
    client, _ = client_for({}, versions=[])
    with pytest.raises(SefariaError, match="no longer available"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_reference_resolving_to_different_book_is_rejected():
    response = text_response(["זר"], indexTitle="Different Book")
    client, _ = client_for({}, extra_routes={"/api/v3/texts/Other 1": response})
    with pytest.raises(SefariaError, match="different book"):
        client.download(DownloadRequest(BOOK, EDITION, reference="Other 1"))


def test_normalization_removes_markup_footnotes_entities_and_preserves_niqqud():
    source = '<p>שָׁלוֹם&nbsp;עולם<sup>1</sup><i class="footnote">הערה</i></p><p>אמת<br>נכון<script>bad()</script><style>bad</style></p>'
    assert normalize_text(source) == "שָׁלוֹם עולם\nאמת\nנכון"
    assert normalize_text("שָׁלוֹם־עוֹלָם׃", strip_niqqud=True) == "שלום־עולם׃"
    assert normalize_text("א\u200fב&thinsp;ג &amp; ד") == "אב ג & ד"


def test_default_download_preserves_niqqud_option_changes_filename():
    client, _ = client_for(text_response([["שָׁלוֹם"]]))
    vocalized = client.download(DownloadRequest(BOOK, EDITION)).documents[0]
    plain = client.download(DownloadRequest(BOOK, EDITION, strip_niqqud=True)).documents[0]
    assert vocalized.text == "שָׁלוֹם\n"
    assert plain.text == "שלום\n"
    assert vocalized.filename != plain.filename


def test_missing_segments_warn_without_edition_fallback():
    client, _ = client_for(text_response([["", "תוכן", "<sup>2</sup>"]]))
    bundle = client.download(DownloadRequest(BOOK, EDITION))
    assert bundle.documents[0].text == "תוכן\n"
    assert len(bundle.warnings) == 1
    assert "2 empty" in bundle.warnings[0]


def test_no_text_is_an_error():
    client, _ = client_for(text_response([[], []]))
    with pytest.raises(SefariaError, match="contains no text"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_filenames_contain_no_paths_and_are_bounded():
    version = {**VERSION, "versionTitle": "../../bad:edition?\\name" * 20}
    response = text_response([["תוכן"]])
    response["versions"] = [{**version, "text": [["תוכן"]]}]
    client, _ = client_for(response, versions=[version])
    doc = client.download(DownloadRequest(BOOK, Edition(version["versionTitle"]))).documents[0]
    assert Path(doc.filename).name == doc.filename
    assert not any(char in doc.filename for char in '<>:"/\\|?*')
    assert len(doc.filename) <= 125


CATALOG = [{"category": "Mussar", "contents": [{"title": BOOK.title, "heTitle": BOOK.he_title, "authors": [{"en": "Author"}]}]}]


def test_catalog_is_flattened_sorted_and_atomically_cached(tmp_path):
    session = Session({"/api/index": CATALOG})
    client = SefariaClient(tmp_path, session)
    books = client.catalog()
    assert books == [BOOK]
    assert session.calls[0][1] == {"include_authors": 1}
    assert client.catalog() == books
    assert len(session.calls) == 1
    assert json.loads((tmp_path / "sefaria_catalog.json").read_text())["catalog"] == CATALOG


@pytest.mark.parametrize("cached", ["garbage", json.dumps({"fetched_at": 0, "catalog": CATALOG}), json.dumps({"fetched_at": time.time(), "catalog": {"bad": "shape"}})])
def test_corrupt_or_expired_cache_refetches(tmp_path, cached):
    (tmp_path / "sefaria_catalog.json").write_text(cached)
    session = Session({"/api/index": CATALOG})
    assert SefariaClient(tmp_path, session).catalog() == [BOOK]
    assert len(session.calls) == 1


def test_refresh_bypasses_valid_catalog_cache(tmp_path):
    session = Session({"/api/index": CATALOG})
    client = SefariaClient(tmp_path, session)
    client.catalog()
    client.catalog(refresh=True)
    assert len(session.calls) == 2


def test_details_are_cached_per_client_and_close_closes_session():
    client, session = client_for({})
    assert client.details(BOOK) is client.details(BOOK)
    assert len(session.calls) == 2
    client.close()
    assert session.closed


@pytest.mark.parametrize("response,match", [
    (Response({}, status=404), "HTTP 404"),
    (Response({"error": "Unknown reference"}), "Unknown reference"),
    (Response(invalid_json=True), "invalid JSON"),
    (requests.Timeout("Timed out"), "Could not connect"),
])
def test_network_and_api_errors_are_descriptive(response, match):
    session = Session({"/api/index": response})
    with pytest.raises(SefariaError, match=match):
        SefariaClient(session=session).catalog()


def test_transient_error_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("app.services.sefaria.time.sleep", lambda _: None)
    monotonic_calls = iter(range(100))
    monkeypatch.setattr("app.services.sefaria.time.monotonic", lambda: next(monotonic_calls))
    session = Session({"/api/index": [Response({}, 503), Response({}, 429), Response(CATALOG)]})
    assert SefariaClient(session=session).catalog() == [BOOK]
    assert len(session.calls) == 3


def test_persistent_server_error_stops_after_two_retries(monkeypatch):
    monkeypatch.setattr("app.services.sefaria.time.sleep", lambda _: None)
    monotonic_calls = iter(range(100))
    monkeypatch.setattr("app.services.sefaria.time.monotonic", lambda: next(monotonic_calls))
    session = Session({"/api/index": Response({}, 503)})
    with pytest.raises(SefariaError, match="HTTP 503"):
        SefariaClient(session=session).catalog()
    assert len(session.calls) == 3


def test_cancel_before_start_does_not_request_network():
    session = Session({})
    with pytest.raises(SefariaCancelled):
        SefariaClient(session=session).catalog(cancelled=lambda: True)
    assert session.calls == []


def test_cancel_in_retry_delay_is_cooperative():
    session = Session({"/api/index": Response({}, 429, headers={"Retry-After": "5"})})
    checks = []

    def cancelled():
        checks.append(True)
        return len(checks) >= 4

    with pytest.raises(SefariaCancelled):
        SefariaClient(session=session).catalog(cancelled=cancelled)
    assert len(session.calls) == 1


def test_cancel_after_final_progress_never_returns_importable_bundle():
    client, _ = client_for(text_response([["תוכן"]]))
    cancel = False

    def progress(value):
        nonlocal cancel
        if value == 100:
            cancel = True

    with pytest.raises(SefariaCancelled):
        client.download(DownloadRequest(BOOK, EDITION), progress=progress, cancelled=lambda: cancel)


def test_section_limit_fails_without_partial_bundle():
    client, _ = client_for(text_response([["אחד"], ["שניים"]]))
    client.MAX_SECTIONS = 1
    with pytest.raises(SefariaError, match="section download limit"):
        client.download(DownloadRequest(BOOK, EDITION))


def test_missing_book_schema_fails_explicitly():
    index = deepcopy(INDEX)
    del index["schema"]
    client, _ = client_for(text_response([["תוכן"]]), index=index)
    with pytest.raises(SefariaError, match="completeness cannot be verified"):
        client.download(DownloadRequest(BOOK, EDITION))
