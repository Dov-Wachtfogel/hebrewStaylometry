"""Document identities stay intact while display names and groups become readable."""

import pytest

from app.utils.labels import document_display_name, document_groups
from app.project import parse_filename_metadata


@pytest.mark.parametrize("filename, expected", [
    ("II Samuel 9 - Tanach with Text Only_fddf6c435b.txt", "II Samuel 9"),
    ("II Samuel 9 - Tanach with Text Only_fddf6c435b_2.txt", "II Samuel 9"),
    ("Abarbanel on II Kings - Abarbanel, Tel Aviv 1960_b384596862.txt", "Abarbanel on II Kings"),
    ("שמואל ב 9_ABCDEF012345.TXT", "שמואל ב 9"),
    ("כהן_הספר_1900.txt", "כהן הספר 1900"),
])
def test_filename_cleanup(filename, expected):
    assert document_display_name(filename) == expected


def test_references_override_filenames_and_titles_are_bounded():
    name = "book - edition_fddf6c435b.txt"
    metadata = {"ref": "II Samuel 9:1-4", "title": "Old title"}
    assert document_display_name(name, metadata) == "II Samuel 9:1-4"
    long_title = "מאוד ארוך " * 15
    result = document_display_name(name, {"title": long_title})
    assert len(result) <= 42 and result.endswith("…")
    assert document_display_name(name, {"title": long_title}, max_length=None) == long_title.strip()


def test_missing_author_falls_back_to_real_books():
    names = ["a.txt", "b.txt", "c.txt"]
    metadata = {"a.txt": {"book": "II Samuel", "ref": "II Samuel 9"},
                "b.txt": {"book": "II Samuel", "ref": "II Samuel 10"},
                "c.txt": {"book": "I Kings", "ref": "I Kings 2"}}
    for requested in ["auto", "author"]:
        groups, field = document_groups(names, metadata, requested)
        assert field == "book"
        assert groups == {"a.txt": "II Samuel", "b.txt": "II Samuel", "c.txt": "I Kings"}


def test_auto_prefers_populated_metadata_and_partial_authors_are_not_guessed():
    names = ["a.txt", "b.txt"]
    metadata = {"a.txt": {"author": "כהן", "period": "modern"},
                "b.txt": {"author": "לוי", "period": "medieval"}}
    groups, field = document_groups(names, metadata)
    assert field == "author" and groups == {"a.txt": "כהן", "b.txt": "לוי"}
    assert document_groups(names, metadata, "period")[0] == {"a.txt": "modern", "b.txt": "medieval"}
    metadata["b.txt"]["author"] = ""
    assert document_groups(names, metadata, "author")[0]["b.txt"] == "Unknown author"
    assert document_groups(names, metadata, "none") == (None, "")


def test_books_can_be_derived_from_section_names_without_author_metadata():
    names = ["II Samuel 9:1-2 - edition_fddf6c435b.txt", "II Samuel 10 - edition_abc123def4.txt"]
    groups, field = document_groups(names, {})
    assert field == "book" and set(groups.values()) == {"II Samuel"}


def test_imported_sefaria_filename_hash_is_not_used_as_title_or_author_group():
    names = ["II Samuel 9 - Tanach with Text Only_fddf6c435b.txt",
             "II Samuel 10 - Tanach with Text Only_abc123def4.txt"]
    metadata = {name: parse_filename_metadata(name) for name in names}
    assert [document_display_name(name, metadata[name]) for name in names] == ["II Samuel 9", "II Samuel 10"]
    groups, field = document_groups(names, metadata)
    assert field == "book" and set(groups.values()) == {"II Samuel"}
