"""Project persistence, import safety, and Hebrew data-integrity checks."""

from __future__ import annotations

import codecs
import json
from pathlib import Path

import pytest

from app.project import CONFIG_FILENAME, DEFAULT_SETTINGS, Project, parse_filename_metadata
from app.utils.hebrew import normalize_hebrew
from app.utils.io import atomic_write_json, read_hebrew_text


def _source(tmp_path: Path, filename: str = "Author_Title_1900.txt") -> Path:
    source_dir = tmp_path / "sources"
    source_dir.mkdir(exist_ok=True)
    source = source_dir / filename
    source.write_text("שלום עולם\n", encoding="utf-8")
    return source


def _config(root: Path) -> dict:
    return json.loads((root / CONFIG_FILENAME).read_text(encoding="utf-8"))


def test_roundtrip_preserves_metadata_settings_and_original_bytes(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project", "Hebrew corpus")
    source = _source(tmp_path, "רבי_ספר_1900.txt")
    [record] = project.import_texts([source])
    project.update_metadata(record.text_id, "Genre", "אגדה")
    project.settings["analysis"]["n_mfw"] = 100
    project.save()

    reopened = Project.open(project.root)
    assert reopened.name == "Hebrew corpus"
    assert reopened.get_record(record.text_id).metadata["genre"] == "אגדה"
    assert reopened.settings["analysis"]["n_mfw"] == 100
    assert (reopened.raw_dir / record.filename).read_bytes() == source.read_bytes()
    assert all(path.is_dir() for path in (reopened.raw_dir, reopened.processed_dir, reopened.outputs_dir))


def test_project_creation_does_not_overwrite_configuration(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    original = (project.root / CONFIG_FILENAME).read_bytes()
    with pytest.raises(FileExistsError):
        Project.create(project.root, "Replacement")
    assert (project.root / CONFIG_FILENAME).read_bytes() == original


def test_default_settings_are_not_shared_between_projects(tmp_path: Path) -> None:
    first = Project.create(tmp_path / "one")
    second = Project.create(tmp_path / "two")
    first.settings["analysis"]["stop_words"].append("של")
    assert second.settings["analysis"]["stop_words"] == []


def test_open_merges_missing_defaults_and_preserves_extra_settings(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    data = _config(project.root)
    data["settings"] = {
        "nlp": {
            "batch_size": 2,
            "local_files_only": True,
            "code_revision": "reviewed-commit",
            "max_sequence_length": 128,
            "custom_option": {"nested": ["value"]},
        },
        "analysis": {"n_mfw": 100},
        "custom_section": {"enabled": True},
    }
    atomic_write_json(project.root / CONFIG_FILENAME, data)

    reopened = Project.open(project.root)
    assert reopened.settings["nlp"] == {
        **DEFAULT_SETTINGS["nlp"], **data["settings"]["nlp"],
    }
    assert reopened.settings["analysis"] == {**DEFAULT_SETTINGS["analysis"], "n_mfw": 100}
    assert reopened.settings["custom_section"] == {"enabled": True}
    reopened.settings["analysis"]["stop_words"].append("של")
    assert DEFAULT_SETTINGS["analysis"]["stop_words"] == []
    reopened.save()
    assert Project.open(project.root).settings == reopened.settings


@pytest.mark.parametrize(
    ("name", "invalid"),
    [
        ("batch_size", True), ("batch_size", False), ("batch_size", 0),
        ("batch_size", -1), ("batch_size", 1.5), ("batch_size", "8"),
        ("model_name", None), ("model_name", ""), ("model_name", "  "),
        ("revision", 42), ("revision", ""), ("device", []), ("device", "\t"),
        ("strip_niqqud", "false"), ("strip_niqqud", 1),
        ("segmented", None), ("segmented", 0),
        ("backend", "other"), ("backend", None),
        ("local_files_only", "true"), ("local_files_only", 1),
        ("code_revision", 42), ("code_revision", None),
        ("max_sequence_length", True), ("max_sequence_length", 3),
        ("max_sequence_length", "128"),
    ],
)
def test_open_rejects_invalid_nlp_values(tmp_path: Path, name: str, invalid: object) -> None:
    project = Project.create(tmp_path / "project")
    data = _config(project.root)
    data["settings"]["nlp"][name] = invalid
    atomic_write_json(project.root / CONFIG_FILENAME, data)
    with pytest.raises(ValueError, match=name):
        Project.open(project.root)


@pytest.mark.parametrize("invalid", [None, [], "dicta", 8, True])
def test_open_rejects_invalid_nlp_section(tmp_path: Path, invalid: object) -> None:
    project = Project.create(tmp_path / "project")
    data = _config(project.root)
    data["settings"]["nlp"] = invalid
    atomic_write_json(project.root / CONFIG_FILENAME, data)
    with pytest.raises(ValueError, match="nlp.*JSON object"):
        Project.open(project.root)


def test_save_rejects_invalid_nlp_without_replacing_config(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    previous = (project.root / CONFIG_FILENAME).read_bytes()
    project.settings["nlp"]["batch_size"] = True
    with pytest.raises(ValueError, match="batch_size"):
        project.save()
    assert (project.root / CONFIG_FILENAME).read_bytes() == previous


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Author_Title_1900.txt", {"author": "Author", "title": "Title", "date": "1900"}),
        ("Author_Long_Title_1900.txt", {"author": "Author", "title": "Long_Title", "date": "1900"}),
        ("רבי_ספר.txt", {"author": "רבי", "title": "ספר", "date": ""}),
        ("ספר.txt", {"author": "", "title": "ספר", "date": ""}),
    ],
)
def test_filename_metadata(filename: str, expected: dict[str, str]) -> None:
    actual = parse_filename_metadata(filename)
    assert all(actual[key] == value for key, value in expected.items())


def test_duplicate_import_preserves_both_files_and_original_metadata(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    source = _source(tmp_path)
    first, second = project.import_texts([source, source])
    assert first.filename == "Author_Title_1900.txt"
    assert second.filename == "Author_Title_1900_2.txt"
    assert first.text_id != second.text_id
    assert second.metadata["date"] == "1900"
    assert len(list(project.raw_dir.glob("*.txt"))) == 2


def test_import_validates_all_sources_before_copying(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    with pytest.raises(FileNotFoundError):
        project.import_texts([_source(tmp_path), tmp_path / "missing.txt"])
    assert project.texts == []
    assert list(project.raw_dir.iterdir()) == []


def test_content_import_preserves_unicode_and_saves_metadata_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.create(tmp_path / "project")
    initial_config = (project.root / CONFIG_FILENAME).read_bytes()
    text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים\r\nשורה שנייה — שלום עולם!\n"
    original_save = project.save
    saved_records: list[list[dict[str, str]]] = []

    def inspect_save() -> None:
        assert (project.root / CONFIG_FILENAME).read_bytes() == initial_config
        assert len(list(project.raw_dir.iterdir())) == 2
        saved_records.append([record.metadata.copy() for record in project.texts])
        original_save()

    monkeypatch.setattr(project, "save", inspect_save)
    first, second = project.import_text_content([
        ("Unknown_Book_1900.txt", text, {"Title": "בראשית", "Author": "", "Ref": "Genesis", "Source": "Sefaria"}),
        ("Unknown_Book_1900.txt", "שמות\n", {"Title": "שמות", "Ref": "Exodus", "Chapter": 2}),
    ])
    assert (project.raw_dir / first.filename).read_bytes() == text.encode("utf-8")
    assert second.filename == "Unknown_Book_1900_2.txt"
    assert second.metadata["date"] == "1900"
    assert second.metadata["chapter"] == "2"
    assert len(saved_records) == 1
    assert saved_records[0][0]["title"] == "בראשית"
    assert saved_records[0][0]["author"] == ""
    assert saved_records[0][0]["source"] == "Sefaria"
    reopened = Project.open(project.root)
    assert [(record.filename, record.metadata) for record in reopened.texts] == [
        (record.filename, record.metadata) for record in (first, second)
    ]


def test_content_import_avoids_case_insensitive_existing_files_and_directories(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    unmanaged = project.raw_dir / "Book.TXT"
    unmanaged.write_bytes(b"Original user file")
    reserved = project.raw_dir / "book_2.txt"
    reserved.mkdir()
    first, second = project.import_text_content([
        ("book.txt", "שלום", {}), ("BOOK.txt", "עולם", {}),
    ])
    assert first.filename == "book_3.txt"
    assert second.filename == "BOOK_4.txt"
    assert unmanaged.read_bytes() == b"Original user file"
    assert reserved.is_dir()
    assert first.text_id != second.text_id
    assert first.metadata["title"] == "book"
    assert second.metadata["title"] == "BOOK"


@pytest.mark.parametrize("filename", ["../outside.txt", "..\\outside.txt", "bad.csv", "", "NUL\x00.txt", None, 3])
def test_content_import_rejects_invalid_filename_before_any_write(tmp_path: Path, filename: object) -> None:
    project = Project.create(tmp_path / "project")
    config = (project.root / CONFIG_FILENAME).read_bytes()
    with pytest.raises(ValueError, match="filename"):
        project.import_text_content([("good.txt", "שלום", {}), (filename, "עולם", {})])
    assert project.texts == []
    assert list(project.raw_dir.iterdir()) == []
    assert (project.root / CONFIG_FILENAME).read_bytes() == config


@pytest.mark.parametrize("text", ["", " \n\t", None, 3, b"text"])
def test_content_import_rejects_empty_or_nonstring_text_before_any_write(tmp_path: Path, text: object) -> None:
    project = Project.create(tmp_path / "project")
    config = (project.root / CONFIG_FILENAME).read_bytes()
    with pytest.raises(ValueError, match="nonempty string"):
        project.import_text_content([("good.txt", "שלום", {}), ("bad.txt", text, {})])
    assert project.texts == []
    assert list(project.raw_dir.iterdir()) == []
    assert (project.root / CONFIG_FILENAME).read_bytes() == config


@pytest.mark.parametrize("metadata", [None, [], {"Title": []}, {"Ref": {}}, {"": "value"}, {42: "value"}])
def test_content_import_validates_all_metadata_before_any_write(tmp_path: Path, metadata: object) -> None:
    project = Project.create(tmp_path / "project")
    config = (project.root / CONFIG_FILENAME).read_bytes()
    with pytest.raises(ValueError):
        project.import_text_content([("good.txt", "שלום", {}), ("bad.txt", "עולם", metadata)])
    assert project.texts == []
    assert list(project.raw_dir.iterdir()) == []
    assert (project.root / CONFIG_FILENAME).read_bytes() == config


def test_content_import_validates_entire_generator_and_unicode_before_writes(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")

    def interrupted_download():
        yield "good.txt", "שלום", {}
        raise OSError("Download incomplete")

    with pytest.raises(OSError, match="Download incomplete"):
        project.import_text_content(interrupted_download())
    with pytest.raises(UnicodeEncodeError):
        project.import_text_content([("good.txt", "שלום", {}), ("bad.txt", "\ud800", {})])
    assert project.texts == []
    assert list(project.raw_dir.iterdir()) == []


@pytest.mark.parametrize("document", [None, ("missing.txt", "שלום"), ("too.txt", "שלום", {}, "extra")])
def test_content_import_rejects_malformed_document(tmp_path: Path, document: object) -> None:
    project = Project.create(tmp_path / "project")
    with pytest.raises(ValueError, match="filename, text, and metadata"):
        project.import_text_content([document])
    assert list(project.raw_dir.iterdir()) == []


def test_empty_content_import_does_not_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project.create(tmp_path / "project")

    def unexpected_save() -> None:
        pytest.fail("An empty batch must not rewrite the project configuration")

    monkeypatch.setattr(project, "save", unexpected_save)
    assert project.import_text_content(iter(())) == []


@pytest.mark.parametrize("failure_stage", ["open", "write", "flush", "save"])
def test_content_import_failure_rolls_back_only_its_own_records_and_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    project = Project.create(tmp_path / "project")
    [existing] = project.import_text_content([("existing.txt", "מקור", {"author": "Original author"})])
    unmanaged = project.raw_dir / "unmanaged.txt"
    unmanaged.write_bytes(b"User data")
    previous_config = (project.root / CONFIG_FILENAME).read_bytes()
    previous_content = (project.raw_dir / existing.filename).read_bytes()

    if failure_stage in {"open", "write"}:
        original_open = Path.open

        class PartialWrite:
            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()

            def write(self, content: bytes) -> None:
                self.handle.write(content[:3])
                raise OSError("Disk is full")

        def failing_open(path: Path, mode: str = "r", *args, **kwargs):
            if path == project.raw_dir / "second.txt" and mode == "xb":
                if failure_stage == "open":
                    raise OSError("Disk is full")
                return PartialWrite(original_open(path, mode, *args, **kwargs))
            return original_open(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", failing_open)
    elif failure_stage == "flush":
        import app.project as project_module

        def fail_fsync(descriptor: int) -> None:
            raise OSError("Disk is full")

        monkeypatch.setattr(project_module.os, "fsync", fail_fsync)
    else:
        def fail_save() -> None:
            raise OSError("Disk is full")

        monkeypatch.setattr(project, "save", fail_save)

    with pytest.raises(OSError, match="Disk is full"):
        project.import_text_content([("first.txt", "ראשון", {}), ("second.txt", "שני", {})])
    assert project.texts == [existing]
    assert project.texts[0] is existing
    assert (project.raw_dir / existing.filename).read_bytes() == previous_content
    assert unmanaged.read_bytes() == b"User data"
    assert set(path.name for path in project.raw_dir.iterdir()) == {existing.filename, unmanaged.name}
    assert (project.root / CONFIG_FILENAME).read_bytes() == previous_config


def test_content_import_exclusive_creation_preserves_concurrent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = Project.create(tmp_path / "project")
    previous_config = (project.root / CONFIG_FILENAME).read_bytes()
    concurrent = project.raw_dir / "second.txt"
    original_open = Path.open

    def concurrent_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == concurrent and mode == "xb":
            with original_open(path, "wb") as handle:
                handle.write(b"Written by another process")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", concurrent_open)
    with pytest.raises(FileExistsError):
        project.import_text_content([("first.txt", "ראשון", {}), ("second.txt", "שני", {})])
    assert project.texts == []
    assert concurrent.read_bytes() == b"Written by another process"
    assert list(project.raw_dir.iterdir()) == [concurrent]
    assert (project.root / CONFIG_FILENAME).read_bytes() == previous_config


@pytest.mark.parametrize("as_list", [False, True])
def test_external_metadata_mapping_and_record_list(tmp_path: Path, as_list: bool) -> None:
    project = Project.create(tmp_path / "project")
    [record] = project.import_texts([_source(tmp_path)])
    values = {"Author": "New author", "Period": "Medieval", "Chapter": 3}
    data = [{"filename": record.filename, **values}] if as_list else {record.filename: values}
    metadata_file = tmp_path / "metadata.json"
    atomic_write_json(metadata_file, data)
    assert project.import_metadata(metadata_file) == 1
    reopened = Project.open(project.root)
    assert reopened.texts[0].metadata["author"] == "New author"
    assert reopened.texts[0].metadata["period"] == "Medieval"
    assert reopened.texts[0].metadata["chapter"] == "3"


def test_invalid_external_metadata_is_atomic(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    [record] = project.import_texts([_source(tmp_path)])
    original = record.metadata.copy()
    metadata_file = tmp_path / "metadata.json"
    atomic_write_json(metadata_file, [
        {"filename": record.filename, "author": "Changed"},
        {"filename": "../escape.txt", "author": "Invalid"},
    ])
    with pytest.raises(ValueError):
        project.import_metadata(metadata_file)
    assert record.metadata == original
    assert Project.open(project.root).texts[0].metadata == original


@pytest.mark.parametrize("key", ["author", "custom_field"])
def test_metadata_edit_rolls_back_when_save_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    project = Project.create(tmp_path / "project")
    [record] = project.import_texts([_source(tmp_path)])
    original_metadata = record.metadata.copy()
    original_config = (project.root / CONFIG_FILENAME).read_bytes()

    def fail_save() -> None:
        raise OSError("Disk is full")

    monkeypatch.setattr(project, "save", fail_save)
    with pytest.raises(OSError, match="Disk is full"):
        project.update_metadata(record.text_id, key, "Changed")
    assert record.metadata == original_metadata
    assert (project.root / CONFIG_FILENAME).read_bytes() == original_config


@pytest.mark.parametrize("filename", ["../outside.txt", "/tmp/outside.txt", "..\\outside.txt", "C:outside.txt", "bad.csv"])
def test_open_rejects_unsafe_record_filenames(tmp_path: Path, filename: str) -> None:
    project = Project.create(tmp_path / "project")
    data = _config(project.root)
    data["texts"] = [{"text_id": "text1", "filename": filename, "metadata": {}}]
    atomic_write_json(project.root / CONFIG_FILENAME, data)
    with pytest.raises(ValueError):
        Project.open(project.root)


@pytest.mark.parametrize("duplicate", ["text_id", "filename"])
def test_open_rejects_duplicate_ids_and_case_insensitive_filenames(tmp_path: Path, duplicate: str) -> None:
    project = Project.create(tmp_path / "project")
    data = _config(project.root)
    records = [
        {"text_id": "first", "filename": "first.txt", "metadata": {}},
        {"text_id": "second", "filename": "second.txt", "metadata": {}},
    ]
    records[1][duplicate] = records[0][duplicate].upper() if duplicate == "filename" else records[0][duplicate]
    data["texts"] = records
    atomic_write_json(project.root / CONFIG_FILENAME, data)
    with pytest.raises(ValueError):
        Project.open(project.root)


@pytest.mark.parametrize("invalid", [{}, {"schema_version": 99}, {"schema_version": True}, []])
def test_open_rejects_invalid_schema(tmp_path: Path, invalid: object) -> None:
    root = tmp_path / "project"
    root.mkdir()
    atomic_write_json(root / CONFIG_FILENAME, invalid)
    with pytest.raises(ValueError):
        Project.open(root)


def test_open_rejects_symlinked_managed_directory(tmp_path: Path) -> None:
    project = Project.create(tmp_path / "project")
    project.raw_dir.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    project.raw_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        Project.open(project.root)


def test_niqqud_removal_preserves_hebrew_letters_punctuation_and_nonhebrew_marks() -> None:
    text = "שָׁלוֹם, עוֹלָם! café a\u0301"
    assert normalize_hebrew(text) == "שלום, עולם! café á"
    assert normalize_hebrew("שָׁלוֹם", strip_niqqud=False) == "שָׁלוֹם"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1255"])
def test_hebrew_reader_supports_utf8_bom_and_cp1255(tmp_path: Path, encoding: str) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes("שלום\r\nעולם".encode(encoding))
    assert read_hebrew_text(source) == "שלום\nעולם"


def test_malformed_bom_marked_utf8_raises(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(codecs.BOM_UTF8 + b"\xff")
    with pytest.raises(UnicodeDecodeError):
        read_hebrew_text(source)


def test_atomic_json_failure_preserves_existing_config(tmp_path: Path) -> None:
    destination = tmp_path / "data.json"
    atomic_write_json(destination, {"name": "Original"})
    with pytest.raises(ValueError):
        atomic_write_json(destination, {"bad": float("nan")})
    assert json.loads(destination.read_text(encoding="utf-8")) == {"name": "Original"}
