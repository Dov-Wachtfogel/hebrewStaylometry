"""Exercise library selection and the actual threaded project import workflow."""

from threading import Event

import pandas as pd
from PySide6.QtCore import Qt

from app.gui.main_window import MainWindow
from app.gui.sefaria import BookListModel, SefariaView
from app.nlp.cache import save_cache
from app.project import Project
from app.services.sefaria import Book, BookDetails, DownloadBundle, DownloadedText, Edition, SefariaCancelled, SefariaError


BOOKS = [Book("Genesis", "בראשית", ("Tanakh",)), Book("Jonah", "יונה", ("Tanakh", "Prophets"))]
EDITION = Edition("Test Hebrew edition", "מהדורה עברית", "https://example.org/source", "CC-BY-SA")


def details(book, editions=None):
    enriched = Book(book.title, book.he_title, book.categories, "Additional author metadata")
    return BookDetails(enriched, [EDITION] if editions is None else editions, {"title": book.title})


def select_book(view, query):
    view.search.setText(query)
    view.books.setCurrentIndex(view.proxy.index(0, 0))


def test_hebrew_search_edition_selection_and_download_options(qtbot):
    view = SefariaView()
    qtbot.addWidget(view)
    view.set_catalog(BOOKS)
    select_book(view, "יוֹנָה")
    assert view.proxy.rowCount() == 1
    assert view.selected_book().title == "Jonah"
    assert view.selected_book().he_title in view.proxy.data(view.proxy.index(0, 0))
    view.set_details(details(BOOKS[1]))
    assert not view.download_button.isEnabled()
    view.set_project_available(True)
    assert view.download_button.isEnabled()
    assert "CC-BY-SA" in view.edition_info.text()
    view.all_chapters.setChecked(False)
    assert view.reference.isEnabled()
    assert view.download_button.text() == "Download selected passage"
    view.reference.setText("Jonah 2:1-3")
    view.split_sections.setChecked(True)
    view.strip_niqqud.setChecked(True)
    with qtbot.waitSignal(view.download_requested) as signal:
        view.download_button.click()
    request = signal.args[0]
    assert request.book.title == "Jonah"
    assert request.edition.version_title == EDITION.version_title
    assert request.reference == "Jonah 2:1-3"
    assert request.split_sections and request.strip_niqqud
    view.set_busy(True)
    assert not view.search.isEnabled()
    assert not view.download_button.isEnabled()
    assert view.cancel_button.isEnabled()


def test_complete_book_is_default_and_ignores_a_previous_passage(qtbot):
    view = SefariaView()
    qtbot.addWidget(view)
    view.set_catalog(BOOKS)
    view.search.setText("Jonah")
    view.set_details(details(BOOKS[1]))
    view.set_project_available(True)
    assert view.all_chapters.isChecked()
    assert not view.reference.isEnabled()
    assert view.download_button.text() == "Download all chapters"
    view.reference.setText("Jonah 2:1-3")
    with qtbot.waitSignal(view.download_requested) as signal:
        view.download_button.click()
    request = signal.args[0]
    assert request.reference == "Jonah"
    assert not request.split_sections
    view.split_sections.setChecked(True)
    with qtbot.waitSignal(view.download_requested) as signal:
        view.download_button.click()
    assert signal.args[0].reference == "Jonah"
    assert signal.args[0].split_sections
    view.all_chapters.setChecked(False)
    assert view.reference.isEnabled()
    assert view.reference.text() == "Jonah 2:1-3"
    with qtbot.waitSignal(view.download_requested) as signal:
        view.download_button.click()
    assert signal.args[0].reference == "Jonah 2:1-3"


def test_passage_mode_clears_book_title_and_rejects_blank_reference(qtbot):
    view = SefariaView()
    qtbot.addWidget(view)
    view.set_catalog(BOOKS)
    view.search.setText("Jonah")
    view.set_details(details(BOOKS[1]))
    view.set_project_available(True)
    requests = []
    view.download_requested.connect(requests.append)
    assert view.reference.text() == "Jonah"
    view.all_chapters.setChecked(False)
    assert not view.reference.text()
    assert not view.download_button.isEnabled()
    view._download()
    assert requests == []
    view.reference.setText("   ")
    assert not view.download_button.isEnabled()
    view._download()
    assert requests == []
    view.reference.setText("Jonah 3")
    assert view.download_button.isEnabled()
    view._download()
    assert requests[0].reference == "Jonah 3"
    view.all_chapters.setChecked(True)
    view.all_chapters.setChecked(False)
    assert view.reference.text() == "Jonah 3"
    view.all_chapters.setChecked(True)
    view.reference.setText("יונה")
    view.all_chapters.setChecked(False)
    assert not view.reference.text()
    assert not view.download_button.isEnabled()
    view.search.setText("Genesis")
    view.set_details(details(BOOKS[0]))
    assert not view.reference.text()
    assert not view.download_button.isEnabled()


def test_search_selects_filtered_book_and_clears_stale_details(qtbot):
    view = SefariaView()
    qtbot.addWidget(view)
    view.set_catalog(BOOKS)
    view.set_project_available(True)
    view.set_details(details(BOOKS[0]))
    assert view.download_button.isEnabled()
    assert view.search.isClearButtonEnabled()
    assert "Hebrew" in view.search_label.text()
    view.search.setText("יוֹנָה")
    assert view.selected_book().title == "Jonah"
    assert view.details is None
    assert not view.download_button.isEnabled()
    with qtbot.waitSignal(view.editions_requested) as signal:
        qtbot.keyPress(view.search, Qt.Key.Key_Return)
    assert signal.args[0].title == "Jonah"
    view.set_details(details(BOOKS[1]))
    assert view.download_button.isEnabled()
    view.search.setText("No matching book")
    assert view.proxy.rowCount() == 0
    assert view.selected_book() is None
    assert view.details is None
    assert not view.download_button.isEnabled()
    assert "0 of 2 books" == view.count.text()
    view.search.clear()
    assert view.proxy.rowCount() == 2
    assert view.selected_book() is not None
    view.search.setText("genesis")
    assert view.selected_book().title == "Genesis"


def test_enter_search_before_catalog_requests_library(qtbot):
    view = SefariaView()
    qtbot.addWidget(view)
    view.search.setText("יונה")
    with qtbot.waitSignal(view.catalog_requested) as signal:
        qtbot.keyPress(view.search, Qt.Key.Key_Return)
    assert signal.args == [False]
    view.set_catalog(BOOKS)
    assert view.selected_book().title == "Jonah"
    assert view.proxy.rowCount() == 1


def test_catalog_reset_and_stale_editions_cannot_download(qtbot):
    view = SefariaView()
    qtbot.addWidget(view)
    view.set_project_available(True)
    view.set_catalog(BOOKS)
    select_book(view, "Jonah")
    view.set_details(details(BOOKS[1]))
    assert view.download_button.isEnabled()
    select_book(view, "Genesis")
    view.set_details(details(BOOKS[1]))
    assert view.details is None
    assert not view.download_button.isEnabled()
    view.set_details(details(BOOKS[0], editions=[]))
    assert "No Hebrew editions" in view.edition_info.text()
    view.set_details(details(BOOKS[0]))
    assert view.download_button.isEnabled()
    # A model reset can clear Qt's selection without emitting currentChanged.
    view.set_catalog([])
    assert view.details is None
    assert view.edition.count() == 0
    assert not view.download_button.isEnabled()
    assert not view.reference.text()


def test_catalog_model_accepts_iterators(qtbot):
    model = BookListModel()
    model.set_books(iter(BOOKS))
    assert len(model.searchable) == model.rowCount() == 2


def make_cached_project(tmp_path):
    project = Project.create(tmp_path / "project")
    record = project.import_text_content([("original.txt", "שלום עולם", {"author": "Original author"})])[0]
    frame = pd.DataFrame([[1, "שלום", "שלום", "NOUN", "{}", 1], [2, "עולם", "עולם", "NOUN", "{}", 1]],
                         columns=["token_id", "word", "lemma", "pos_tag", "morph_features", "sentence_id"])
    save_cache(project, record, project.settings["nlp"], frame)
    return project, record


def install_fake_client(monkeypatch, *, fail_download=False):
    from app.services import workers

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def catalog(self, **kwargs):
            calls.append(("catalog", kwargs["refresh"]))
            return BOOKS

        def details(self, book, **kwargs):
            return details(book)

        def download(self, request, **kwargs):
            calls.append(("download", request))
            if fail_download:
                raise SefariaError("Selected edition unavailable")
            kwargs["status"]("Downloading Jonah")
            kwargs["progress"](50)
            return DownloadBundle([
                DownloadedText("יונה 1.txt", "וַיְהִי דְבַר ה׳\n", {"source": "Sefaria", "ref": "Jonah 1", "version_title": EDITION.version_title}),
                DownloadedText("יונה 2.txt", "וַיִּתְפַּלֵּל יוֹנָה\n", {"source": "Sefaria", "ref": "Jonah 2", "version_title": EDITION.version_title}),
            ], ["One empty segment was omitted."], 2)

        def close(self):
            pass

    monkeypatch.setattr(workers, "SefariaClient", FakeClient)
    return calls


def open_download_view(window, qtbot):
    window.open_sefaria()
    qtbot.waitUntil(lambda: window._worker is None, timeout=5000)
    select_book(window.sefaria, "Jonah")
    window.sefaria.editions_button.click()
    qtbot.waitUntil(lambda: window._worker is None, timeout=5000)


def test_main_window_download_preserves_caches_and_imports_metadata(qtbot, tmp_path, monkeypatch):
    calls = install_fake_client(monkeypatch)
    project, original = make_cached_project(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    window.sefaria_cache_dir = tmp_path / "cache"
    window.open_project(project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=5000)
    old_frame = window.frames[original.text_id]
    window._check_all(Qt.CheckState.Unchecked)
    open_download_view(window, qtbot)
    assert window.tabs.currentWidget() is window.sefaria
    assert window.sefaria.download_button.isEnabled()
    window.sefaria.split_sections.setChecked(True)
    window.sefaria.download_button.click()
    qtbot.waitUntil(lambda: window._worker is None, timeout=5000)
    saved = Project.open(project.root)
    assert len(saved.texts) == window.explorer.count() == 3
    assert window.frames == {original.text_id: old_frame}
    assert window.project.get_record(original.text_id).metadata["author"] == "Original author"
    assert window.explorer.item(0).checkState() == Qt.CheckState.Unchecked
    assert window.explorer.item(1).checkState() == Qt.CheckState.Checked
    for record in saved.texts[1:]:
        assert record.metadata["source"] == "Sefaria"
        assert record.metadata["version_title"] == EDITION.version_title
        assert (saved.raw_dir / record.filename).read_text(encoding="utf-8").endswith("\n")
        assert record.text_id not in window.frames
    assert window.sefaria.progress.value() == 100
    assert "One empty segment" in window.sefaria.status.text()
    assert "Preprocessing" in window.sefaria.status.text()
    assert calls[0] == ("catalog", False)
    assert calls[1][1].split_sections
    assert calls[1][1].reference == "Jonah"
    window.close()


def test_download_error_recovers_controls_without_project_changes(qtbot, tmp_path, monkeypatch):
    install_fake_client(monkeypatch, fail_download=True)
    project, original = make_cached_project(tmp_path)
    window = MainWindow()
    qtbot.addWidget(window)
    window.open_project(project.root)
    qtbot.waitUntil(lambda: window._worker is None, timeout=5000)
    config_path = project.root / "project_config.json"
    config = config_path.read_bytes()
    open_download_view(window, qtbot)
    window.sefaria.download_button.click()
    qtbot.waitUntil(lambda: window._worker is None, timeout=5000)
    assert config_path.read_bytes() == config
    assert len(window.project.texts) == 1
    assert original.text_id in window.frames
    assert "Selected edition unavailable" in window.sefaria.status.text()
    assert window.sefaria.download_button.isEnabled()
    assert not window.sefaria.cancel_button.isEnabled()
    window.close()


def test_closing_during_library_fetch_cancels_and_waits_for_worker(qtbot, tmp_path, monkeypatch):
    from app.services import workers

    entered, closed = Event(), Event()

    class WaitingClient:
        def __init__(self, **kwargs):
            pass

        def catalog(self, *, cancelled, **kwargs):
            entered.set()
            while not cancelled():
                closed.wait(0.01)
            raise SefariaCancelled()

        def close(self):
            closed.set()

    monkeypatch.setattr(workers, "SefariaClient", WaitingClient)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.sefaria_cache_dir = tmp_path / "cache"
    window.open_sefaria()
    qtbot.waitUntil(entered.is_set, timeout=5000)
    assert window._worker is not None
    window.close()
    assert window._pending_close
    qtbot.waitUntil(lambda: window._worker is None and not window.isVisible(), timeout=5000)
    assert closed.is_set()
