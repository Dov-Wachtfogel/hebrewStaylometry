"""Sefaria library browsing and edition-aware downloads, in logical Unicode."""

from __future__ import annotations

from PySide6.QtCore import QAbstractListModel, QModelIndex, QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListView, QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget

from app.services.sefaria import Book, BookDetails, DownloadRequest
from app.utils.hebrew import normalize_hebrew


class BookListModel(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.books: list[Book] = []
        self.searchable: list[str] = []

    def set_books(self, books):
        self.beginResetModel()
        self.books = list(books)
        self.searchable = [normalize_hebrew(" ".join((book.title, book.he_title, book.author, *book.categories))).casefold() for book in self.books]
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.books)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        book = self.books[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return f"{book.title}  ·  {book.he_title}" if book.he_title else book.title
        if role == Qt.ItemDataRole.ToolTipRole:
            return "\n".join(filter(None, (book.title, book.he_title, " / ".join(book.categories), book.author)))
        if role == Qt.ItemDataRole.UserRole:
            return book
        return None


class BookFilter(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.terms = []

    def search(self, query):
        modern_api = hasattr(self, "beginFilterChange") and hasattr(self, "endFilterChange")
        if modern_api:
            self.beginFilterChange()
        self.terms = normalize_hebrew(query).casefold().split()
        if modern_api:
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)
        else:
            self.invalidateFilter()

    def filterAcceptsRow(self, source_row, parent):
        return all(term in self.sourceModel().searchable[source_row] for term in self.terms)


class SefariaView(QWidget):
    catalog_requested = Signal(bool)
    editions_requested = Signal(object)
    download_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_available = False
        self._busy = False
        self._catalog_loaded = False
        self.details: BookDetails | None = None
        layout = QVBoxLayout(self)
        intro = QLabel("Download Hebrew books from Sefaria directly into your project. Search by English or Hebrew title, choose an edition, and import the full book or a reference.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        top = QHBoxLayout()
        self.load_button = QPushButton("Load library")
        self.load_button.clicked.connect(lambda: self.catalog_requested.emit(False))
        self.refresh_button = QPushButton("Refresh library")
        self.refresh_button.clicked.connect(lambda: self.catalog_requested.emit(True))
        top.addWidget(self.load_button)
        top.addWidget(self.refresh_button)
        top.addStretch()
        layout.addLayout(top)
        search_row = QHBoxLayout()
        self.search_label = QLabel("Book search (Hebrew / English)")
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search titles, authors, or categories in Hebrew / English…")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Book search in Hebrew or English")
        self.search_label.setBuddy(self.search)
        self.search.returnPressed.connect(self._search_entered)
        search_row.addWidget(self.search_label)
        search_row.addWidget(self.search, 1)
        layout.addLayout(search_row)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.model = BookListModel(self)
        self.proxy = BookFilter(self)
        self.proxy.setSourceModel(self.model)
        self.books = QListView()
        self.books.setModel(self.proxy)
        self.books.setUniformItemSizes(True)
        self.books.selectionModel().currentChanged.connect(self._selection_changed)
        self.books.doubleClicked.connect(lambda index: self._request_editions())
        self.search.textChanged.connect(self._search)
        splitter.addWidget(self.books)
        self.controls = QWidget()
        form = QFormLayout(self.controls)
        self.book_title = QLabel("Select a book from the library")
        self.book_title.setTextFormat(Qt.TextFormat.PlainText)
        self.book_title.setWordWrap(True)
        self.editions_button = QPushButton("Load Hebrew editions")
        self.editions_button.clicked.connect(self._request_editions)
        self.edition = QComboBox()
        self.edition.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.edition.setMinimumContentsLength(30)
        self.edition.currentIndexChanged.connect(self._edition_changed)
        self.edition_info = QLabel()
        self.edition_info.setTextFormat(Qt.TextFormat.PlainText)
        self.edition_info.setWordWrap(True)
        self.reference = QLineEdit()
        self.reference.setPlaceholderText("For example: Genesis 1:1-2:3")
        self.all_chapters = QCheckBox("Download all chapters of this book")
        self.all_chapters.setChecked(True)
        self.all_chapters.toggled.connect(self._scope_changed)
        self.scope_hint = QLabel()
        self.scope_hint.setWordWrap(True)
        self.split_sections = QCheckBox("Import chapters/sections as separate texts")
        self.strip_niqqud = QCheckBox("Remove niqqud and cantillation in downloaded text")
        self.download_button = QPushButton("Download into project")
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self._download)
        self.project_hint = QLabel("Open or create a project to download texts.")
        self.project_hint.setWordWrap(True)
        form.addRow(self.book_title)
        form.addRow(self.editions_button)
        form.addRow("Hebrew edition", self.edition)
        form.addRow(self.edition_info)
        form.addRow(self.all_chapters)
        form.addRow("Chapter or passage", self.reference)
        form.addRow(self.scope_hint)
        form.addRow(self.split_sections)
        form.addRow(self.strip_niqqud)
        form.addRow(self.download_button)
        form.addRow(self.project_hint)
        note = QLabel("Source links and edition/license metadata are saved with each text. Downloads contain clean text without HTML or footnotes. After importing, use Preprocessing to create Dicta tokens.")
        note.setWordWrap(True)
        form.addRow(note)
        splitter.addWidget(self.controls)
        splitter.setSizes([700, 550])
        layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.count = QLabel("Library not loaded")
        self.cancel_button = QPushButton("Cancel current task")
        self.cancel_button.clicked.connect(self.cancel_requested)
        footer.addWidget(self.count)
        footer.addStretch()
        footer.addWidget(self.cancel_button)
        layout.addLayout(footer)
        self.status = QLabel("Click Load library to browse available books.")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.reference.textChanged.connect(self._update_enabled)
        self.set_busy(False)

    def selected_book(self):
        index = self.books.currentIndex()
        return self.proxy.data(index, Qt.ItemDataRole.UserRole) if index.isValid() else None

    def set_catalog(self, books):
        self._catalog_loaded = True
        self.model.set_books(books)
        self._update_count()
        self._selection_changed(QModelIndex(), QModelIndex())
        if self.proxy.rowCount():
            self.books.setCurrentIndex(self.proxy.index(0, 0))

    def _search(self, text):
        self.proxy.search(text)
        self._update_count()
        if not self.books.currentIndex().isValid():
            if self.proxy.rowCount():
                self.books.setCurrentIndex(self.proxy.index(0, 0))
            else:
                # Filtering may invalidate the current index without notifying
                # the selection model; discard editions for the previous book.
                self._selection_changed(QModelIndex(), QModelIndex())

    def _search_entered(self):
        if self._busy:
            return
        if not self._catalog_loaded:
            self.catalog_requested.emit(False)
            return
        if self.selected_book() is None and self.proxy.rowCount():
            self.books.setCurrentIndex(self.proxy.index(0, 0))
        self._request_editions()

    def _update_count(self):
        self.count.setText(f"{self.proxy.rowCount():,} of {len(self.model.books):,} books" if self._catalog_loaded else "Library not loaded")

    def _selection_changed(self, current, previous):
        self.details = None
        self.edition.clear()
        book = self.selected_book()
        self.book_title.setText("\n".join(filter(None, (book.title, book.he_title, " / ".join(book.categories)))) if book else "Select a book from the library")
        self.reference.setText(book.title if book and self.all_chapters.isChecked() else "")
        self._update_enabled()

    def _request_editions(self):
        book = self.selected_book()
        if book and not self._busy:
            self.editions_requested.emit(book)

    def set_details(self, details):
        selected = self.selected_book()
        if selected is None or details.book.title != selected.title:
            return
        self.details = details
        self.edition.clear()
        for edition in details.editions:
            label = edition.version_title
            if edition.he_title:
                label += " · " + edition.he_title
            self.edition.addItem(label, edition)
        if not details.editions:
            self.edition_info.setText("No Hebrew editions are available for this book.")
        self._update_enabled()

    def _edition_changed(self, index):
        edition = self.edition.currentData()
        self.edition_info.setText(f"License: {edition.license or 'Unspecified'}\nSource: {edition.source or 'Sefaria'}" if edition else "")
        self._update_enabled()

    def _scope_changed(self, whole_book):
        book = self.selected_book()
        if not whole_book and book:
            current = normalize_hebrew(self.reference.text().strip()).casefold()
            titles = {normalize_hebrew(title).casefold() for title in (book.title, book.he_title) if title}
            if current in titles:
                self.reference.clear()
        self._update_enabled()

    def _download(self):
        if self.download_button.isEnabled():
            self.download_requested.emit(DownloadRequest(
                book=self.details.book, edition=self.edition.currentData(),
                reference=self.details.book.title if self.all_chapters.isChecked() else self.reference.text().strip(),
                split_sections=self.split_sections.isChecked(),
                strip_niqqud=self.strip_niqqud.isChecked(),
            ))

    def set_project_available(self, available):
        self._project_available = available
        self.project_hint.setText("Downloads will be imported into the current project." if available else "Open or create a project to download texts.")
        self._update_enabled()

    def set_busy(self, busy):
        self._busy = busy
        self.load_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.search.setEnabled(not busy)
        self.books.setEnabled(not busy)
        self.controls.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self._update_enabled()

    def _update_enabled(self):
        whole_book = self.all_chapters.isChecked()
        self.reference.setEnabled(not self._busy and not whole_book)
        self.download_button.setText("Download all chapters" if whole_book else "Download selected passage")
        self.scope_hint.setText(
            "Download the complete book in one task. Choose separate texts below to import each chapter/section individually."
            if whole_book else "Enter a chapter or passage above. Choose separate texts below to split the selected passage by section."
        )
        self.editions_button.setEnabled(not self._busy and self.selected_book() is not None)
        selected = self.selected_book()
        matching = selected is not None and self.details is not None and selected.title == self.details.book.title
        reference_available = whole_book or bool(self.reference.text().strip())
        self.download_button.setEnabled(not self._busy and self._project_available and matching and self.edition.currentData() is not None and reference_available)
