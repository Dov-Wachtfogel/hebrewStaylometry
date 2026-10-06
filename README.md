# Hebrew Stylometry Studio

A local PySide6 desktop application for managing Hebrew corpora, caching Dicta
annotations, and exploring lexical, grammatical, and multivariate stylistic
patterns. The existing `oldProject` notebook is retained as a reference.

## Run

Python 3.10 or newer is required. In this PyCharm project, dependencies are
installed in `.venv`, including CPU PyTorch.

```bash
.venv/bin/python main.py
# Open a saved project and load its validated token caches:
.venv/bin/python main.py --project /path/to/project
# Create an example corpus using the supplied Hebrew texts:
.venv/bin/python scripts/create_example_project.py /path/to/example_project
# The supplied example_project is already processed with the real Dicta model:
.venv/bin/python main.py --project example_project
```

For a fresh environment:

```bash
python -m venv .venv
# Linux/macOS; use .venv\Scripts\python.exe on Windows.
# CPU-only installation avoids downloading CUDA packages:
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
```

GPU users can install the appropriate PyTorch build from the
[official installation selector](https://pytorch.org/get-started/locally/).
Select `cuda` or `mps` in Preprocessing if the installed build and hardware
support it. A CUDA selection with unavailable hardware produces a task error.

1. Choose **New project** and select a project folder, or open an existing one.
2. Import `.txt` files, or choose **Download from Sefaria** to browse Hebrew
   books. UTF-8, UTF-8 with BOM, and Windows Hebrew CP1255 files are supported.
   The original file bytes are copied into the project.
3. Edit metadata cells or import `metadata.json`.
4. Open **Preprocessing**, choose the Dicta model/options, and click
   **Process corpus**. Logs and a nonmodal progress dialog show progress.
5. Check the desired texts in the explorer. Open **Analysis**, choose the
   analysis and parameters, then run it. Inspect individual tokens in **Tokens**.
6. Export filtered/sorted tables to CSV/XLSX, charts to PNG/SVG/PDF, or a report
   bundle containing CSV, PNG, settings/metadata JSON, and Markdown.

The first NLP run downloads the model and its custom Python code from Hugging
Face. Subsequent runs use the Hugging Face model cache. The launcher reuses the
local `.model_cache` when present; `--model-cache /path/to/cache` selects another
folder. Opening a project with
valid token caches does not load or download any NLP model. Model loading and
inference happen entirely in a `QThread`. Cancellation is cooperative between
bounded batches; an ongoing model download/load finishes before cancellation.
Closing the window during a job requests cancellation and closes once the
worker has stopped, avoiding `QThread: Destroyed while thread is still running`.

## Package layout

```text
main.py                         PyCharm-friendly entry point
requirements.txt                runtime dependencies
requirements-dev.txt            runtime + testing dependencies
pyproject.toml                  installable package/CLI and test configuration
app/
  __main__.py                   application bootstrap
  project.py                    validated project/text metadata and settings
  gui/
    main_window.py              project, preprocessing, analysis, token views
    dashboard.py                tabbed analysis parameter controls
    models.py                   editable metadata and pandas Qt models
    tables.py                   sorting/filtering and visible-row exports
    visualization.py            embedded Matplotlib and Hebrew chart labels
    sefaria.py                   Hebrew/English library search and edition selection
    workers.py                  background analysis jobs
  services/
    sefaria.py                   Sefaria API, catalog cache and clean text retrieval
    workers.py                  background catalog, editions and atomic downloads
  nlp/
    pipeline.py                 Dicta adapter, sentence/chunk and UD/JSON parsing
    cache.py                    source/settings fingerprints and CSV integrity
    workers.py                  cache loading and preprocessing QThreads
  analysis/
    common.py                   validated corpora and frequency/feature matrices
    lexical.py                  MFW, POS, diversity, n-grams, length statistics
    multivariate.py             PCA, Burrows' Delta, HCA, t-SNE
    rolling.py                  independently fitted rolling Delta
    engine.py                   extensible analysis dispatcher
  utils/
    hebrew.py                   Unicode normalization and plotting-only bidi
    io.py                       atomic JSON/CSV, encoding, CSV/XLSX exports
examples/
  texts/                        original short Hebrew example texts
  metadata.json                 optional metadata for the example files
example_project/                example corpus with real cached Dicta annotations
scripts/
  create_example_project.py     create a corpus, optionally process via QThread
  smoke_dicta.py                optional real pretrained-model integration check
  smoke_long_dicta.py           diagnose/verify long text with the cached model
tests/                          numerical, persistence, NLP, QThread, GUI checks
oldProject/                     original reference notebook (unchanged)
```

Each user project is independent of the application source:

```text
my_project/
  project_config.json
  raw_texts/
    Author_Title_Date.txt
  processed_data/
    Author_Title_Date_tokens.csv
    Author_Title_Date_tokens.meta.json
  outputs/
    report_YYYYMMDD_HHMMSS_microseconds/
      results.csv
      chart.png
      analysis_settings.json
      report.md
```

The supplied short example texts and their author/date labels are illustrative,
not historical sources. Their caches were generated by the real Dicta model.

`project_config.json` contains `schema_version`, `name`, `texts` (stable IDs,
filenames and metadata), and `settings.nlp`/`settings.analysis`. Files with
colliding names are imported under a unique suffix; originals are preserved.
Managed filenames and project structure are validated when opening a project.
Metadata changes and config/cache writes use atomic file replacement.

## Download books from Sefaria

1. Open or create a project, then choose **Download from Sefaria** in the
   toolbar, or open the **Sefaria** tab and click **Load library**.
2. Use the **Book search (Hebrew / English)** field to search titles, authors
   or categories. Select a book
   and click **Load Hebrew editions**. Choose the edition you want; its source
   and license are shown before downloading. Pressing Enter in the search field
   also loads the selected book's editions.
3. Leave **Download all chapters of this book** checked to retrieve the complete
   book in one task. Uncheck it to enter a reference such as `Genesis 1:1-2:3`
   in **Chapter or passage**.
   Choose **Import chapters/sections as separate texts** to create independent
   documents for analysis; otherwise the selected text becomes one document.
4. Click **Download all chapters** or **Download selected passage**. Open **Preprocessing** to generate Dicta
   annotations, then analyze the downloaded texts like any other corpus.

The integration uses Sefaria's public
[library catalog](https://developers.sefaria.org/reference/get-index) and
[edition-aware text API](https://developers.sefaria.org/reference/get-v3-texts).
No account or API key is needed. Catalogs are cached in the operating system's
application cache for seven days; **Refresh library** fetches a fresh copy.
Network requests and imports run in a `QThread`, with progress, status and
cooperative cancellation. Requests have timeouts and bounded retries.

Downloads preserve the chosen Hebrew edition and do not fill gaps from other
editions. Whole-book retrieval includes the book's schema sections, including
separate introductions when present. HTML and footnote markup are removed;
niqqud and cantillation are kept unless the removal option is checked. Source
links, reference, edition and license are saved as editable metadata. Existing
project texts are preserved, and filename collisions get unique suffixes.
An unsuccessful or cancelled fetch imports no partial batch. Cancellation
during the final disk commit lets that atomic import finish coherently.

## Metadata

`Author_Title_Date.txt` populates author/title/date. Underscores inside the title
are supported; the first segment is the author and the last is the date.
Default editable fields are author, title, genre, period, date and chapter.
Imported additional scalar fields appear as extra table columns.

The external JSON can map project filenames to metadata:

```json
{
  "Author_Title_1900.txt": {
    "author": "Author",
    "genre": "prose",
    "period": "modern",
    "date": "1900",
    "chapter": "1"
  }
}
```

It can also be a list of records containing `filename`, with fields flat or
nested under `metadata`. Unmatched filenames are ignored; the GUI reports how
many texts were updated. Import validates the whole batch before committing
updates.

## NLP and caching

The default model is `dicta-il/dictabert-tiny-joint`; the full
`dicta-il/dictabert-joint` model is also selectable. Integration follows
[Dicta's joint model card](https://huggingface.co/dicta-il/dictabert-joint)
and its
[published prediction implementation](https://huggingface.co/dicta-il/dictabert-joint/blob/main/BertForJointParsing.py).
Transformers is constrained to a compatible 4.x range in `requirements.txt`.

Segmented mode requests CoNLL-U output and retains individual function words,
including Hebrew prefixes and reconstructed possessive morphemes. Structural
multiword range rows and empty nodes are skipped. Surface mode retains each
original word, with prefix/suffix morphology in `morph_features` JSON. The
required prefix, morphology, lemma, and syntax heads are enabled for UD output;
surface JSON mode avoids the syntax head. NER is not loaded.
The checkpoint's unused `classifier.weight` and `classifier.bias` notice refers
to that disabled NER head, and does not indicate missing POS or lemma weights.

Dicta can occasionally segment a standalone Hebrew reference letter into a
prefix plus an empty stem. The adapter recognizes this specific invalid
segmentation using the original surface form from the CoNLL-U multiword row,
and keeps one source token with the model's stem POS and lemma. The original
segments and the recovery reason are retained in `morph_features`. Other
missing forms or POS tags remain explicit errors with sentence/token details.

Every cache has these columns:

```text
token_id,word,lemma,pos_tag,morph_features,sentence_id
```

Token and sentence IDs are positive integers; token IDs are consecutive across
the document. `morph_features` is a JSON string. Sentence splitting uses line
breaks and sentence-final punctuation. It is a heuristic and can split
abbreviations. Long sentences are split into tokenizer-aware input chunks
without silent truncation; their original sentence ID is retained. Chunking
reduces the model's context and should be considered for long passages.

`split_hebrew_text()` preserves sentence punctuation and closing quotes while
splitting sentences and paragraphs. Fast-tokenizer subword offsets are scanned
once for long sentences; only bounded chunk candidates are reencoded to verify
the model limit. Chunk order, sentence IDs, and all words are retained.

New projects default to **32 chunks per batch**. Set **Batch size** to 64 in
Preprocessing when memory permits, or lower it for a smaller memory budget.
Existing projects retain their saved batch size. Dicta's `predict()` evaluates
the supplied list as one tensor batch; it has no `batch_size` keyword, so the
wrapper explicitly slices that list. Prediction runs under
`torch.inference_mode()` after `model.eval()`.

The sidecar records SHA-256 fingerprints of source bytes, normalized NLP
settings, and cache schema, plus a digest of the CSV itself. A source edit,
changed preprocessing setting, corrupt CSV, or interrupted write invalidates
the cache. A source edit during inference prevents stale tokens being stamped
as current. CSVs without provenance sidecars are retained but must be
reprocessed. Pin the model **Revision** to a Hugging Face commit SHA for
reproducible runs; a moving `main` revision cannot promise identical model
weights across machines or dates.

The worker hashes the same bytes it decodes and rechecks the source once before
saving. It accumulates token records in memory, builds one DataFrame, and writes
one atomic bulk CSV per completed document. Cache schema 2 invalidates older
token caches so the updated sentence splitting is applied on reprocessing.

Worker lifecycle signals are Qt's `started` and `finished`, plus
`progress(int)`, `status(str)`, `document_ready(str, object)`,
`document_error(str, str)`, `completed(object)`, `error(str)`, and `cancelled()`.
The GUI owns worker references until `finished`. Individual document failures
are logged and batch processing continues; model loading failures terminate
the task. Completed document caches survive cancellation or a later failure.
`DictaWorker` handles processing (`ProcessingWorker` remains a compatibility
alias). Progress signals are monotonic and emitted at most once per integer
percentage for the whole job. Token tables are refreshed at job completion;
reading, inference, cache validation, and CSV writing stay in the worker thread.

## Analysis definitions

| Analysis | Behavior |
| --- | --- |
| MFW | Corpus top-N words/lemmas, token/document counts and relative frequencies; configurable stop words |
| POS distribution | Document × POS proportions, including punctuation |
| Lexical diversity | TTR, MATTR, hapax/dis counts and type proportions |
| PCA 2D/3D | Standardized MFW or POS features, explained variance, loadings; zoom/pan and 3D rotation |
| Burrows' Delta | Average absolute differences in sample z-scores (`ddof=1`); constant features excluded |
| HCA | Delta or Euclidean distances; average, complete, single or Ward linkage; Ward requires Euclidean |
| t-SNE | Reproducible seed; perplexity adjusted to corpus size; exploratory scatter |
| Word/POS n-grams | Corpus counts/proportions; n-grams never span sentences |
| Sentence/word lengths | Histograms, means, standard deviations and medians |
| Rolling Delta | Full sliding windows in one target against independent reference documents |

Lexical analyses normalize Unicode, casefold Latin letters, remove Hebrew
combining marks, exclude punctuation by default, and use lemma fallback to word.
Selected feature frequencies are divided by the total eligible token count,
not only counts in the selected top-N vocabulary. Length statistics retain
lexical words regardless of the stop-word list. With segmented NLP output,
lengths measure morpheme tokens; use surface mode to measure surface words.

MATTR uses full windows, or the whole text as one window for a shorter document.
PCA requires at least three documents for 2D or four for 3D, and enough
independent feature variation. The dashboard explains these requirements when
a selected corpus cannot support PCA. Downloading a whole book into one file
creates one document; importing its chapters separately creates multiple
documents for a chapter comparison. Delta requires at least two varied documents.
Rolling Delta requires a target plus at least two separate references. Its
vocabulary, mean and sample standard deviation are fitted only on the references
to avoid target leakage. Windows use eligible lexical token offsets; result
`start`/`end` map back to the original token table (zero-based, end exclusive).
Partial trailing windows are omitted. Distance changes are stylistic evidence
to investigate, not an automatic authorship verdict.

The GUI uses Matplotlib's `FigureCanvasQTAgg` and navigation toolbar. Qt receives
logical Unicode Hebrew unchanged. Matplotlib 3.11+ also receives logical text
because its renderer handles bidi natively; older versions use
`bidi.algorithm.get_display` at the chart boundary. This avoids double reversal
on newer renderers. DejaVu Sans supplies Hebrew glyphs. Hebrew does not require
Arabic letter reshaping. CSV/XLSX contain logical Hebrew text.

Scatter plots display short document references/titles, removing filename hashes,
edition suffixes, and `.txt`. **Show labels** toggles canvas text; the label mode
offers titles, numbers, or hover only. Auto uses titles for up to 15 points and a
numbered document list for larger plots. `adjustText` separates 2D labels with
subtle leader lines and reapplies layout after resizing. Dense plots and PCA 3D
use hover and the document list; selecting a document highlights its point.
Hover tooltips retain the complete filename and metadata.

**Scatter color by** offers auto, book, author, period, and other metadata fields.
Auto selects populated metadata; an entirely missing requested field falls back
to a populated one, including book names inferred from section references.
Changing the color group updates the existing scatter without rerunning analysis.
Analysis tables and exported data retain original filenames.

## Extend and verify

`run_analysis(corpus, kind, params)` returns an `AnalysisResult` with a DataFrame,
rendering payload and interpretation text. Register new analyses in
`app/analysis/engine.py`; render their payload in `gui/visualization.py`.
Calculation modules depend on pandas/scientific libraries and have no Qt or
filesystem responsibilities.

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
# Optional: downloads and runs the actual tiny model, outside the unit suite.
.venv/bin/python scripts/smoke_dicta.py --output /tmp/dicta_tokens.csv
# Optional: verify a long source against the cached model without changing it.
HF_HOME="$PWD/.model_cache" .venv/bin/python scripts/smoke_long_dicta.py /path/to/book.txt --verify --batch-size 8
```

The test suite covers numeric definitions, empty/small corpora, sentence
boundaries, reference-only fitting, source/settings/cache corruption, atomic
project persistence, real QThread lifecycle/cancellation, editable/filterable
tables, all chart types, and report exports. Default tests use small synthetic
token fixtures or fake model predictions; the separate smoke script verifies
the live pretrained model.

Dicta models have their own attribution and license requirements. Refer to
the linked model card and cite its paper when publishing results.
