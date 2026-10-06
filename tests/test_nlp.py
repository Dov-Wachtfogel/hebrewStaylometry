"""Deterministic adapter/cache tests; never download model weights."""

import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app.nlp.cache import cache_paths, load_cache, save_cache, validate_token_frame
from app.nlp.pipeline import (
    DictaPipeline, ProcessingCancelled, TOKEN_COLUMNS, empty_token_frame,
    parse_json_sentence, parse_ud_sentence, split_hebrew_text,
)
from app.project import Project, TextRecord


# Exact cached Dicta tiny-joint prediction for כ״א, including its malformed
# all-consuming prefix segmentation of the reference letter כ.
EMPTY_PREFIX_UD = [
    "# sent_id = 1", "# text = כ״א",
    "1-2\tכ\t_\t_\t_\t_\t_\t_\t_\t_",
    "1\tכ\tכ\tADP\tADP\t_\t2\tcase\t_\t_",
    "2\t\t\tADV\tADV\t\t0\troot\t_\t_",
    "3\t״\t״\tPUNCT\tPUNCT\t\t2\tpunct\t_\t_",
    "4\tא\tא\tADJ\tADJ\t\t2\tfixed\t_\t_",
]


class WordTokenizer:
    model_max_length = 6

    def encode(self, text, add_special_tokens=True, truncation=False, verbose=True):
        assert truncation is False
        return [0] * (len(text.split()) + (2 if add_special_tokens else 0))


class FakeJointModel:
    config = SimpleNamespace(max_position_embeddings=6)

    def __init__(self):
        self.calls = []

    def predict(self, sentences, tokenizer, **kwargs):
        self.calls.append((sentences, kwargs))
        assert all(len(tokenizer.encode(sentence)) <= 6 for sentence in sentences)
        if kwargs["output_style"] == "ud":
            return [[f"{i}\t{word}\t{word}\tNOUN\tNOUN\t_\t0\troot\t_\t_"
                     for i, word in enumerate(sentence.split(), 1)] for sentence in sentences]
        return [{"tokens": [{"token": word, "lex": word, "morph": {
            "pos": "NOUN", "feats": {}, "prefixes": ["ADP"], "suffix": "PRON",
            "suffix_feats": {"Person": "3"},
        }} for word in sentence.split()]} for sentence in sentences]


def fake_pipeline(**settings):
    pipeline = DictaPipeline({"batch_size": 2, **settings})
    pipeline.tokenizer = WordTokenizer()
    pipeline.model = FakeJointModel()
    return pipeline


def make_project(tmp_path):
    project = Project.create(tmp_path / "corpus")
    record = TextRecord("doc1", "מחבר_ספר_1900.txt")
    project.texts.append(record)
    (project.raw_dir / record.filename).write_text("שלום עולם", encoding="utf-8")
    project.save()
    return project, record


def token_frame():
    return pd.DataFrame([
        [1, "שלום", "שלום", "NOUN", "{}", 1],
        [2, "NA", "", "X", "", 1],
    ], columns=TOKEN_COLUMNS)


def test_ud_keeps_prefixes_and_suffixes_ignores_structural_rows():
    result = [
        "# sent_id = 1", "# text = ובספרו",
        "1-5\tובספרו\t_\t_\t_\t_\t_\t_\t_\t_",
        "1\tו\tו\tCCONJ\tCCONJ\t_\t3\tcc\t_\t_",
        "2\tב\tב\tADP\tADP\t_\t3\tcase\t_\t_",
        "3\tספר_\tספר\tNOUN\tNOUN\tGender=Masc|Number=Sing\t0\troot\t_\t_",
        "3.1\tה\tה\tDET\tDET\t_\t3\tdet\t_\t_",
        "4\t_של_\tשל\tADP\tADP\t_\t5\tcase\t_\t_",
        "5\t_הוא\tהוא\tPRON\tPRON\tPerson=3\t3\tnmod:poss\t_\t_",
    ]
    rows = parse_ud_sentence(result, sentence_id=7)
    assert [row["word"] for row in rows] == ["ו", "ב", "ספר", "של", "הוא"]
    assert [row["pos_tag"] for row in rows] == ["CCONJ", "ADP", "NOUN", "ADP", "PRON"]
    assert {row["sentence_id"] for row in rows} == {7}
    assert json.loads(rows[2]["morph_features"])["Gender"] == "Masc"
    assert json.loads(rows[2]["morph_features"])["_ud_form"] == "ספר_"


def test_surface_json_retains_attached_morphology_and_blank_lemma_fallback():
    morph = {"pos": "NOUN", "feats": {"Gender": "Masc"}, "prefixes": ["ADP"],
             "suffix": "PRON", "suffix_feats": {"Person": "3"}}
    rows = parse_json_sentence({"tokens": [{"token": "בספרו", "lex": "[BLANK]", "morph": morph}]}, 2)
    assert rows[0]["word"] == rows[0]["lemma"] == "בספרו"
    assert json.loads(rows[0]["morph_features"]) == morph


def test_empty_prefix_stem_uses_source_word_without_duplicating_a_prefix():
    # Exact output of cached Dicta tiny joint for the input כ״א. Its prefix
    # prediction consumes the entire reference letter, leaving no stem.
    rows = parse_ud_sentence(EMPTY_PREFIX_UD, 7)
    assert [row["word"] for row in rows] == ["כ", "״", "א"]
    assert [row["pos_tag"] for row in rows] == ["ADV", "PUNCT", "ADJ"]
    assert rows[0]["lemma"] == "כ"
    assert json.loads(rows[0]["morph_features"]) == {
        "_ud_form": "", "_ud_word_source": "multiword_token",
        "_ud_recovery": "empty_stem_after_prefix",
        "_ud_segments": [
            {"word": "כ", "lemma": "כ", "pos_tag": "ADP", "morph_features": "_"},
            {"word": "", "lemma": "", "pos_tag": "ADV", "morph_features": ""},
        ],
    }


def test_source_backed_recovery_keeps_model_lemma_even_when_unrelated_to_surface():
    result = [
        "249-250\tכ\t_\t_\t_\t_\t_\t_\t_\t_",
        "249\tכ\tכ\tADP\tADP\t_\t250\tcase\t_\t_",
        "250\t\tיא\tPROPN\tPROPN\t\t248\tnmod\t_\t_",
    ]
    rows = parse_ud_sentence(result, 388)
    assert len(rows) == 1
    assert rows[0]["word"] == "כ"
    assert rows[0]["lemma"] == "יא"
    assert rows[0]["pos_tag"] == "PROPN"


@pytest.mark.parametrize("form,lemma", [("_", "_"), ("_", "שלום"), ("_", "[BLANK]"), ("", "שלום")])
def test_ud_missing_words_without_a_supported_reconstruction_are_not_discarded(form, lemma):
    row = f"12\t{form}\t{lemma}\tNOUN\tNOUN\t_\t0\troot\t_\t_"
    with pytest.raises(ValueError, match="no recoverable word.*sentence 7, UD token 12"):
        parse_ud_sentence([row], 7)


def test_ud_missing_pos_remains_an_explicit_error_even_with_a_valid_surface_and_lemma():
    row = "12\tו\tו\t_\t_\t_\t0\troot\t_\t_"
    with pytest.raises(ValueError, match="no POS tag.*sentence 7, UD token 12"):
        parse_ud_sentence([row], 7)


@pytest.mark.parametrize("surface", ["ב", "כמילה", ""])
def test_empty_stem_with_unmatched_prefixes_is_not_reconstructed(surface):
    result = list(EMPTY_PREFIX_UD)
    result[2] = f"1-2\t{surface}\t_\t_\t_\t_\t_\t_\t_\t_"
    with pytest.raises(ValueError, match="no recoverable word"):
        parse_ud_sentence(result, 1)


def test_duplicate_ud_ids_are_not_silently_collapsed_by_source_recovery():
    result = list(EMPTY_PREFIX_UD)
    result.insert(4, result[3])
    with pytest.raises(ValueError, match="duplicate UD token IDs"):
        parse_ud_sentence(result, 1)


def test_recovered_reference_and_long_chunks_keep_consecutive_token_ids():
    pipeline = fake_pipeline()
    regular_predict = pipeline.model.predict

    def predict(sentences, tokenizer, **kwargs):
        results = regular_predict(sentences, tokenizer, **kwargs)
        for index, sentence in enumerate(sentences):
            if sentence == "כ״א":
                results[index] = list(EMPTY_PREFIX_UD)
        return results

    pipeline.model.predict = predict
    words = [f"מילה{i}" for i in range(11)]
    frame = validate_token_frame(pipeline.process("כ״א\n" + " ".join(words)))
    assert frame.word.tolist() == ["כ", "״", "א"] + words
    assert frame.token_id.tolist() == list(range(1, 15))
    assert frame.sentence_id.tolist() == [1] * 3 + [2] * 11
    assert frame.pos_tag.iloc[0] == "ADV"
    assert json.loads(frame.morph_features.iloc[0])["_ud_recovery"] == "empty_stem_after_prefix"


@pytest.mark.parametrize("parser,result", [
    (parse_ud_sentence, ["invalid"]),
    (parse_ud_sentence, ["1\tשלום\tשלום\t_\t_\t_"]),
    (parse_json_sentence, {"tokens": [{"token": "שלום"}]}),
    (parse_json_sentence, {}),
])
def test_bad_model_output_is_not_fake_tagged(parser, result):
    with pytest.raises(ValueError):
        parser(result, 1)


def test_long_sentence_chunks_without_token_loss_or_new_sentence_ids():
    pipeline = fake_pipeline()
    words = [f"מילה{i}" for i in range(11)]
    progress = []
    frame = pipeline.process(" ".join(words) + "\nשלום עולם", progress.append)
    assert frame.columns.tolist() == TOKEN_COLUMNS
    assert frame.word.tolist() == words + ["שלום", "עולם"]
    assert frame.token_id.tolist() == list(range(1, 14))
    assert frame.sentence_id.tolist() == [1] * 11 + [2] * 2
    assert progress[0] == 0 and progress[-1] == 100
    assert progress == sorted(progress)
    for _, kwargs in pipeline.model.calls:
        assert kwargs["truncation"] is False
        assert kwargs["compute_syntax_mst"] is True


def test_chunk_limits_use_subwords_and_suppress_only_length_measurement_warnings():
    class SubwordTokenizer(WordTokenizer):
        def __init__(self):
            self.warnings = []
            self.measurements = []

        def encode(self, text, add_special_tokens=True, truncation=False, verbose=True):
            assert truncation is False
            length = sum(3 if word.startswith("ארוכה") else 1 for word in text.split())
            length += 2 if add_special_tokens else 0
            self.measurements.append((text, length, verbose))
            if verbose and length > self.model_max_length:
                self.warnings.append(text)
            return [0] * length

    pipeline = fake_pipeline()
    pipeline.tokenizer = SubwordTokenizer()
    words = ["שלום", "ארוכהא", "עולם", "ארוכהב", "ארוכהג", "ספר"]
    frame = pipeline.process(" ".join(words))
    assert frame.word.tolist() == words
    assert frame.sentence_id.tolist() == [1] * len(words)
    assert not pipeline.tokenizer.warnings
    assert any(length > 6 and verbose is False
               for _, length, verbose in pipeline.tokenizer.measurements)
    assert all(len(pipeline.tokenizer.encode(chunk)) <= 6
               for batch, _ in pipeline.model.calls for chunk in batch)
    assert len([chunk for batch, _ in pipeline.model.calls for chunk in batch]) == 3


def test_configured_sequence_limit_includes_special_tokens():
    pipeline = fake_pipeline(max_sequence_length=4)
    frame = pipeline.process("שלום עולם ספר חדש האחרון")
    assert frame.word.tolist() == ["שלום", "עולם", "ספר", "חדש", "האחרון"]
    chunks = [chunk for batch, _ in pipeline.model.calls for chunk in batch]
    assert chunks == ["שלום עולם", "ספר חדש", "האחרון"]
    assert all(len(pipeline.tokenizer.encode(chunk)) <= 4 for chunk in chunks)


def test_hebrew_sentence_splitting_keeps_quotes_abbreviations_and_paragraphs():
    text = 'הרמב״ם אמר: "שָׁלוֹם עולם!" עוד משפט׃\r\n\r\nפסקה בלי נקודה\nסיום?'
    assert split_hebrew_text(text) == [
        'הרמב״ם אמר: "שָׁלוֹם עולם!"', "עוד משפט׃", "פסקה בלי נקודה", "סיום?",
    ]
    assert split_hebrew_text(" \n\r\n ") == []


def test_fast_chunking_scans_offsets_once_and_only_reencodes_bounded_strings():
    import re

    class OffsetTokenizer(WordTokenizer):
        is_fast = True

        def __init__(self):
            self.scans = 0
            self.measured_words = 0

        def __call__(self, text, **kwargs):
            self.scans += 1
            assert kwargs["add_special_tokens"] is kwargs["truncation"] is False
            assert kwargs["return_offsets_mapping"] is True
            return {"offset_mapping": [
                (word.start(), word.end())
                for word in re.finditer(r"\S+", text)
                for _ in range(3 if word.group().startswith("ארוכה") else 1)
            ]}

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def encode(self, text, **kwargs):
            words = text.split()
            assert len(words) <= self.model_max_length
            self.measured_words += len(words)
            return [0] * (sum(3 if word.startswith("ארוכה") else 1 for word in words) + 2)

    pipeline = fake_pipeline()
    pipeline.tokenizer = OffsetTokenizer()
    words = ["שלום", "ארוכהא", "עולם", "ארוכהב", "ספר"] * 1000
    chunks = pipeline._chunks(" ".join(words), lambda: False)
    assert [word for chunk in chunks for word in chunk.split()] == words
    assert pipeline.tokenizer.scans == 1
    assert pipeline.tokenizer.measured_words == len(words)
    assert all(len(pipeline.tokenizer.encode(chunk)) <= 6 for chunk in chunks)


@pytest.mark.parametrize("batch_size", [32, 64])
def test_explicit_tensor_batches_run_in_inference_mode_with_bounded_progress(batch_size):
    pipeline = fake_pipeline(batch_size=batch_size)
    active = []

    @contextmanager
    def inference_mode():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    pipeline._torch = SimpleNamespace(inference_mode=inference_mode)
    original_predict = pipeline.model.predict

    def predict(sentences, tokenizer, **kwargs):
        assert active
        assert len(sentences) <= batch_size
        return original_predict(sentences, tokenizer, **kwargs)

    pipeline.model.predict = predict
    progress = []
    count = batch_size * 120 + 3
    frame = pipeline.process("\n".join(["שלום"] * count), progress.append)
    sizes = [len(sentences) for sentences, _ in pipeline.model.calls]
    assert sizes == [batch_size] * 120 + [3]
    assert frame.word.tolist() == ["שלום"] * count
    assert frame.sentence_id.tolist() == list(range(1, count + 1))
    assert progress[0] == 0 and progress[-1] == 100
    assert progress == sorted(set(progress))
    assert len(progress) <= 101


def test_surface_mode_calls_json_with_disabled_syntax():
    pipeline = fake_pipeline(segmented=False)
    frame = pipeline.process("שלום עולם")
    assert pipeline.model.calls[0][1]["output_style"] == "json"
    assert pipeline.model.calls[0][1]["compute_syntax_mst"] is None
    assert json.loads(frame.morph_features.iloc[0])["prefixes"] == ["ADP"]


def test_oversized_single_word_raises_instead_of_truncating():
    pipeline = fake_pipeline()
    pipeline.tokenizer.encode = lambda text, **kwargs: list(range(len(text) + 2))
    with pytest.raises(ValueError, match="single whitespace-delimited word"):
        pipeline.process("מאודארוכהבליורווחים")
    assert not pipeline.model.calls


def test_empty_document_and_niqqud_normalization():
    pipeline = fake_pipeline()
    assert pipeline.process("\n   ").empty
    assert pipeline.process("שָׁלוֹם").word.tolist() == ["שלום"]
    assert pipeline.process("שלום").token_id.dtype == "int64"


def test_third_party_model_exit_is_a_recoverable_error():
    pipeline = fake_pipeline()

    def abort(*args, **kwargs):
        raise SystemExit(0)

    pipeline.model.predict = abort
    with pytest.raises(RuntimeError, match="Dicta aborted"):
        pipeline.process("שלום")


@pytest.mark.parametrize("settings", [
    {"batch_size": 0}, {"batch_size": True}, {"segmented": "false"},
    {"strip_niqqud": 1}, {"device": ""}, {"backend": "basic"},
])
def test_invalid_settings_raise_actionable_errors(settings):
    with pytest.raises(ValueError):
        DictaPipeline(settings)


def test_cancellation_between_batches_returns_no_partial_frame():
    pipeline = fake_pipeline(batch_size=1)
    progress = []
    with pytest.raises(ProcessingCancelled):
        pipeline.process("שלום עולם\nספר חדש", progress.append, lambda: bool(progress and progress[-1] > 0))
    assert len(pipeline.model.calls) == 1


def test_load_preserves_published_syntax_head_size_and_device(monkeypatch):
    config = SimpleNamespace(syntax_head_size=128)
    model = Mock()
    auto_config = SimpleNamespace(from_pretrained=Mock(return_value=config))
    auto_model = SimpleNamespace(from_pretrained=Mock(return_value=model))
    auto_tokenizer = SimpleNamespace(from_pretrained=Mock(return_value=WordTokenizer()))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=auto_config, AutoModel=auto_model, AutoTokenizer=auto_tokenizer))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    pipeline = DictaPipeline()
    pipeline.load()
    pipeline.load()
    kwargs = auto_model.from_pretrained.call_args.kwargs
    assert kwargs["syntax_head_size"] == 128
    assert kwargs["do_syntax"] is kwargs["do_prefix"] is kwargs["do_morph"] is kwargs["do_lex"] is True
    assert kwargs["do_ner"] is False
    assert kwargs["use_safetensors"] is True
    assert auto_model.from_pretrained.call_count == 1
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()


def test_cache_round_trip_preserves_hebrew_blank_fields_and_literal_na(tmp_path):
    project, record = make_project(tmp_path)
    settings = project.settings["nlp"]
    save_cache(project, record, settings, token_frame())
    loaded = load_cache(project, record, settings)
    pd.testing.assert_frame_equal(loaded, token_frame())
    csv_path, metadata_path = cache_paths(project, record)
    assert csv_path.name == "מחבר_ספר_1900_tokens.csv"
    assert metadata_path.is_file()


def test_cache_invalidation_source_settings_and_csv_corruption(tmp_path):
    project, record = make_project(tmp_path)
    settings = project.settings["nlp"]
    save_cache(project, record, settings, token_frame())
    assert load_cache(project, record, {**settings, "segmented": False}) is None
    (project.raw_dir / record.filename).write_text("טקסט שונה", encoding="utf-8")
    assert load_cache(project, record, settings) is None
    save_cache(project, record, settings, token_frame())
    csv_path, metadata_path = cache_paths(project, record)
    csv_path.write_text("bad,data", encoding="utf-8")
    assert load_cache(project, record, settings) is None
    save_cache(project, record, settings, token_frame())
    metadata_path.write_text("{invalid", encoding="utf-8")
    assert load_cache(project, record, settings) is None


def test_legacy_cache_without_provenance_requires_reprocessing(tmp_path):
    project, record = make_project(tmp_path)
    csv_path, _ = cache_paths(project, record)
    token_frame().to_csv(csv_path, index=False)
    assert load_cache(project, record, project.settings["nlp"]) is None
    assert csv_path.is_file()


@pytest.mark.parametrize("column,values", [
    ("token_id", [1, 3]), ("token_id", [1, 1]), ("token_id", [1, 1.5]),
    ("sentence_id", [2, 1]), ("sentence_id", [1, 0]),
    ("word", ["שלום", ""]), ("pos_tag", ["NOUN", ""]),
])
def test_cache_frame_validation_rejects_broken_schema(column, values):
    frame = token_frame()
    frame[column] = values
    with pytest.raises(ValueError):
        validate_token_frame(frame)


def test_empty_cache_is_valid(tmp_path):
    project, record = make_project(tmp_path)
    save_cache(project, record, project.settings["nlp"], empty_token_frame())
    loaded = load_cache(project, record, project.settings["nlp"])
    assert loaded is not None and loaded.empty
    assert loaded.columns.tolist() == TOKEN_COLUMNS
