"""Numerical and boundary regression tests for the analysis engine."""

import numpy as np
import pandas as pd
import pytest

from app.analysis import AnalysisError, normalize_word, run_analysis


def tokens(words, *, sentences=None, tags=None):
    """Build a small cached-token fixture with the production CSV schema."""

    return pd.DataFrame({
        "token_id": range(len(words)), "word": words, "lemma": words,
        "pos_tag": tags or ["NOUN"] * len(words),
        "morph_features": [""] * len(words),
        "sentence_id": sentences if sentences is not None else [0] * len(words),
    })


def varied_corpus():
    return {
        "A": tokens(["a", "a", "b", "c"]),
        "B": tokens(["a", "b", "b", "d"]),
        "C": tokens(["b", "c", "d", "d"]),
    }


def test_mfw_denominator_is_all_tokens_and_input_is_unchanged():
    corpus = {"A": tokens(["a", "a", "b", "."], tags=["NOUN", "NOUN", "NOUN", "PUNCT"])}
    original = corpus["A"].copy(deep=True)
    result = run_analysis(corpus, "MFW", {"n_mfw": 1})
    assert result.table.index.tolist() == ["a"]
    assert result.table.loc["a", "token_count"] == 2
    assert result.payload["matrix"].loc["A", "a"] == pytest.approx(2 / 3)
    pd.testing.assert_frame_equal(corpus["A"], original)


def test_burrows_delta_matches_sample_zscore_definition():
    corpus = varied_corpus()
    result = run_analysis(corpus, "Burrows Delta", {"n_mfw": 4})
    matrix = run_analysis(corpus, "MFW", {"n_mfw": 4}).payload["matrix"]
    z = (matrix - matrix.mean()) / matrix.std(ddof=1)
    expected = np.abs(z.loc["A"] - z.loc["B"]).mean()
    assert result.table.loc["A", "B"] == pytest.approx(expected)
    np.testing.assert_allclose(result.table, result.table.T)
    np.testing.assert_allclose(np.diag(result.table), 0)


def test_delta_excludes_constant_features_and_reports_no_variation():
    corpus = {"A": tokens(["x", "a"]), "B": tokens(["x", "b"])}
    result = run_analysis(corpus, "Burrows Delta", {})
    assert "x" not in result.payload["features"]
    assert result.table.loc["A", "B"] == pytest.approx(np.sqrt(2))
    with pytest.raises(AnalysisError, match="No variable features"):
        run_analysis({"A": tokens(["x"]), "B": tokens(["x"])}, "Burrows Delta", {})


def test_lexical_diversity_mattr_hapax_and_dis_legomena():
    result = run_analysis({"A": tokens(["a", "a", "b", "b", "c", "d"])}, "Lexical diversity", {"mattr_window": 3})
    row = result.table.loc["A"]
    assert row["TTR"] == pytest.approx(4 / 6)
    assert row["MATTR"] == pytest.approx(0.75)
    assert row["hapax_count"] == 2
    assert row["dis_count"] == 2


def test_word_and_pos_ngrams_never_cross_sentences():
    corpus = {"A": tokens(["a", "b", "c", "d"], sentences=[0, 0, 1, 1], tags=["NOUN", "VERB", "ADJ", "ADP"])}
    words = run_analysis(corpus, "Word n-grams", {"ngram_n": 2}).table
    pos = run_analysis(corpus, "POS n-grams", {"ngram_n": 2}).table
    assert set(words.index) == {"a b", "c d"}
    assert set(pos.index) == {"NOUN VERB", "ADJ ADP"}
    assert "b c" not in words.index


def test_pos_proportions_include_punctuation_and_sum_to_one():
    result = run_analysis({"A": tokens(["a", "b", "."], tags=["NOUN", "VERB", "PUNCT"])}, "POS distribution", {})
    assert result.table.loc["A", "PUNCT"] == pytest.approx(1 / 3)
    assert result.table.loc["A"].sum() == pytest.approx(1)


def test_rolling_reference_fit_excludes_target_and_preserves_source_offsets():
    references = {
        "R1": tokens(["a", "a", "a", "b"]),
        "R2": tokens(["a", "b", "b", "b"]),
    }
    corpus = {
        **references,
        "target": tokens([".", "a", "a", ".", "b", "b"], tags=["PUNCT", "NOUN", "NOUN", "PUNCT", "NOUN", "NOUN"]),
    }
    settings = {"target": "target", "window_size": 2, "step_size": 2, "n_mfw": 2}
    result = run_analysis(corpus, "Rolling Delta", settings)
    assert result.table["start"].tolist() == [1, 4]
    assert result.table["end"].tolist() == [3, 6]
    assert result.table.loc[0, "Delta: R1"] == pytest.approx(1 / np.sqrt(2))
    changed = {**references, "target": tokens(["targetonly"] * 20)}
    changed_result = run_analysis(changed, "Rolling Delta", settings)
    pd.testing.assert_series_equal(result.payload["reference_mean"], changed_result.payload["reference_mean"])
    pd.testing.assert_series_equal(result.payload["reference_std"], changed_result.payload["reference_std"])
    assert "targetonly" not in changed_result.payload["features"]


def test_hca_ward_rejects_delta_but_accepts_euclidean():
    with pytest.raises(AnalysisError, match="requires Euclidean"):
        run_analysis(varied_corpus(), "Hierarchical clustering", {"distance": "delta", "linkage": "ward"})
    result = run_analysis(varied_corpus(), "Hierarchical clustering", {"distance": "euclidean", "linkage": "ward"})
    assert result.payload["linkage"].shape == (2, 4)
    assert result.payload["labels"] == ["A", "B", "C"]


def test_pca_dimension_validation_and_valid_projection():
    with pytest.raises(AnalysisError, match="PCA 3D needs"):
        run_analysis(varied_corpus(), "PCA 3D", {})
    result = run_analysis(varied_corpus(), "PCA 2D", {})
    assert result.table.shape == (3, 2)
    assert result.table.columns.tolist() == ["PC1", "PC2"]
    assert sum(result.payload["explained_variance"]) == pytest.approx(1)


def test_hebrew_normalization_and_stop_words():
    assert normalize_word("שָׁלוֹם") == "שלום"
    result = run_analysis({"A": tokens(["שָׁלוֹם", "שלום", "עולם"])}, "MFW", {"stop_words": ["שלום"]})
    assert result.table.index.tolist() == ["עולם"]


@pytest.mark.parametrize("kind", ["MFW", "POS distribution", "Burrows Delta"])
def test_empty_corpus_reports_actionable_error(kind):
    with pytest.raises(AnalysisError, match="Load and preprocess"):
        run_analysis({}, kind, {})


def test_non_contiguous_repeated_sentence_ids_do_not_merge_ngrams():
    result = run_analysis({"A": tokens(["a", "b", "c", "d", "e", "f"], sentences=[0, 0, 1, 1, 0, 0])}, "Word n-grams", {})
    assert set(result.table.index) == {"a b", "c d", "e f"}


def test_short_mattr_window_and_missing_lemma_fallback():
    frame = tokens(["a", "b"])
    frame.loc[0, "lemma"] = None
    result = run_analysis({"A": frame}, "Lexical diversity", {"mattr_window": 100})
    assert result.table.loc["A", "MATTR"] == 1


def test_invalid_integer_parameter_is_not_truncated():
    with pytest.raises(AnalysisError, match="positive integer"):
        run_analysis(varied_corpus(), "MFW", {"n_mfw": 2.5})


def test_pos_feature_basis_for_delta():
    corpus = {
        "A": tokens(["a", "a"], tags=["NOUN", "NOUN"]),
        "B": tokens(["a", "a"], tags=["VERB", "VERB"]),
    }
    result = run_analysis(corpus, "Burrows Delta", {"feature_type": "pos"})
    assert set(result.payload["features"]) == {"NOUN", "VERB"}
    assert result.table.loc["A", "B"] == pytest.approx(np.sqrt(2))


def test_tsne_perplexity_reduction_is_reported(monkeypatch):
    """Test the small-corpus guard independently of stochastic optimization."""

    from app.analysis import multivariate

    arguments = {}

    class RecordingTSNE:
        def __init__(self, **kwargs):
            arguments.update(kwargs)

        def fit_transform(self, matrix):
            return np.zeros((len(matrix), 2))

    monkeypatch.setattr(multivariate, "TSNE", RecordingTSNE)
    result = run_analysis(varied_corpus(), "t-SNE", {"perplexity": 30})
    assert arguments["perplexity"] == 2
    assert result.payload["perplexity"] == 2
    assert "reduced from 30 to 2" in result.description


def test_length_statistics_ignore_vocalization_and_keep_stop_words():
    corpus = {"A": tokens(["שָׁלוֹם", "עולם", ".", "שלום"], sentences=[0, 0, 0, 1], tags=["NOUN", "NOUN", "PUNCT", "NOUN"])}
    result = run_analysis(corpus, "Sentence / word length", {"stop_words": ["שלום"]})
    assert result.payload["word_lengths"]["A"] == [4, 4, 4]
    assert result.payload["sentence_lengths"]["A"] == [2, 1]
    assert result.table.loc["A", "sentence_mean"] == pytest.approx(1.5)
