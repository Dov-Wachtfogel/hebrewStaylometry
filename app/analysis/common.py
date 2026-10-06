"""Validated corpus access shared by the stylometric analyses.

Frequency matrices use documents as rows and features as columns. Their
denominators include all eligible tokens, not only the selected vocabulary.
This makes comparisons with a fixed reference vocabulary meaningful.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Mapping

import numpy as np
import pandas as pd


class AnalysisError(ValueError):
    """A corpus or parameter cannot support the requested analysis."""


@dataclass
class AnalysisResult:
    """A result table and the additional data needed to render a chart.

    ``payload`` deliberately keeps NumPy arrays and DataFrames in their native
    form: exporters can use ``table`` directly, while renderers avoid reparsing
    numerical data. Result tables label their document index ``document``.
    """

    kind: str
    table: pd.DataFrame
    payload: dict[str, Any] = dataclass_field(default_factory=dict)
    description: str = ""


def normalize_word(value: Any) -> str:
    """Normalize a lexical feature, removing Hebrew cantillation and niqqud.

    Hebrew base letters are preserved. Unicode normalization also makes
    canonically equivalent text compare consistently. Latin text is casefolded.
    """

    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFC", str(value)).strip()
    text = "".join(
        char
        for char in text
        if not ("\u0591" <= char <= "\u05c7" and unicodedata.combining(char))
        and char not in {"\u200b", "\u200c", "\u200d", "\ufeff"}
    )
    return text.casefold()


def integer_parameter(params: Mapping[str, Any], name: str, default: int) -> int:
    """Read a positive integer without silently truncating fractional values."""

    value = params.get(name, default)
    if isinstance(value, bool):
        raise AnalysisError(f"{name} must be a positive integer.")
    try:
        result = int(value)
        if float(value) != result or result < 1:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError(f"{name} must be a positive integer.") from exc
    return result


def validated_corpus(corpus: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Copy and validate token tables without modifying cached input frames."""

    if not corpus:
        raise AnalysisError("Load and preprocess at least one document first.")
    result: dict[str, pd.DataFrame] = {}
    for label, frame in corpus.items():
        if not isinstance(frame, pd.DataFrame):
            raise AnalysisError(f"{label}: expected a token DataFrame.")
        missing = {"word", "lemma", "pos_tag", "sentence_id"}.difference(frame.columns)
        if missing:
            raise AnalysisError(f"{label}: missing token columns: {', '.join(sorted(missing))}.")
        document = str(label)
        if document in result:
            raise AnalysisError("Document labels must be unique.")
        copied = frame.copy().reset_index(drop=True)
        copied["_position"] = np.arange(len(copied), dtype=int)
        for column in ("word", "lemma", "pos_tag"):
            copied[column] = copied[column].fillna("").astype(str).str.strip()
        copied["pos_tag"] = copied["pos_tag"].str.upper()
        # Missing sentence IDs form separate runs; they must never accidentally
        # join two sentences into a single n-gram stream.
        sentence = copied["sentence_id"]
        copied["_sentence_run"] = (
            sentence.ne(sentence.shift()) | sentence.isna() | sentence.shift().isna()
        ).cumsum()
        result[document] = copied
    return result


def lexical_rows(frame: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Return eligible lexical rows with normalized ``_feature`` values.

    An absent lemma falls back to the surface word. Punctuation and configured
    stop words are excluded, preserving source row offsets for rolling analysis.
    """

    column = params.get("field", "lemma")
    if column not in {"word", "lemma"}:
        raise AnalysisError("field must be 'word' or 'lemma'.")
    rows = frame.copy()
    values = rows[column]
    if column == "lemma":
        values = values.where(values.ne(""), rows["word"])
    rows["_feature"] = values.map(normalize_word)
    stop_values = params.get("stop_words", []) or []
    if isinstance(stop_values, str):
        raise AnalysisError("stop_words must be a list of words, not a single string.")
    stop_words = {normalize_word(value) for value in stop_values}
    keep = rows["_feature"].ne("") & ~rows["_feature"].isin(stop_words)
    if not params.get("include_punctuation", False):
        keep &= rows["pos_tag"].ne("PUNCT")
    return rows.loc[keep].copy()


def lexical_documents(
    corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any], *, allow_empty: bool = False
) -> dict[str, pd.DataFrame]:
    """Prepare lexical rows and report empty documents before analysis."""

    result = {label: lexical_rows(frame, params) for label, frame in corpus.items()}
    empty = [label for label, frame in result.items() if frame.empty]
    if len(empty) == len(result):
        raise AnalysisError("The corpus has no lexical tokens after filtering.")
    if empty and not allow_empty:
        raise AnalysisError(f"No lexical tokens after filtering in: {', '.join(empty)}.")
    return result


def counts_and_frequencies(
    documents: Mapping[str, pd.DataFrame], vocabulary: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count a fixed vocabulary and divide by each document's full token count."""

    counts = pd.DataFrame(
        [Counter(frame["_feature"]) for frame in documents.values()],
        index=pd.Index(documents, name="document"),
    ).reindex(columns=vocabulary, fill_value=0).fillna(0).astype(int)
    sizes = pd.Series({label: len(frame) for label, frame in documents.items()})
    frequencies = counts.div(sizes.replace(0, np.nan), axis=0).fillna(0.0)
    return counts, frequencies


def mfw_matrix(
    corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Build an MFW matrix from corpus-wide counts with deterministic tie breaks."""

    documents = lexical_documents(corpus, params)
    limit = integer_parameter(params, "n_mfw", 100)
    totals = Counter(feature for frame in documents.values() for feature in frame["_feature"])
    vocabulary = [feature for feature, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]]
    counts, frequencies = counts_and_frequencies(documents, vocabulary)
    return counts, frequencies, documents


def pos_matrix(corpus: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return POS counts and proportions, including annotated punctuation."""

    counters = {label: Counter(frame.loc[frame["pos_tag"].ne(""), "pos_tag"]) for label, frame in corpus.items()}
    empty = [label for label, counter in counters.items() if not counter]
    if empty:
        raise AnalysisError(f"No POS annotations in: {', '.join(empty)}.")
    counts = pd.DataFrame.from_dict(counters, orient="index").fillna(0).astype(int)
    counts = counts.reindex(sorted(counts.columns), axis=1)
    counts.index.name = "document"
    return counts, counts.div(counts.sum(axis=1), axis=0)


def feature_matrix(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> pd.DataFrame:
    """Choose the MFW or POS feature space for multivariate analyses."""

    space = str(params.get("feature_type", params.get("feature_space", "mfw"))).lower()
    if space == "mfw":
        return mfw_matrix(corpus, params)[1]
    if space == "pos":
        return pos_matrix(corpus)[1]
    raise AnalysisError("feature_type must be 'mfw' or 'pos'.")


def sample_zscores(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fit sample z-scores, dropping exactly constant or undefined features.

    Burrows' Delta uses the sample standard deviation (``ddof=1``). A constant
    feature contributes no discrimination and must not dilute the mean distance.
    """

    if len(matrix) < 2:
        raise AnalysisError("Burrows' Delta needs at least two documents.")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1)
    keep = matrix.nunique(axis=0).gt(1) & std.notna() & std.gt(0) & np.isfinite(std)
    if not keep.any():
        raise AnalysisError("No variable features remain; increase N MFW or use more varied documents.")
    mean, std = mean.loc[keep], std.loc[keep]
    return (matrix.loc[:, keep] - mean) / std, mean, std
