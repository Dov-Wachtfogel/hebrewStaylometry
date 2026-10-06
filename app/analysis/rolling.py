"""Sliding-window comparison against an independently fitted reference corpus."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import (
    AnalysisError,
    AnalysisResult,
    counts_and_frequencies,
    integer_parameter,
    lexical_rows,
    mfw_matrix,
    sample_zscores,
)


def rolling_delta(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Compare full overlapping target windows to fixed reference texts.

    The target is excluded from vocabulary selection, reference means, and
    sample deviations. Each window uses the reference vocabulary and divides
    feature counts by its full lexical token length. Source ``start`` offsets
    are inclusive, ``end`` offsets exclusive, and both refer to the original
    token table, including skipped punctuation and stop words.
    """

    target = str(params.get("target", ""))
    if target not in corpus:
        raise AnalysisError("Choose a target document for rolling analysis.")
    references = {label: frame for label, frame in corpus.items() if label != target}
    if len(references) < 2:
        raise AnalysisError("Rolling Delta needs a target and at least two separate reference documents.")
    window_size = integer_parameter(params, "window_size", 500)
    step_size = integer_parameter(params, "step_size", 100)
    _, reference_matrix, _ = mfw_matrix(references, params)
    reference_z, mean, std = sample_zscores(reference_matrix)
    tokens = lexical_rows(corpus[target], params)
    if len(tokens) < window_size:
        raise AnalysisError(f"The target has {len(tokens)} eligible tokens, fewer than the {window_size}-token window; reduce window size.")
    rows: list[dict[str, Any]] = []
    vocabulary = list(mean.index)
    for offset in range(0, len(tokens) - window_size + 1, step_size):
        window = tokens.iloc[offset : offset + window_size]
        _, frequency = counts_and_frequencies({target: window}, vocabulary)
        window_z = (frequency.iloc[0] - mean) / std
        distances = reference_z.sub(window_z, axis=1).abs().mean(axis=1)
        row: dict[str, Any] = {
            "start": int(window["_position"].iloc[0]),
            "end": int(window["_position"].iloc[-1]) + 1,
            "lexical_start": offset,
            "lexical_end": offset + window_size,
        }
        row.update({f"Delta: {label}": float(value) for label, value in distances.items()})
        nearest = str(distances.idxmin())
        row["nearest_reference"] = nearest
        rows.append(row)
    table = pd.DataFrame(rows)
    table.index.name = "window"
    return AnalysisResult("Rolling Delta", table, {
        "target": target, "references": list(references), "features": vocabulary,
        "reference_matrix": reference_matrix.loc[:, vocabulary],
        "reference_mean": mean, "reference_std": std,
        "reference_zscores": reference_z, "distance_columns": [f"Delta: {label}" for label in references],
        "window_size": window_size, "step_size": step_size,
    }, f"{len(table)} full {window_size}-token windows with step {step_size}; vocabulary and z-scores are fitted only on {len(references)} reference documents. Source offsets are zero-based with an exclusive end.")
