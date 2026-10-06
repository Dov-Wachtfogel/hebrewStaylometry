"""Public dispatch API for the desktop application's analysis dashboard."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import pandas as pd

from .common import AnalysisError, AnalysisResult, validated_corpus
from .lexical import lexical_diversity, length_statistics, most_frequent_words, ngram_frequencies, pos_distribution
from .multivariate import burrows_delta, hierarchical_clustering, pca_projection, tsne_projection
from .rolling import rolling_delta


ANALYSIS_KINDS = (
    "MFW", "POS distribution", "Lexical diversity", "PCA 2D", "PCA 3D",
    "Burrows Delta", "Hierarchical clustering", "t-SNE", "Word n-grams",
    "POS n-grams", "Sentence / word length", "Rolling Delta",
)

_ANALYSES: dict[str, Callable[[Mapping[str, pd.DataFrame], Mapping[str, Any]], AnalysisResult]] = {
    "MFW": most_frequent_words,
    "POS distribution": pos_distribution,
    "Lexical diversity": lexical_diversity,
    "PCA 2D": lambda corpus, params: pca_projection(corpus, params, 2),
    "PCA 3D": lambda corpus, params: pca_projection(corpus, params, 3),
    "Burrows Delta": burrows_delta,
    "Hierarchical clustering": hierarchical_clustering,
    "t-SNE": tsne_projection,
    "Word n-grams": ngram_frequencies,
    "POS n-grams": lambda corpus, params: ngram_frequencies(corpus, params, pos=True),
    "Sentence / word length": length_statistics,
    "Rolling Delta": rolling_delta,
}


def run_analysis(
    corpus: Mapping[str, pd.DataFrame], kind: str, params: Mapping[str, Any] | None = None
) -> AnalysisResult:
    """Run a supported analysis against cached token tables.

    The input corpus is copied and never modified. Invalid data or unsupported
    settings raise ``AnalysisError`` with a user-facing explanation. Call this
    function from a background worker for large corpora; it performs no UI or
    filesystem operations. Feature selection is MFW by default; set
    ``feature_type='pos'`` for multivariate analysis of POS proportions
    (``feature_space`` is accepted as an alias).
    """

    if kind not in _ANALYSES:
        raise AnalysisError(f"Unknown analysis: {kind}.")
    return _ANALYSES[kind](validated_corpus(corpus), dict(params or {}))
