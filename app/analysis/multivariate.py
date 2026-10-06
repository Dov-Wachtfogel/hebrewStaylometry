"""Distance, clustering, and low-dimensional corpus projections."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage as scipy_linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .common import AnalysisError, AnalysisResult, feature_matrix, sample_zscores


def delta_distances(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Calculate classic Delta: mean absolute difference in sample z-scores."""

    zscores, mean, std = sample_zscores(matrix)
    distances = squareform(pdist(zscores.to_numpy(), metric="cityblock") / zscores.shape[1])
    table = pd.DataFrame(distances, index=matrix.index, columns=matrix.index)
    return table, zscores, mean, std


def burrows_delta(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Build a symmetric Burrows' Delta distance table with a zero diagonal."""

    matrix = feature_matrix(corpus, params)
    table, zscores, mean, std = delta_distances(matrix)
    return AnalysisResult("Burrows Delta", table, {"matrix": matrix, "zscores": zscores, "mean": mean, "std": std, "features": list(zscores.columns)}, f"Classic Burrows' Delta over {zscores.shape[1]} variable features; sample standard deviation (ddof=1).")


def pca_projection(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any], dimensions: int) -> AnalysisResult:
    """Project standardized MFW or POS features onto principal components.

    Require enough independent variation to populate the requested dimensions,
    avoiding apparently three-dimensional plots with a necessarily zero axis.
    """

    matrix = feature_matrix(corpus, params)
    variable = matrix.loc[:, matrix.nunique(axis=0).gt(1)]
    if len(variable) < dimensions + 1 or variable.shape[1] < dimensions:
        raise AnalysisError(f"PCA {dimensions}D needs at least {dimensions + 1} documents and {dimensions} variable features.")
    standardized = StandardScaler().fit_transform(variable)
    if np.linalg.matrix_rank(standardized) < dimensions:
        raise AnalysisError(f"The corpus has fewer than {dimensions} independent directions of variation; use fewer PCA dimensions.")
    estimator = PCA(n_components=dimensions, svd_solver="full")
    coordinates = estimator.fit_transform(standardized)
    table = pd.DataFrame(coordinates, index=matrix.index, columns=[f"PC{index + 1}" for index in range(dimensions)])
    return AnalysisResult(f"PCA {dimensions}D", table, {"matrix": matrix, "features": list(variable.columns), "explained_variance": estimator.explained_variance_ratio_, "loadings": pd.DataFrame(estimator.components_.T, index=variable.columns, columns=table.columns)}, f"PCA on standardized features; the displayed components explain {estimator.explained_variance_ratio_.sum():.1%} of variance.")


def hierarchical_clustering(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Cluster texts from a condensed Delta or Euclidean distance matrix.

    Ward, centroid, and median linkage require Euclidean geometry and are
    rejected for Delta rather than applying them to a non-Euclidean matrix.
    """

    matrix = feature_matrix(corpus, params)
    if len(matrix) < 2:
        raise AnalysisError("Hierarchical clustering needs at least two documents.")
    distance = str(params.get("distance", "delta")).lower()
    method = str(params.get("linkage", "average")).lower()
    methods = {"single", "complete", "average", "weighted", "centroid", "median", "ward"}
    if method not in methods:
        raise AnalysisError(f"Unknown linkage method: {method}.")
    if distance == "delta":
        if method in {"ward", "centroid", "median"}:
            raise AnalysisError(f"{method.title()} linkage requires Euclidean distance; choose average, complete, single, or weighted for Delta.")
        table = delta_distances(matrix)[0]
        condensed = squareform(table.to_numpy(), checks=False)
    elif distance == "euclidean":
        condensed = pdist(matrix.to_numpy(), metric="euclidean")
        table = pd.DataFrame(squareform(condensed), index=matrix.index, columns=matrix.index)
    else:
        raise AnalysisError("distance must be 'delta' or 'euclidean'.")
    tree = scipy_linkage(condensed, method=method, optimal_ordering=True)
    return AnalysisResult("Hierarchical clustering", table, {"linkage": tree, "labels": list(matrix.index), "distance": distance, "method": method, "matrix": matrix}, f"{method.title()} linkage on {distance} distances; {len(matrix)} documents.")


def tsne_projection(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Produce a reproducible t-SNE projection, adjusting small-corpus perplexity."""

    matrix = feature_matrix(corpus, params)
    if len(matrix) < 3:
        raise AnalysisError("t-SNE needs at least three documents; larger corpora are more informative.")
    variable = matrix.loc[:, matrix.nunique(axis=0).gt(1)]
    if variable.empty:
        raise AnalysisError("t-SNE needs at least one variable feature.")
    try:
        requested = float(params.get("perplexity", 30))
        if not np.isfinite(requested) or requested <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise AnalysisError("perplexity must be a positive finite number.") from exc
    perplexity = min(requested, float(len(matrix) - 1))
    try:
        random_state = int(params.get("random_state", 42))
    except (TypeError, ValueError) as exc:
        raise AnalysisError("random_state must be an integer.") from exc
    estimator = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="pca" if variable.shape[1] >= 2 else "random", learning_rate="auto")
    coordinates = estimator.fit_transform(StandardScaler().fit_transform(variable))
    table = pd.DataFrame(coordinates, index=matrix.index, columns=["TSNE1", "TSNE2"])
    adjusted = f" Perplexity reduced from {requested:g} to {perplexity:g} for this corpus." if requested != perplexity else ""
    return AnalysisResult("t-SNE", table, {"matrix": matrix, "perplexity": perplexity, "random_state": random_state}, f"Exploratory t-SNE projection; axis distances are not global corpus distances.{adjusted}")
