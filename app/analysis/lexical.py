"""Lexical, part-of-speech, n-gram, and length summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .common import (
    AnalysisError,
    AnalysisResult,
    integer_parameter,
    lexical_documents,
    lexical_rows,
    mfw_matrix,
    normalize_word,
    pos_matrix,
)


def most_frequent_words(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Summarize global MFW counts and retain document frequency matrices."""

    counts, matrix, documents = mfw_matrix(corpus, params)
    total_tokens = sum(len(frame) for frame in documents.values())
    table = pd.DataFrame({
        "token_count": counts.sum(axis=0),
        "corpus_frequency": counts.sum(axis=0) / total_tokens,
        "document_frequency": counts.gt(0).sum(axis=0),
    })
    table.index.name = "feature"
    return AnalysisResult("MFW", table, {"matrix": matrix, "counts": counts}, f"{len(table)} most frequent lexical features; proportions use all eligible tokens.")


def pos_distribution(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Calculate POS proportions for each document; include punctuation tags."""

    counts, matrix = pos_matrix(corpus)
    return AnalysisResult("POS distribution", matrix, {"counts": counts, "matrix": matrix}, "Relative frequencies of annotated POS categories, including PUNCT.")


def moving_average_ttr(tokens: list[str], window: int) -> float:
    """Average TTR over all full overlapping windows, using O(n) updates.

    A short nonempty document uses a single window containing the whole text;
    an empty sequence returns NaN rather than inventing a diversity score.
    """

    if not tokens:
        return float("nan")
    width = min(window, len(tokens))
    counter = Counter(tokens[:width])
    total = float(len(counter))
    for position in range(width, len(tokens)):
        outgoing = tokens[position - width]
        counter[outgoing] -= 1
        if counter[outgoing] == 0:
            del counter[outgoing]
        counter[tokens[position]] += 1
        total += len(counter)
    return total / ((len(tokens) - width + 1) * width)


def lexical_diversity(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Calculate TTR, MATTR, and counts of once- and twice-occurring types."""

    window = integer_parameter(params, "mattr_window", 100)
    documents = lexical_documents(corpus, params, allow_empty=True)
    rows: dict[str, dict[str, int | float]] = {}
    for label, frame in documents.items():
        tokens = frame["_feature"].tolist()
        counter = Counter(tokens)
        hapax = sum(value == 1 for value in counter.values())
        dis = sum(value == 2 for value in counter.values())
        rows[label] = {
            "tokens": len(tokens), "types": len(counter),
            "TTR": len(counter) / len(tokens) if tokens else np.nan,
            "MATTR": moving_average_ttr(tokens, window),
            "hapax_count": hapax, "dis_count": dis,
            "hapax_type_ratio": hapax / len(counter) if counter else np.nan,
            "dis_type_ratio": dis / len(counter) if counter else np.nan,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "document"
    return AnalysisResult("Lexical diversity", table, {"mattr_window": window}, f"MATTR averages all overlapping {window}-token windows; shorter documents use their full length. Hapax/dis ratios use types as denominator.")


def ngram_frequencies(
    corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any], *, pos: bool = False
) -> AnalysisResult:
    """Count n-grams separately within each contiguous annotated sentence.

    Word n-grams use the same lexical filtering as MFW. POS n-grams include
    punctuation categories. Neither form can cross a sentence boundary.
    """

    order = integer_parameter(params, "ngram_n", 2)
    limit = integer_parameter(params, "n_mfw", 100)
    counters: dict[str, Counter[str]] = {}
    for label, frame in corpus.items():
        counter: Counter[str] = Counter()
        for _, sentence in frame.groupby("_sentence_run", sort=False):
            if pos:
                sequence = sentence.loc[sentence["pos_tag"].ne(""), "pos_tag"].tolist()
            else:
                sequence = lexical_rows(sentence, params)["_feature"].tolist()
            counter.update(" ".join(sequence[index : index + order]) for index in range(len(sequence) - order + 1))
        counters[label] = counter
    totals: Counter[str] = Counter()
    for counter in counters.values():
        totals.update(counter)
    if not totals:
        raise AnalysisError(f"No within-sentence {order}-grams are available; reduce the n-gram order.")
    vocabulary = [feature for feature, _ in sorted(totals.items(), key=lambda item: (-item[1], item[0]))[:limit]]
    counts = pd.DataFrame.from_dict(counters, orient="index").reindex(index=list(corpus), columns=vocabulary).fillna(0).astype(int)
    counts.index.name = "document"
    denominators = pd.Series({label: sum(counter.values()) for label, counter in counters.items()})
    matrix = counts.div(denominators.replace(0, np.nan), axis=0).fillna(0.0)
    table = pd.DataFrame({
        "token_count": counts.sum(axis=0),
        "corpus_frequency": counts.sum(axis=0) / sum(totals.values()),
        "document_frequency": counts.gt(0).sum(axis=0),
    })
    table.index.name = "ngram"
    kind = "POS n-grams" if pos else "Word n-grams"
    return AnalysisResult(kind, table, {"matrix": matrix, "counts": counts, "ngram_n": order}, f"Top {len(table)} within-sentence {order}-grams; frequencies use all eligible n-grams.")


def length_statistics(corpus: Mapping[str, pd.DataFrame], params: Mapping[str, Any]) -> AnalysisResult:
    """Summarize unvocalized surface-word lengths and lexical sentence lengths.

    Stop words remain part of word and sentence length measurements. Standard
    deviations are descriptive population deviations (``ddof=0``).
    """

    rows: dict[str, dict[str, int | float]] = {}
    word_lengths: dict[str, list[int]] = {}
    sentence_lengths: dict[str, list[int]] = {}
    for label, frame in corpus.items():
        lexical = lexical_rows(frame, {"field": "word"})
        words = [len(normalize_word(word)) for word in lexical["word"]]
        sentences = lexical.groupby("_sentence_run", sort=False).size().astype(int).tolist()
        word_lengths[label], sentence_lengths[label] = words, sentences
        row: dict[str, int | float] = {"tokens": len(words), "sentences": len(sentences)}
        for prefix, values in (("word", words), ("sentence", sentences)):
            row[f"{prefix}_mean"] = float(np.mean(values)) if values else np.nan
            row[f"{prefix}_std"] = float(np.std(values, ddof=0)) if values else np.nan
            row[f"{prefix}_median"] = float(np.median(values)) if values else np.nan
        rows[label] = row
    if not any(word_lengths.values()):
        raise AnalysisError("No surface words are available for length analysis.")
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "document"
    return AnalysisResult("Sentence / word length", table, {"word_lengths": word_lengths, "sentence_lengths": sentence_lengths}, "Word lengths exclude Hebrew vocalization marks; sentence lengths exclude punctuation. Standard deviations use ddof=0.")
