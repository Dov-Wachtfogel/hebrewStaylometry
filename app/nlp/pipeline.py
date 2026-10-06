"""Adapter for Dicta's joint Hebrew segmentation/lemma/POS model.

The model card documents ``model.predict(sentences, tokenizer, output_style=...)``:
https://huggingface.co/dicta-il/dictabert-tiny-joint
UD output requires prefix, morphology, syntax, and lexicographic heads. Model
dependencies are imported only by ``load()``, which the GUI calls in a QThread.
"""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from contextlib import nullcontext
from typing import Any, Callable

import pandas as pd

from app.utils.hebrew import normalize_hebrew

TOKEN_COLUMNS = ["token_id", "word", "lemma", "pos_tag", "morph_features", "sentence_id"]

DEFAULT_NLP_SETTINGS: dict[str, Any] = {
    "backend": "dicta",
    "model_name": "dicta-il/dictabert-tiny-joint",
    "revision": "main",
    "device": "cpu",
    "strip_niqqud": True,
    "segmented": True,
    "batch_size": 32,
}


class ProcessingCancelled(Exception):
    """Cooperative cancellation between bounded model batches."""


def empty_token_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=TOKEN_COLUMNS)
    return frame.astype({"token_id": "int64", "sentence_id": "int64"})


_SENTENCE_BOUNDARY = re.compile(r"[.!?׃][\"'׳״”’)\]]*(?=\s|$)|[\r\n]+")


def split_hebrew_text(text: str) -> list[str]:
    """Split Hebrew paragraphs/sentences, retaining punctuation and closing quotes.

    Geresh/gershayim within Hebrew abbreviations are not sentence boundaries.
    Period-based abbreviations remain heuristic. Tokenizer-aware length limits
    are applied separately, preserving the original sentence ID across chunks.
    """
    sentences, start = [], 0
    for boundary in _SENTENCE_BOUNDARY.finditer(text):
        sentence = text[start:boundary.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = boundary.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_sentences(text: str) -> list[str]:
    """Compatibility entry point for existing scripts."""
    return split_hebrew_text(text)


def _morph_json(features: dict[str, Any]) -> str:
    return json.dumps(features, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _recover_empty_prefix_groups(
    table: list[list[str]],
) -> tuple[dict[int, tuple[str, dict[str, Any]]], set[int]]:
    """Recover only a source-backed, wholly consumed prefix segmentation.

    Dicta occasionally segments a standalone Hebrew reference letter as a
    prefix plus an empty stem. The CoNLL-U multiword row retains the surface
    form, so collapse that invalid group to one source token. Preserve every
    emitted segment in provenance and keep the stem's predicted POS/lemma.
    Other empty forms remain errors rather than guesses from model lemmas.
    """
    integer_rows = {int(row[0]): row for row in table if row[0].isdigit()}
    recovered: dict[int, tuple[str, dict[str, Any]]] = {}
    skipped: set[int] = set()
    for row in table:
        match = re.fullmatch(r"([1-9]\d*)-([1-9]\d*)", row[0])
        if match is None:
            continue
        start, end = map(int, match.groups())
        surface = row[1]
        if not surface or end <= start or end - start > len(table):
            continue
        token_ids = range(start, end + 1)
        if any(token_id not in integer_rows for token_id in token_ids):
            continue
        group = [integer_rows[token_id] for token_id in token_ids]
        # A recoverable group consists of real prefixes followed by exactly
        # one empty stem, and the prefixes already account for the full word.
        if group[-1][1] != "" or any(not member[1].strip("_") for member in group[:-1]):
            continue
        if "".join(member[1].strip("_") for member in group[:-1]) != surface:
            continue
        if any(member[3] in {"", "_"} for member in group):
            continue
        if any(member[5] not in {"", "_"} and
               any("=" not in feature for feature in member[5].split("|"))
               for member in group):
            continue
        recovered[end] = (surface, {
            "_ud_form": "",
            "_ud_word_source": "multiword_token",
            "_ud_recovery": "empty_stem_after_prefix",
            "_ud_segments": [
                {"word": member[1], "lemma": member[2], "pos_tag": member[3],
                 "morph_features": member[5]}
                for member in group
            ],
        })
        skipped.update(range(start, end))
    return recovered, skipped


def parse_ud_sentence(result: str | list[str], sentence_id: int) -> list[dict[str, Any]]:
    """Flatten CoNLL-U integer rows, preserving Hebrew prefixes and suffixes.

    Multiword range rows and empty nodes describe structure rather than tokens.
    HTB underscores indicating reconstructed morphemes are removed from the
    displayed word and retained in morphological provenance when relevant.
    An invalid all-consuming prefix group is collapsed to its original surface
    word with the stem's model tags and explicit reconstruction provenance.
    """
    if isinstance(result, str):
        lines = result.splitlines()
    elif isinstance(result, list) and all(isinstance(line, str) for line in result):
        lines = result
    else:
        raise ValueError("Dicta returned an invalid UD sentence.")
    table = []
    seen_token_ids: set[int] = set()
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < 6:
            raise ValueError("Dicta returned a malformed CoNLL-U row.")
        if columns[0].isdigit():
            token_id = int(columns[0])
            if token_id in seen_token_ids:
                raise ValueError("Dicta returned duplicate UD token IDs.")
            seen_token_ids.add(token_id)
        table.append(columns)
    recovered, skipped = _recover_empty_prefix_groups(table)
    rows = []
    for columns in table:
        if "-" in columns[0] or "." in columns[0]:
            continue
        if not columns[0].isdigit() or int(columns[0]) < 1:
            raise ValueError("Dicta returned an invalid UD token ID.")
        token_id = int(columns[0])
        if token_id in skipped:
            continue
        word = columns[1].strip("_")
        provenance: dict[str, Any] = {}
        if token_id in recovered:
            word, provenance = recovered[token_id]
        location = f"sentence {sentence_id}, UD token {columns[0]}, form {columns[1][:80]!r}"
        if not word:
            raise ValueError(f"Dicta returned a token with no recoverable word ({location}).")
        if columns[3] in {"", "_"}:
            raise ValueError(f"Dicta returned a token with no POS tag ({location}).")
        features: dict[str, Any] = {}
        if columns[5] not in {"", "_"}:
            for feature in columns[5].split("|"):
                if "=" not in feature:
                    raise ValueError("Dicta returned an invalid morphological feature.")
                key, value = feature.split("=", 1)
                features[key] = value
        if word != columns[1]:
            features["_ud_form"] = columns[1]
        features.update(provenance)
        rows.append({
            "word": word,
            "lemma": word if columns[2] in {"", "_", "[BLANK]"} else columns[2],
            "pos_tag": columns[3],
            "morph_features": _morph_json(features),
            "sentence_id": sentence_id,
        })
    return rows


def parse_json_sentence(result: dict[str, Any], sentence_id: int) -> list[dict[str, Any]]:
    """Keep unsegmented surface words and all attached morphology in JSON."""
    if not isinstance(result, dict) or not isinstance(result.get("tokens"), list):
        raise ValueError("Dicta returned an invalid JSON sentence.")
    rows = []
    for token in result["tokens"]:
        if not isinstance(token, dict):
            raise ValueError("Dicta returned an invalid token.")
        word, morph = token.get("token"), token.get("morph")
        if not isinstance(word, str) or not word or not isinstance(morph, dict) or not morph.get("pos"):
            raise ValueError("Dicta returned a token without its surface word or morphology.")
        lemma = token.get("lex")
        if not isinstance(lemma, str) or lemma in {"", "_", "[BLANK]"}:
            lemma = word
        rows.append({"word": word, "lemma": lemma, "pos_tag": str(morph["pos"]),
                     "morph_features": _morph_json(morph), "sentence_id": sentence_id})
    return rows


class DictaPipeline:
    """One reusable model per batch worker; no downloads during construction."""

    def __init__(self, settings: dict[str, Any] | None = None):
        self.settings = {**DEFAULT_NLP_SETTINGS, **(settings or {})}
        if self.settings["backend"] != "dicta":
            raise ValueError("Unsupported NLP backend. This application uses Dicta's joint model.")
        batch_size = self.settings["batch_size"]
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("NLP batch_size must be a positive integer.")
        for name in ("segmented", "strip_niqqud"):
            if not isinstance(self.settings[name], bool):
                raise ValueError(f"NLP {name} must be a boolean.")
        for name in ("model_name", "revision", "device"):
            if not isinstance(self.settings[name], str) or not self.settings[name].strip():
                raise ValueError(f"NLP {name} must be a nonempty string.")
        if "local_files_only" in self.settings and not isinstance(self.settings["local_files_only"], bool):
            raise ValueError("NLP local_files_only must be a boolean.")
        self.tokenizer: Any = None
        self.model: Any = None
        self._torch: Any = None

    def load(self) -> None:
        """Download/load model weights on the caller's background thread.

        Dicta publishes custom modeling code, so trust_remote_code is needed.
        Set revision (and optionally code_revision) to reviewed commit hashes
        when reproducibility across machines is required.
        """
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Dicta processing requires torch, transformers, and safetensors. Install requirements.txt.") from exc
        self._torch = torch
        device = str(self.settings["device"])
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device. Select CPU.")
        shared = {"revision": self.settings["revision"], "trust_remote_code": True,
                  "local_files_only": bool(self.settings.get("local_files_only", False))}
        if self.settings.get("code_revision"):
            shared["code_revision"] = self.settings["code_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings["model_name"], **shared)
        segmented = bool(self.settings["segmented"])
        config = AutoConfig.from_pretrained(self.settings["model_name"], **shared)
        model = AutoModel.from_pretrained(
            self.settings["model_name"], **shared, use_safetensors=True,
            config=config, syntax_head_size=getattr(config, "syntax_head_size", 128),
            do_syntax=segmented, do_ner=False, do_prefix=True, do_lex=True, do_morph=True,
        )
        model.to(device)
        model.eval()
        self.model = model

    def _max_input_length(self) -> int:
        lengths = [int(getattr(self.model.config, "max_position_embeddings", 512))]
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
            lengths.append(tokenizer_limit)
        configured = self.settings.get("max_sequence_length")
        if configured is not None:
            if isinstance(configured, bool) or not isinstance(configured, int) or configured < 4:
                raise ValueError("max_sequence_length must be an integer of at least 4.")
            lengths.append(configured)
        limit = min(lengths)
        if limit < 4:
            raise ValueError("The NLP tokenizer/model has an invalid input limit.")
        return limit

    def _input_length(self, text: str) -> int:
        # Measuring a long sentence is intentional: it is split below before
        # prediction. Avoid the tokenizer's inference-limit warning for this
        # measurement only; truncation stays disabled so no text is lost.
        return len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False))

    def _chunks(self, sentence: str, is_cancelled: Callable[[], bool]) -> list[str]:
        """Use one subword offset scan, then verify only bounded candidates.

        Searching the entire remaining sentence for every chunk repeatedly
        tokenizes huge overlapping strings. Fast-tokenizer offsets let us walk
        word boundaries once. Slow/custom tokenizers use a bounded word window.
        """
        if is_cancelled():
            raise ProcessingCancelled()
        limit = self._max_input_length()
        spans = list(re.finditer(r"\S+", sentence))
        if not spans:
            return []
        token_ends = None
        if getattr(self.tokenizer, "is_fast", False):
            encoded = self.tokenizer(
                sentence, add_special_tokens=False, truncation=False,
                return_offsets_mapping=True, verbose=False,
            )
            offsets = encoded["offset_mapping"]
            special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
            budget = limit - special_tokens
            if budget < 1:
                raise ValueError("The model input limit leaves no room for text tokens.")
            if len(offsets) <= budget:
                return [sentence]
            token_ends, token_index = [], 0
            for index, span in enumerate(spans):
                if index % 256 == 0 and is_cancelled():
                    raise ProcessingCancelled()
                while token_index < len(offsets) and offsets[token_index][0] < span.end():
                    token_index += 1
                token_ends.append(token_index)
        elif self._input_length(sentence) <= limit:
            return [sentence]

        chunks, start = [], 0
        while start < len(spans):
            if is_cancelled():
                raise ProcessingCancelled()
            begin = spans[start].start()
            if token_ends is not None:
                previous = token_ends[start - 1] if start else 0
                end = bisect_right(token_ends, previous + budget, lo=start)
            else:
                end = min(start + limit, len(spans))
            if end <= start:
                raise ValueError("A single whitespace-delimited word exceeds the model input limit; add a word boundary before processing.")
            chunk = sentence[begin:spans[end - 1].end()]
            # Retokenizing a substring can change edge wordpieces. Verify the
            # final candidate and shrink only this bounded window if necessary.
            if self._input_length(chunk) > limit:
                if self._input_length(sentence[begin:spans[start].end()]) > limit:
                    raise ValueError("A single whitespace-delimited word exceeds the model input limit; add a word boundary before processing.")
                low, high = start + 1, end - 1
                while low < high:
                    if is_cancelled():
                        raise ProcessingCancelled()
                    middle = (low + high + 1) // 2
                    if self._input_length(sentence[begin:spans[middle - 1].end()]) <= limit:
                        low = middle
                    else:
                        high = middle - 1
                end = low
                chunk = sentence[begin:spans[end - 1].end()]
            chunks.append(chunk)
            start = end
        return chunks

    def process(
        self, text: str, progress: Callable[[int], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> pd.DataFrame:
        """Return validated logical Hebrew tokens; call load() before processing."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Load the Dicta model before processing a text.")
        cancelled = is_cancelled or (lambda: False)
        if cancelled():
            raise ProcessingCancelled()
        if progress:
            progress(0)
        normalized = normalize_hebrew(text, strip_niqqud=bool(self.settings["strip_niqqud"]))
        inputs = [(sentence_id, chunk) for sentence_id, sentence in enumerate(split_hebrew_text(normalized), 1)
                  for chunk in self._chunks(sentence, cancelled)]
        if not inputs:
            if progress:
                progress(100)
            return empty_token_frame()
        rows: list[dict[str, Any]] = []
        batch_size = self.settings["batch_size"]
        segmented = bool(self.settings["segmented"])
        parser = parse_ud_sentence if segmented else parse_json_sentence
        last_progress = 0
        for start in range(0, len(inputs), batch_size):
            if cancelled():
                raise ProcessingCancelled()
            batch = inputs[start:start + batch_size]
            context = self._torch.inference_mode() if self._torch is not None else nullcontext()
            with context:
                try:
                    results = self.model.predict(
                        # Dicta predicts the supplied list as one tensor batch;
                        # its API has no batch_size keyword. Bound the list here.
                        [sentence for _, sentence in batch], self.tokenizer,
                        padding="longest", truncation=False,
                        compute_syntax_mst=True if segmented else None,
                        output_style="ud" if segmented else "json",
                    )
                except SystemExit as exc:
                    # Some published Dicta revisions exit on malformed UD
                    # conversion. Translate that into a recoverable document
                    # error so the worker's lifecycle signals remain coherent.
                    raise RuntimeError("Dicta aborted while converting its prediction to tokens.") from exc
            if cancelled():
                raise ProcessingCancelled()
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError("Dicta did not return one result for every input sentence chunk.")
            for (sentence_id, _), result in zip(batch, results):
                parsed = parser(result, sentence_id)
                if not parsed:
                    raise ValueError("Dicta returned no tokens for a nonempty sentence chunk.")
                rows.extend(parsed)
            percent = round(100 * min(start + batch_size, len(inputs)) / len(inputs))
            if progress and percent > last_progress:
                progress(percent)
                last_progress = percent
        frame = pd.DataFrame.from_records(rows, columns=TOKEN_COLUMNS[1:])
        frame.insert(0, "token_id", range(1, len(frame) + 1))
        return frame
