"""Read-only regression check for Dicta output from long Hebrew source texts.

Run with a source path and --diagnose to print the first raw UD token whose
FORM becomes empty when reconstruction underscores are removed. --verify
instead checks the current production adapter across the complete source.
No corpus files or cache files are modified. Model weights must be cached.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nlp.cache import validate_token_frame
from app.nlp.pipeline import DictaPipeline, split_sentences
from app.utils.hebrew import normalize_hebrew
from app.utils.io import atomic_write_csv, atomic_write_json, read_hebrew_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--diagnose", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--max-sentences", type=int, help="Limit diagnosis to the first N original sentences.")
    parser.add_argument("--sentence-id", type=int, help="Diagnose only this 1-based original sentence.")
    parser.add_argument("--output", type=Path, help="Optional verified CSV destination; no cache sidecar is written.")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    source_digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    text = read_hebrew_text(args.source)
    pipeline = DictaPipeline({"device": "cpu", "local_files_only": True,
                              "batch_size": args.batch_size,
                              "max_sequence_length": args.max_sequence_length})
    start = time.monotonic()
    print("Loading cached Dicta tiny model on CPU.", flush=True)
    pipeline.load()
    print(json.dumps({"source_chars": len(text), "threads": args.threads,
                      "model_revision": pipeline.model.config._commit_hash,
                      "model_code": type(pipeline.model).__module__}, ensure_ascii=False), flush=True)
    if args.verify:
        last_report = [-1]

        def progress(value: int) -> None:
            milestone = value // 5
            if milestone != last_report[0]:
                print(f"Progress={value}% elapsed={time.monotonic() - start:.1f}s", flush=True)
                last_report[0] = milestone

        frame = validate_token_frame(pipeline.process(text, progress=progress))
        print(json.dumps({"validated_tokens": len(frame),
                          "sentences": frame.sentence_id.nunique(),
                          "elapsed_seconds": round(time.monotonic() - start, 2)}, ensure_ascii=False), flush=True)
        if args.output is not None:
            atomic_write_csv(args.output, frame)
            atomic_write_json(args.output.with_suffix(".meta.json"), {
                "source_sha256": source_digest, "source_filename": args.source.name,
                "settings": pipeline.settings, "resolved_model_revision": pipeline.model.config._commit_hash,
                "resolved_model_code": type(pipeline.model).__module__,
                "token_count": len(frame), "sentence_count": int(frame.sentence_id.nunique()),
            })
            print(f"Verified token CSV: {args.output.resolve()}", flush=True)
    else:
        sentences = split_sentences(normalize_hebrew(text))
        if args.max_sentences:
            sentences = sentences[:args.max_sentences]
        print(f"Diagnosing {len(sentences)} original sentence(s).", flush=True)
        chunk_count = 0
        for sentence_id, sentence in enumerate(sentences, 1):
            if args.sentence_id is not None and sentence_id != args.sentence_id:
                continue
            chunks = pipeline._chunks(sentence, lambda: False)
            for chunk_index, chunk in enumerate(chunks, 1):
                chunk_count += 1
                with torch.inference_mode():
                    results = pipeline.model.predict([chunk], pipeline.tokenizer, padding="longest",
                                                     truncation=False, compute_syntax_mst=True, output_style="ud")
                previous = None
                for line in results[0]:
                    if not line or line.startswith("#"):
                        continue
                    columns = line.split("\t")
                    if len(columns) >= 6 and columns[0].isdigit() and not columns[1].strip("_"):
                        print(json.dumps({"sentence_id": sentence_id, "chunk_index": chunk_index,
                                          "chunk_sha256": hashlib.sha256(chunk.encode()).hexdigest(),
                                          "raw_ud_row": line, "preceding_ud_row": previous,
                                          "chunk_context": chunk[:220], "elapsed_seconds": round(time.monotonic() - start, 2)},
                                         ensure_ascii=False, indent=2), flush=True)
                        with torch.inference_mode():
                            json_output = pipeline.model.predict([chunk], pipeline.tokenizer, padding="longest",
                                                                 truncation=False, compute_syntax_mst=True, output_style="json")
                        relevant = [token for token in json_output[0]["tokens"]
                                    if token.get("lex") == columns[2] or token.get("seg", [None])[-1] == ""]
                        print("Relevant JSON tokens: " + json.dumps(relevant, ensure_ascii=False), flush=True)
                        assert hashlib.sha256(args.source.read_bytes()).hexdigest() == source_digest
                        return 0
                    previous = line
            if sentence_id % 10 == 0:
                print(f"Checked sentences={sentence_id} chunks={chunk_count} elapsed={time.monotonic() - start:.1f}s", flush=True)
        print(f"No empty reconstructed UD FORM found in {chunk_count} chunks.", flush=True)
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != source_digest:
        raise RuntimeError("Source changed externally while the read-only smoke check ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
