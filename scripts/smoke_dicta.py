"""Opt-in integration check against the published Dicta tiny joint model.

This downloads model weights on first use. Ordinary pytest runs stay offline.
Run from the project root: ``python scripts/smoke_dicta.py --revision <commit>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

# Permit execution both from the project root and by absolute script path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nlp.cache import validate_token_frame
from app.nlp.pipeline import DictaPipeline, TOKEN_COLUMNS
from app.utils.io import atomic_write_csv

SAMPLE_TEXT = (
    "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות "
    "והחל לפרסם מאמרים הומוריסטיים. הוא קרא בספרו וכתב מכתב."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Hugging Face model revision or commit hash.")
    parser.add_argument("--code-revision", help="Optional commit hash for Dicta's shared modeling-code repository.")
    parser.add_argument("--output", type=Path, help="Optional destination for the validated token CSV.")
    args = parser.parse_args()

    print("Loading Dicta tiny joint model on CPU...", flush=True)
    settings = {"revision": args.revision, "device": "cpu"}
    if args.code_revision is not None:
        settings["code_revision"] = args.code_revision
    pipeline = DictaPipeline(settings)
    pipeline.load()
    frame = pipeline.process(SAMPLE_TEXT, progress=lambda value: print(f"Processing: {value}%", flush=True))
    frame = validate_token_frame(frame)
    if frame.columns.tolist() != TOKEN_COLUMNS:
        raise RuntimeError("Dicta integration returned unexpected CSV columns.")
    if frame.empty or frame.token_id.tolist() != list(range(1, len(frame) + 1)):
        raise RuntimeError("Dicta integration returned empty or nonsequential tokens.")
    if set(frame.sentence_id) != {1, 2}:
        raise RuntimeError("The two sample sentences did not retain separate sentence IDs.")

    # These are real model predictions, not a fabricated tokenizer fallback.
    prefixes = set(zip(frame.word, frame.pos_tag))
    if not {("ב", "ADP"), ("ו", "CCONJ")} <= prefixes:
        raise RuntimeError("Dicta UD output did not preserve segmented Hebrew prefixes.")
    pronouns = frame.loc[frame.pos_tag.eq("PRON")]
    if not any("_ud_form" in json.loads(value) for value in pronouns.morph_features):
        raise RuntimeError("Dicta UD output did not preserve an attached pronominal suffix.")
    if not any(frame.word.ne(frame.lemma)):
        raise RuntimeError("Dicta integration did not produce any lemmatized inflections.")

    details = {
        "model_name": pipeline.settings["model_name"],
        "requested_revision": args.revision,
        "resolved_revision": getattr(pipeline.model.config, "_commit_hash", None),
        "requested_code_revision": args.code_revision,
        "resolved_model_code": type(pipeline.model).__module__,
        "device": "cpu",
        "torch": version("torch"),
        "transformers": version("transformers"),
        "token_count": len(frame),
        "sentence_count": frame.sentence_id.nunique(),
        "pos_categories": sorted(frame.pos_tag.unique().tolist()),
    }
    print(json.dumps(details, ensure_ascii=False, indent=2), flush=True)
    print(frame.to_string(index=False), flush=True)
    if args.output is not None:
        atomic_write_csv(args.output, frame)
        print(f"Token CSV: {args.output.resolve()}", flush=True)
    print("Dicta integration smoke check passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
