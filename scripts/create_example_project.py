"""Create a portable example corpus, optionally preprocess it using a QThread."""

import argparse
import sys
from pathlib import Path

# Allow running this script directly from any working directory.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from app.project import Project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New project folder")
    parser.add_argument("--process", action="store_true", help="Process texts with the real Dicta model in a QThread")
    args = parser.parse_args()
    project = Project.create(args.destination, name="Hebrew example corpus")
    project.settings["analysis"].update({"n_mfw": 100, "mattr_window": 30, "window_size": 30, "step_size": 10})
    project.import_texts(sorted((SOURCE_ROOT / "examples" / "texts").glob("*.txt")))
    project.import_metadata(SOURCE_ROOT / "examples" / "metadata.json")
    print(f"Created {project.root} with {len(project.texts)} texts.", flush=True)
    if not args.process:
        return 0

    from PySide6.QtCore import QCoreApplication
    from app.nlp.workers import ProcessingWorker

    application = QCoreApplication([])
    worker = ProcessingWorker(project)
    outcome = {"error": False, "processed": 0}
    worker.status.connect(lambda status: print(status, flush=True))
    worker.document_error.connect(lambda text_id, message: print(message, file=sys.stderr, flush=True))

    def completed(frames):
        outcome["processed"] = len(frames)

    def error(message):
        outcome["error"] = True
        print(message, file=sys.stderr, flush=True)

    worker.completed.connect(completed)
    worker.error.connect(error)
    worker.finished.connect(application.quit)
    worker.start()
    application.exec()
    worker.wait()
    return 1 if outcome["error"] or outcome["processed"] != len(project.texts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
