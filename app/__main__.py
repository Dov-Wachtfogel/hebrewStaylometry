"""Application bootstrap; expensive NLP dependencies are loaded on demand."""

import argparse
import logging
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hebrew Stylometry Studio")
    parser.add_argument("--project", type=Path, help="Open an existing project folder")
    parser.add_argument("--model-cache", type=Path, help="Hugging Face download cache folder")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    local_cache = Path(__file__).resolve().parents[1] / ".model_cache"
    if args.model_cache:
        os.environ["HF_HOME"] = str(args.model_cache.expanduser().resolve())
    elif local_cache.is_dir():
        os.environ.setdefault("HF_HOME", str(local_cache))

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    application = QApplication(sys.argv[:1])
    application.setApplicationName("Hebrew Stylometry Studio")
    application.setOrganizationName("Hebrew Stylometry")
    application.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
    application.setStyle("Fusion")
    window = MainWindow()
    window.show()
    if args.project:
        window.open_project(args.project)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
