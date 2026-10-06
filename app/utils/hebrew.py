"""Keep Hebrew logical text intact; apply bidi only at plotting boundaries."""

from __future__ import annotations

import unicodedata


def normalize_hebrew(text: str, strip_niqqud: bool = True) -> str:
    """Normalize Unicode and optionally remove Hebrew vowel/cantillation marks.

    Punctuation, spaces, Hebrew letters, and combining marks in other scripts
    are preserved. Qt widgets must receive this logical text directly.
    """
    normalized = unicodedata.normalize("NFC", text)
    if not strip_niqqud:
        return normalized
    return "".join(
        character
        for character in normalized
        if not (
            "\u0591" <= character <= "\u05c7"
            and unicodedata.category(character).startswith("M")
        )
    )


def display_hebrew(text: object) -> str:
    """Return a label appropriate for the installed Matplotlib text renderer.

    Do not use this for Qt, CSV, NLP, or other systems supporting bidi natively.
    Matplotlib 3.11+ performs its own bidirectional layout. Pre-reordering there
    would reverse Hebrew a second time. Older versions need visual-order text.
    Hebrew requires bidirectional ordering, not Arabic letter reshaping.
    """
    import matplotlib
    from packaging.version import Version

    logical_text = str(text)
    if Version(matplotlib.__version__).release >= (3, 11):
        return logical_text

    from bidi.algorithm import get_display

    return get_display(logical_text)
