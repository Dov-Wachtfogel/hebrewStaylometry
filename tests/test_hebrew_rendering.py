"""Regressions for native Matplotlib bidi versus legacy visual-order labels."""

import matplotlib
import pytest

from app.utils.hebrew import display_hebrew


@pytest.mark.parametrize("version", ["3.11.0", "3.11.2", "3.11.0rc1", "3.12.0.dev0", "4.0.0"])
@pytest.mark.parametrize("label", ["כהן", "לוי_המסע_1900.txt", "Delta: רחל_המכתב_1920.txt", "POS distribution"])
def test_native_renderer_receives_logical_text(monkeypatch, version, label):
    monkeypatch.setattr(matplotlib, "__version__", version)
    assert display_hebrew(label) == label


@pytest.mark.parametrize("version", ["3.8.0", "3.9.4", "3.10.9"])
def test_legacy_renderer_receives_visual_order(monkeypatch, version):
    monkeypatch.setattr(matplotlib, "__version__", version)
    assert display_hebrew("כהן") == "ןהכ"
    assert display_hebrew("שלום") == "םולש"
    assert display_hebrew("PCA 2D") == "PCA 2D"
