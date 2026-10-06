"""Expected corpus limits are validation messages; actual crashes retain traces."""

import logging

from app.gui.workers import AnalysisWorker


def test_expected_analysis_error_is_a_validation_signal(qtbot, caplog):
    worker = AnalysisWorker({}, "PCA 2D", {})
    errors, results = [], []
    worker.error.connect(errors.append)
    worker.completed.connect(results.append)
    with caplog.at_level(logging.INFO, logger="app.gui.workers"):
        with qtbot.waitSignal(worker.validation_error, timeout=3000) as signal:
            worker.start()
        assert worker.wait(3000)
    assert signal.args[0]
    assert not errors and not results
    assert not any(record.exc_info for record in caplog.records)


def test_unexpected_analysis_error_keeps_diagnostic_trace(qtbot, monkeypatch, caplog):
    def broken_analysis(*args):
        raise RuntimeError("Unexpected numerical failure")

    monkeypatch.setattr("app.gui.workers.run_analysis", broken_analysis)
    worker = AnalysisWorker({}, "MFW", {})
    validations = []
    worker.validation_error.connect(validations.append)
    with qtbot.waitSignal(worker.error, timeout=3000) as signal:
        worker.start()
    assert worker.wait(3000)
    assert signal.args == ["Unexpected numerical failure"]
    assert not validations
    assert any(record.exc_info for record in caplog.records)
