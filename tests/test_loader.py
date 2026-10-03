import pytest

from finarena.config import Settings
from finarena.loader import load_registry


def test_require_llm_fails_startup_when_adapter_is_unreadable(tmp_path, monkeypatch):
    import joblib
    from sklearn.dummy import DummyClassifier

    d = tmp_path
    joblib.dump(DummyClassifier().fit([[0], [1], [2]], [0, 1, 2]), d / "sentiment_tfidf.joblib")
    (d / "sentiment_lora").mkdir()  # exists but holds no adapter -> loading must fail
    s = Settings(env="dev", artifact_dir=d, require_llm=True)
    with pytest.raises((ImportError, OSError, ValueError)):  # ImportError when torch is not installed (CI)
        load_registry(s)
    s2 = Settings(env="dev", artifact_dir=d, require_llm=False)
    reg = load_registry(s2)  # degraded but alive
    assert reg.sentiment is not None and reg.sentiment.accurate is None


def test_missing_llm_dependencies_log_one_warning_line_not_a_traceback(tmp_path, monkeypatch, caplog):
    import logging
    import sys

    import joblib
    from sklearn.dummy import DummyClassifier

    joblib.dump(DummyClassifier().fit([[0], [1], [2]], [0, 1, 2]), tmp_path / "sentiment_tfidf.joblib")
    (tmp_path / "sentiment_lora").mkdir()
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError
    with caplog.at_level(logging.WARNING, logger="finarena"):
        reg = load_registry(Settings(env="dev", artifact_dir=tmp_path))
    assert reg.sentiment.accurate is None
    assert [r.levelno for r in caplog.records] == [logging.WARNING] and caplog.records[0].exc_info is None
