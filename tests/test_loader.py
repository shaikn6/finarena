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
