"""Unit tests for the cost router (SentimentService.predict): when the cascade escalates TF-IDF -> LLM.

These exercise the decision directly, without the HTTP layer or any real model, so each row's probabilities can
differ (the API-level tests use a tiled stub where every row is identical).
"""
import numpy as np
import pytest

from finarena.models.sentiment import SentimentService

# Label order is [Bearish, Bullish, Neutral]; a "confident" row peaks above the threshold, an "uncertain" one below.
CONFIDENT = (0.95, 0.03, 0.02)
UNCERTAIN = (0.40, 0.30, 0.30)
LLM_ANSWER = (0.10, 0.80, 0.10)  # the stub LLM always answers "Bullish"


class Recorder:
    """Returns a preset probability per text (or a constant) and records which texts each tier was asked to score."""

    def __init__(self, name, table=None, constant=None):
        self.name, self.table, self.constant, self.calls = name, table or {}, constant, []

    def proba(self, texts):
        self.calls.append(list(texts))
        rows = [self.constant if self.constant is not None else self.table[t] for t in texts]
        return np.array(rows, dtype=float)


def service(fast_table, threshold=0.8, with_llm=True):
    fast = Recorder("fast", table=fast_table)
    accurate = Recorder("llm", constant=LLM_ANSWER) if with_llm else None
    return SentimentService(fast, accurate, threshold), fast, accurate


def test_auto_escalates_only_uncertain_rows_and_passes_them_in_order():
    svc, fast, accurate = service({"hi": CONFIDENT, "meh": UNCERTAIN, "nope": UNCERTAIN})
    out = svc.predict(["hi", "meh", "nope"], "auto")

    assert fast.calls == [["hi", "meh", "nope"]]        # fast scores the whole batch once
    assert accurate.calls == [["meh", "nope"]]          # only the uncertain rows reach the LLM, in order
    assert [r["escalated"] for r in out] == [False, True, True]
    assert [r["model"] for r in out] == ["fast", "llm", "llm"]


def test_confident_rows_keep_the_fast_model_and_its_probabilities():
    svc, fast, accurate = service({"hi": CONFIDENT, "meh": UNCERTAIN})
    out = svc.predict(["hi", "meh"], "auto")

    assert out[0]["model"] == "fast" and out[0]["label"] == "Bearish"
    assert out[0]["confidence"] == pytest.approx(0.95)                 # not overwritten by the LLM
    assert out[1]["model"] == "llm" and out[1]["label"] == "Bullish"   # escalated row takes the LLM's answer


def test_probability_exactly_at_threshold_does_not_escalate():
    # Strict `<`: a top probability equal to the threshold stays on the fast model.
    svc, fast, accurate = service({"edge": (0.8, 0.1, 0.1)}, threshold=0.8)
    out = svc.predict(["edge"], "auto")

    assert accurate.calls == [] and out[0]["escalated"] is False and out[0]["model"] == "fast"


def test_fast_strategy_never_calls_the_llm():
    svc, fast, accurate = service({"meh": UNCERTAIN})
    out = svc.predict(["meh"], "fast")

    assert accurate.calls == [] and out[0]["escalated"] is False and out[0]["model"] == "fast"


def test_accurate_strategy_uses_only_the_llm_and_marks_nothing_escalated():
    svc, fast, accurate = service({"hi": CONFIDENT})
    out = svc.predict(["hi"], "accurate")

    assert fast.calls == [] and accurate.calls == [["hi"]]
    assert out[0]["model"] == "llm" and out[0]["escalated"] is False


def test_accurate_without_an_llm_raises_lookuperror():
    svc, _, _ = service({"hi": CONFIDENT}, with_llm=False)
    with pytest.raises(LookupError):
        svc.predict(["hi"], "accurate")


def test_auto_degrades_to_fast_when_no_llm_is_loaded():
    svc, fast, _ = service({"meh": UNCERTAIN}, with_llm=False)
    out = svc.predict(["meh"], "auto")

    assert out[0]["model"] == "fast" and out[0]["escalated"] is False and fast.calls == [["meh"]]
