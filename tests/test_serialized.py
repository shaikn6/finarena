import threading
import time

import numpy as np
import pytest

from finarena.models.concurrency import ModelBusy, Serialized


class Probe:
    """Detects overlapping calls, the condition that aborts torch on Apple MPS."""
    name = "probe"

    def __init__(self, delay=0.02):
        self.active, self.max_active, self.delay, self._g = 0, 0, delay, threading.Lock()

    def proba(self, texts):
        with self._g:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._g:
            self.active -= 1
        return np.ones((len(texts), 3)) / 3


def test_concurrent_callers_never_overlap_inside_the_model():
    probe = Probe()
    m = Serialized(probe, wait_seconds=10)
    threads = [threading.Thread(target=m.proba, args=(["x"],)) for _ in range(16)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert probe.max_active == 1


def test_waiting_longer_than_budget_raises_model_busy_and_recovers():
    probe = Probe(delay=0.3)
    m = Serialized(probe, wait_seconds=0.05)
    t = threading.Thread(target=m.proba, args=(["x"],))
    t.start()
    time.sleep(0.05)
    with pytest.raises(ModelBusy):
        m.proba(["y"])
    t.join()
    assert m.proba(["z"]).shape == (1, 3)  # lock released: the model serves again


def test_exception_inside_model_releases_the_lock():
    class Boom:
        name = "boom"

        def proba(self, texts):
            raise ValueError("bad")

    m = Serialized(Boom(), wait_seconds=0.1)
    for _ in range(2):
        with pytest.raises(ValueError):
            m.proba(["x"])
