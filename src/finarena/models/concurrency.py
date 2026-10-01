"""One mechanism for models that cannot be called from two threads at once (torch on Apple MPS aborts the process)."""
import threading


class ModelBusy(RuntimeError):
    """Raised when a serialized model could not be acquired within the wait budget."""


class Serialized:
    """Proxy that runs every method call on ``model`` one at a time.

    Callers wait up to ``wait_seconds`` for their turn, then get ModelBusy, so overload degrades to a fast 503
    instead of an unbounded queue. Non-callable attributes (e.g. ``name``) pass straight through.
    """

    def __init__(self, model, wait_seconds=20.0):
        self._model, self._wait, self._lock = model, wait_seconds, threading.Lock()

    def __getattr__(self, attr):
        target = getattr(self._model, attr)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            if not self._lock.acquire(timeout=self._wait):
                raise ModelBusy(f"{type(self._model).__name__} is busy; retry shortly")
            try:
                return target(*args, **kwargs)
            finally:
                self._lock.release()

        return call
