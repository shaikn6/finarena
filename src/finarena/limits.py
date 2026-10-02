"""Per-client rate limiting: a token bucket per key, in memory, safe to call from many threads."""
import threading
import time
from collections import OrderedDict


class RateLimiter:
    """Allows a sustained ``per_minute`` requests per client with bursts up to ``per_minute``. ``per_minute=0`` disables it.

    Memory is hard-capped at ``MAX_KEYS`` clients: when full, the least recently seen client is forgotten in O(1),
    so a flood of distinct keys can neither grow memory nor slow the request path.

    State is per process: behind several replicas each enforces its own budget, so size the limit accordingly or
    enforce it at the gateway as well.
    """

    MAX_KEYS = 10_000

    def __init__(self, per_minute, clock=time.monotonic):
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0  # tokens per second
        self._clock, self._lock = clock, threading.Lock()
        self._buckets = OrderedDict()  # key -> (tokens, last_seen), least recently used first

    def check(self, key):
        """Return (allowed, retry_after_seconds)."""
        if self.capacity <= 0:
            return True, 0.0
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.pop(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            if len(self._buckets) > self.MAX_KEYS:
                self._buckets.popitem(last=False)
        return allowed, 0.0 if allowed else (1.0 - tokens) / self.rate
