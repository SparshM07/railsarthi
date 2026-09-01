"""Small, dependency-free runtime safeguards for a single API process."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class _CachedValue:
    expires_at: float
    value: object


class TTLCache:
    """Thread-safe bounded TTL cache; intended for provider reads only."""

    def __init__(self, max_entries: int = 512):
        self.max_entries = max_entries
        self._values: dict[str, _CachedValue] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get_or_set(self, key: str, ttl_seconds: float, fetch: Callable[[], T]) -> T:
        now = monotonic()
        with self._lock:
            cached = self._values.get(key)
            if cached and cached.expires_at > now:
                self.hits += 1
                return cached.value  # type: ignore[return-value]
            self.misses += 1

        # Do network work outside the lock. A rare duplicated request is safer
        # than blocking every provider call behind a slow network response.
        value = fetch()
        with self._lock:
            if len(self._values) >= self.max_entries:
                expired = [name for name, item in self._values.items() if item.expires_at <= now]
                for name in expired:
                    self._values.pop(name, None)
                if len(self._values) >= self.max_entries:
                    self._values.pop(next(iter(self._values)))
            self._values[key] = _CachedValue(now + ttl_seconds, value)
        return value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._values), "hits": self.hits, "misses": self.misses}


class SlidingWindowRateLimiter:
    """Thread-safe, in-memory per-key rate limiter."""

    def __init__(self, limit: int, window_seconds: float = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = monotonic()
        with self._lock:
            timestamps = self._requests.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.limit:
                return False
            timestamps.append(now)
            return True


class RuntimeMetrics:
    """Counters deliberately kept aggregate: no customer or train data."""

    def __init__(self):
        self._counters: Counter[str] = Counter()
        self._lock = Lock()

    def increment(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)
