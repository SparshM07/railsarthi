"""Unit tests for local provider protection and abuse safeguards."""

import unittest

from backend.runtime import SlidingWindowRateLimiter, TTLCache


class TTLCacheTests(unittest.TestCase):
    def test_reuses_value_within_ttl(self):
        cache = TTLCache()
        calls = 0

        def fetch():
            nonlocal calls
            calls += 1
            return {"value": calls}

        self.assertEqual(cache.get_or_set("key", 60, fetch), {"value": 1})
        self.assertEqual(cache.get_or_set("key", 60, fetch), {"value": 1})
        self.assertEqual(calls, 1)
        self.assertEqual(cache.snapshot()["hits"], 1)


class RateLimiterTests(unittest.TestCase):
    def test_rejects_requests_over_limit(self):
        limiter = SlidingWindowRateLimiter(limit=2)
        self.assertTrue(limiter.allow("client"))
        self.assertTrue(limiter.allow("client"))
        self.assertFalse(limiter.allow("client"))


if __name__ == "__main__":
    unittest.main()
