"""Tests for thread safety of Counter, Histogram, and Gauge."""

import threading
from concurrent.futures import ThreadPoolExecutor, wait

from qe.runtime.metrics import Counter, Gauge, Histogram

# ── Counter thread safety ─────────────────────────────────────────────────


class TestCounterThreadSafety:
    def test_concurrent_increments(self):
        """10 threads x 1000 increments => value == 10000."""
        counter = Counter(name="test_counter")

        def worker():
            for _ in range(1000):
                counter.inc()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == 10_000

    def test_concurrent_inc_with_step(self):
        """10 threads x 500 increments of 2 => value == 10000."""
        counter = Counter(name="test_step")

        def worker():
            for _ in range(500):
                counter.inc(2)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert counter.value == 10_000


# ── Histogram thread safety ──────────────────────────────────────────────


class TestHistogramThreadSafety:
    def test_concurrent_observations_count(self):
        """10 threads x 1000 observations => _count == 10000."""
        hist = Histogram(name="test_hist")

        def worker():
            for i in range(1000):
                hist.observe(float(i % 100))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert hist._count == 10_000

    def test_concurrent_observations_no_data_loss(self):
        """All bucket counts should sum to _count."""
        hist = Histogram(name="test_hist_buckets")

        def worker():
            for i in range(1000):
                hist.observe(float(i % 100))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_in_buckets = sum(hist._counts)
        assert total_in_buckets == hist._count

    def test_concurrent_observations_sum(self):
        """Sum should equal expected total when all threads observe same value."""
        hist = Histogram(name="test_hist_sum")

        def worker():
            for _ in range(1000):
                hist.observe(1.0)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert hist._sum == 10_000.0


# ── Gauge thread safety ──────────────────────────────────────────────────


class TestGaugeThreadSafety:
    def test_concurrent_inc_dec_deterministic(self):
        """5 threads inc 1000 times, 5 threads dec 1000 times => net 0."""
        gauge = Gauge(name="test_gauge")

        def inc_worker():
            for _ in range(1000):
                gauge.inc(1.0)

        def dec_worker():
            for _ in range(1000):
                gauge.dec(1.0)

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=inc_worker))
            threads.append(threading.Thread(target=dec_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert gauge.value == 0.0

    def test_concurrent_inc_only(self):
        """10 threads x 1000 increments => value == 10000."""
        gauge = Gauge(name="test_gauge_inc")

        def worker():
            for _ in range(1000):
                gauge.inc(1.0)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(worker) for _ in range(10)]
            wait(futures)

        assert gauge.value == 10_000.0

    def test_concurrent_set(self):
        """Concurrent set() calls should not corrupt state."""
        gauge = Gauge(name="test_gauge_set")

        def worker(val: float):
            for _ in range(1000):
                gauge.set(val)

        threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert 0.0 <= gauge.value <= 9.0
