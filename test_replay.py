import unittest

from admission import Pool, State, choose
from replay import Config, run
from workload import Request, make_trace


class ReplayTests(unittest.TestCase):
    def test_shared_page_survives_one_release_and_can_be_evicted(self):
        x, y, z = [State(Request(i, 0, "hot", (p, p, i), 2)) for i, p in enumerate([9, 9, 7])]
        pool = Pool(3, 2)
        self.assertEqual(pool.attach(x), 0)
        self.assertEqual(pool.attach(y), 2)
        self.assertEqual(x.pages[0], y.pages[0])
        pool.release(x)
        self.assertEqual(pool.available(), 1)
        pool.release(y)
        pool.attach(z)
        self.assertEqual(pool.available(), 1)
        old_page = pool.prefixes[pool.prefix(x)]
        self.assertEqual(pool.take(), old_page)
        self.assertNotIn(pool.prefix(x), pool.prefixes)

    def test_aged_request_cannot_take_decode_reserve(self):
        active, cold = [State(Request(i, 0, "cold", (i, i, 4, 4), 3)) for i in (1, 2)]
        pool = Pool(4, 2)
        pool.attach(active)
        active.generated.append(42)
        self.assertIsNone(choose([cold], [active], pool, "guarded", 8, 10, 100))

    def test_aging_promotes_cold_head(self):
        cold, hot = [State(Request(i, 0, c, (i, i, 4), 2)) for i, c in enumerate(["cold", "hot"])]
        pool = Pool(4, 2)
        pool.attach(hot)
        pool.release(hot)
        self.assertIs(choose([cold, hot], [], pool, "guarded", 2, 10, 0), hot)
        self.assertIs(choose([cold, hot], [], pool, "guarded", 2, 10, 10), cold)

    def test_pressure_and_determinism(self):
        trace = make_trace(90)
        opts = Config(pages=8, queue_timeout_ms=10000)
        for policy in ("fcfs", "cache", "guarded"):
            stats, states = run(trace, opts, policy)
            self.assertEqual(stats, run(trace, opts, policy)[0])
            self.assertEqual(stats["completed"], len(trace))
            self.assertLessEqual(stats["peak_pinned_pages"], opts.pages)
            self.assertTrue(all(len(s.generated) == s.request.output_tokens for s in states))
        self.assertGreater(run(trace, opts, "fcfs")[0]["recomputed_tokens"], 0)

    def test_rejection_and_queue_deadline(self):
        trace = make_trace(10)
        trace.append(Request(999, 0, "cold", (1,) * 500, 2))
        stats, _ = run(trace, Config(slots=1, queue_timeout_ms=1), "fcfs")
        self.assertEqual(stats["rejected"], 1)
        self.assertGreater(stats["timeouts"], 0)
        self.assertEqual(len(trace), sum(stats[k] for k in ["completed", "timeouts", "rejected"]))
