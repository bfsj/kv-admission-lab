import importlib.util
import unittest
from collections import deque
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("nanovllm"), "optional nano-vLLM environment")
class ReplicaTests(unittest.TestCase):
    def test_affinity_breaks_near_ties_and_avoids_backlog(self):
        from replica_bench import destination
        self.assertEqual(destination("affinity", 0, [2, 3], [False, True]), 1)
        self.assertEqual(destination("affinity", 0, [2, 4], [False, True]), 0)
        self.assertEqual(destination("least", 0, [2, 3], [False, True]), 0)

    def test_guard_counts_pinned_prefix_only_once(self):
        from nanovllm.engine.sequence import Sequence
        from replica_bench import GuardBlocks, PAGE
        running = deque()
        bm = GuardBlocks(SimpleNamespace(num_kvcache_blocks=4), running)
        first = Sequence([8] * PAGE + [7] * PAGE)
        bm.allocate(first, 0)
        first.num_scheduled_tokens = 2 * PAGE
        bm.hash_blocks(first)
        first.append_token(9)
        running.append(first)
        matching = Sequence([8] * PAGE + [1])
        cold = Sequence([2] * PAGE + [1])
        self.assertEqual(bm.can_allocate(matching), 1)
        self.assertEqual(bm.can_allocate(cold), -1)
