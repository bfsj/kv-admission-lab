import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec("nanovllm"), "optional nano-vLLM environment")
class StudyTests(unittest.TestCase):
    def test_deadline_boundaries_and_four_counters(self):
        from replica_bench import verdict
        for t, g, name in [(250,100,"met"),(251,100,"ttft_only"),(250,101,"gap_only"),(251,101,"both")]:
            self.assertEqual(verdict(t,g,250,100), name)

    def test_capacity_requires_every_repeat(self):
        from capacity_study import stable
        cases = [dict(replicas=1, rate=24, slo_met=n, completed=300) for n in (285,300,284)]
        self.assertTrue(stable(cases[:2],1,24))
        self.assertFalse(stable(cases,1,24))
        self.assertFalse(stable(cases,2,24))

    def test_router_prices_prompt_backlog_and_verified_prefix(self):
        from nanovllm.engine.sequence import Sequence
        from replica_bench import route_request, PAGE
        prefix = [1]*PAGE
        bm = SimpleNamespace(hash_to_block_id={42:0}, blocks=[SimpleNamespace(token_ids=prefix)])
        empty = SimpleNamespace(hash_to_block_id={}, blocks=[])
        long = Sequence([2]*700)
        short = Sequence([2]*10)
        lanes = [SimpleNamespace(block_manager=bm, running=[], waiting=[long]),
                 SimpleNamespace(block_manager=empty, running=[], waiting=[short])]
        def pick():
            return route_request("work",1,[1,1],lanes,prefix,42,PAGE+10)
        self.assertEqual(pick(),1)
        lanes[0].waiting = [short]
        self.assertEqual(pick(),0)
        bm.blocks[0].token_ids = [3]*PAGE
        self.assertEqual(pick(),1)
