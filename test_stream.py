import asyncio
import importlib.util
import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace


def has_runtime():
    return all(importlib.util.find_spec(m) for m in ("aiohttp", "nanovllm", "tokenizers"))


@unittest.skipUnless(has_runtime(), "optional serving environment")
class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp.test_utils import TestClient, TestServer
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        from stream_api import application, Service
        cfg = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=1024,
                              eos=3, num_kvcache_blocks=4, kvcache_block_size=256, max_model_len=768)
        self.steps = asyncio.Queue()
        async def infer(lane, batch, prefill):
            future = asyncio.get_running_loop().create_future()
            self.steps.put_nowait((batch, future))
            return await future
        self.s = Service(cfg, 1, infer, limit=2)
        raw = Tokenizer(models.WordLevel({"[UNK]":0, "hello":1, "next":2, "[EOS]":3, "Ã":4, "©":5},
                                        unk_token="[UNK]"))
        raw.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token="[EOS]", unk_token="[UNK]")
        self.tok = tok
        self.client = TestClient(TestServer(application(self.s, tok)))
        await self.client.start_server()
        self.pending = []

    async def asyncTearDown(self):
        for future in self.pending:
            if not future.done():
                future.set_exception(RuntimeError("test ended"))
        await self.client.close()

    async def step(self):
        batch, future = await asyncio.wait_for(self.steps.get(), 3)
        self.pending.append(future)
        return batch, future

    async def eventually(self, condition):
        for _ in range(200):
            if condition():
                return
            await asyncio.sleep(0.005)
        self.fail("condition did not become true")

    async def post(self, maximum=3):
        return await self.client.post("/generate", json={"prompt":"hello", "max_tokens":maximum})

    async def test_sse_order_and_length(self):
        r = await self.post(2)
        for _ in range(2):
            batch, future = await self.step()
            future.set_result([2]*len(batch))
        lines = (await r.text()).splitlines()
        self.assertEqual([x for x in lines if x.startswith("event:")],
                         ["event: token","event: token","event: done"])
        values = [json.loads(x[6:]) for x in lines if x.startswith("data: ")]
        self.assertEqual(values[-1], {"reason":"length","generated":2,"text":"next next","text_tail":""})
        self.assertEqual(self.s.metrics()["replicas"][0]["free"], 4)

    async def test_eos_and_validation(self):
        for bad in [[], {}, {"prompt":"  "}, {"prompt":"x","max_tokens":True},
                    {"prompt":"x","max_tokens":257}, {"prompt":"hello "*769}]:
            r = await self.client.post("/generate", json=bad)
            self.assertEqual(r.status, 400)
        bad_json = await self.client.post("/generate", data=b"{")
        self.assertEqual(bad_json.status, 400)
        r = await self.post()
        _, future = await self.step()
        future.set_result([3])
        self.assertIn('"reason": "eos"', await r.text())
        self.assertEqual(self.s.completed, 1)

    async def test_utf8_split_across_tokens_and_final_incomplete_byte(self):
        from tokenizers.decoders import ByteLevel
        self.tok.backend_tokenizer.decoder = ByteLevel()
        for ids, expected, tail in (([4,5],"é",""),([4],"�","�")):
            response = await self.post(len(ids))
            for value in ids:
                _, future = await self.step()
                future.set_result([value])
            values = [json.loads(x[6:]) for x in (await response.text()).splitlines()
                      if x.startswith("data: ")]
            self.assertEqual(values[-1]["text"],expected)
            self.assertEqual(values[-1]["text_tail"],tail)
            self.assertEqual("".join(v["text"] for v in values[:-1])+tail,expected)

    async def test_disconnect_inflight_defers_release_and_queue_limit_recovers(self):
        r = await self.post()
        _, future = await self.step()
        queued = await self.post()
        full = await self.post()
        self.assertEqual(full.status, 429)
        queued.close()
        await self.eventually(lambda: self.s.cancelled == 1)
        self.assertEqual(self.s.metrics()["replicas"][0]["waiting"], 0)
        r.close()
        await self.eventually(lambda: self.s.cancelled == 2)
        self.assertEqual(self.s.metrics()["replicas"][0]["pinned"], 1)
        future.set_result([2])
        await self.eventually(lambda: not self.s.active)
        self.assertEqual(self.s.metrics()["replicas"][0]["free"], 4)
        again = await self.post(1)
        _, future = await self.step()
        future.set_result([2])
        self.assertIn("event: done", await again.text())

    async def test_shared_page_survives_cancellation(self):
        x = self.s.submit([1]*257, 8)
        _, future = await self.step()
        y = self.s.submit([1]*256+[2], 1)
        future.set_result([2])
        batch, second = await self.step()
        self.assertEqual(batch, [y])
        self.assertEqual(x.block_table[0], y.block_table[0])
        self.s.cancel(x.rid)
        self.s.cancel(x.rid)
        self.assertEqual(self.s.cancelled, 1)
        self.assertEqual(self.s.metrics()["replicas"][0]["pinned"], 2)
        second.set_result([2])
        await self.eventually(lambda: not self.s.active)
        self.assertEqual(self.s.metrics()["replicas"][0]["free"], 4)

    async def test_worker_error_terminates_streams_and_rejects_new_work(self):
        r = await self.post()
        _, future = await self.step()
        queued = await self.post()
        future.set_exception(EOFError())
        for response in (r, queued):
            self.assertIn("event: error", await response.text())
        stats = await (await self.client.get("/metrics")).json()
        self.assertFalse(stats["healthy"])
        self.assertEqual(stats["active"], 0)
        self.assertEqual(stats["replicas"][0]["free"], 4)
        self.assertEqual((await self.post()).status, 503)

    async def test_independent_replica_finishes_while_first_is_blocked(self):
        from stream_api import Service
        service = Service(self.s.config, 2, self.s.infer, router="rr")
        async with asynccontextmanager(service.lifecycle)(None):
            x = service.submit([1]*257, 1)
            y = service.submit([2]*257, 1)
            steps = {}
            for _ in range(2):
                batch, future = await self.step()
                steps[batch[0].rid] = future
            steps[y.rid].set_result([2])
            await self.eventually(lambda: y.rid not in service.active)
            self.assertEqual(service.metrics()["replicas"][1]["free"],4)
            self.assertIn(x.rid,service.inflight)
            service.cancel(x.rid)
            steps[x.rid].set_result([2])
            await self.eventually(lambda: not service.active)
            self.assertEqual([s["pinned"] for s in service.metrics()["replicas"]],[0,0])
