"""Loopback HTTP/SSE generation with cancellable nano-vLLM replica batches."""

import argparse
import asyncio
import itertools
import json
from time import monotonic

from aiohttp import web
from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer

from replica_bench import MeasuredScheduler, PAGE, receive, replicas, route_request


class Service:
    def __init__(self, config, size, infer, router="work", limit=128):
        self.config = config
        self.schedulers = [MeasuredScheduler(config, True) for _ in range(size)]
        self.wake = [asyncio.Event() for _ in range(size)]
        self.active = {}
        self.inflight = set()
        self.ids = itertools.count()
        self.tasks = []
        self.infer, self.router, self.limit = infer, router, limit
        self.closing = False
        self.failure = None
        self.cancelled = self.completed = 0

    def submit(self, ids, maximum):
        if self.failure is not None or self.closing:
            raise web.HTTPServiceUnavailable(text="worker unavailable")
        if len(self.active) >= self.limit:
            raise web.HTTPTooManyRequests(text="request queue is full")
        if not ids or len(ids)+maximum > min(self.config.max_model_len,
                                            PAGE*self.config.num_kvcache_blocks):
            raise web.HTTPBadRequest(text="prompt and output limit exceed context budget")
        rid = next(self.ids)
        prefix = ids[:PAGE]
        loads = [len(s.waiting)+len(s.running) for s in self.schedulers]
        lane = route_request(self.router, rid, loads, self.schedulers, prefix,
                             BlockManager.compute_hash(prefix), len(ids))
        seq = Sequence(ids, SamplingParams(max_tokens=maximum))
        seq.rid, seq.lane, seq.arrival = rid, lane, monotonic()
        seq.stream = asyncio.Queue(maxsize=maximum+2)
        seq.cancelled = False
        self.active[rid] = seq
        self.schedulers[lane].add(seq)
        self.wake[lane].set()
        return seq

    def discard(self, seq):
        scheduler = self.schedulers[seq.lane]
        for queue in (scheduler.waiting, scheduler.running):
            if seq in queue:
                queue.remove(seq)
        if seq.block_table:
            scheduler.block_manager.deallocate(seq)
        self.active.pop(seq.rid, None)

    def cancel(self, rid):
        seq = self.active.get(rid)
        if seq is None or seq.cancelled:
            return
        seq.cancelled = True
        self.cancelled += 1
        # The device may still be writing into the block table for this batch.
        if rid not in self.inflight:
            self.discard(seq)

    async def loop(self, lane):
        scheduler = self.schedulers[lane]
        try:
            while not self.closing and self.failure is None:
                if scheduler.is_finished():
                    self.wake[lane].clear()
                    await self.wake[lane].wait()
                    continue
                batch, prefill = scheduler.schedule()
                self.inflight.update(s.rid for s in batch)
                before = [s.num_completion_tokens for s in batch]
                ids = await self.infer(lane, batch, prefill)
                if len(ids) != len(batch):
                    raise RuntimeError("incomplete batch from worker")
                scheduler.postprocess(batch, ids, prefill)
                for seq, n in zip(batch, before):
                    self.inflight.discard(seq.rid)
                    if seq.cancelled:
                        self.discard(seq)
                        continue
                    if seq.num_completion_tokens > n:
                        seq.stream.put_nowait(("token", seq.last_token))
                    if seq.is_finished:
                        self.active.pop(seq.rid)
                        self.completed += 1
                        reason = "eos" if seq.last_token == scheduler.eos else "length"
                        seq.stream.put_nowait(("done", reason))
        except Exception as exc:
            self.failure = str(exc)
            for event in self.wake:
                event.set()
        finally:
            for seq in list(self.active.values()):
                if seq.lane == lane:
                    seq.stream.put_nowait(("error", "worker unavailable"))
                    self.inflight.discard(seq.rid)
                    self.discard(seq)

    def metrics(self):
        return dict(active=len(self.active), cancelled=self.cancelled, completed=self.completed,
                    healthy=self.failure is None and not self.closing, replicas=[
                        dict(waiting=len(s.waiting), running=len(s.running),
                             pinned=len(s.block_manager.used_block_ids),
                             free=len(s.block_manager.free_block_ids)) for s in self.schedulers])

    async def lifecycle(self, app):
        self.tasks = [asyncio.create_task(self.loop(i)) for i in range(len(self.schedulers))]
        try:
            yield
        finally:
            self.closing = True
            for e in self.wake:
                e.set()
            await asyncio.gather(*self.tasks)


def application(service, tokenizer):
    async def generate(request):
        try:
            body = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="expected a JSON object")
        if not isinstance(body, dict):
            raise web.HTTPBadRequest(text="expected a JSON object")
        prompt, maximum = body.get("prompt"), body.get("max_tokens", 32)
        if not isinstance(prompt, str) or not prompt.strip() or type(maximum) is not int or not 1 <= maximum <= 256:
            raise web.HTTPBadRequest(text="provide a prompt and an integer max_tokens in [1,256]")
        seq = service.submit(tokenizer.encode(prompt, add_special_tokens=False), maximum)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream; charset=utf-8",
                            "Cache-Control": "no-cache", "X-Request-ID": str(seq.rid)})
        try:
            await response.prepare(request)
            decoder = DecodeStream(skip_special_tokens=True)
            rendered = ""
            while True:
                event, value = await seq.stream.get()
                if event == "token":
                    chunk = decoder.step(tokenizer.backend_tokenizer, value) or ""
                    rendered += chunk
                    data = dict(token_id=value, text=chunk)
                elif event == "done":
                    final = tokenizer.decode(seq.completion_token_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
                    data = dict(reason=value, generated=seq.num_completion_tokens, text=final,
                                text_tail=final[len(rendered):])
                else:
                    data = dict(message=value)
                frame = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                await response.write(frame.encode())
                if event != "token":
                    break
            return response
        finally:
            service.cancel(seq.rid)

    async def metrics(request):
        return web.json_response(service.metrics())

    app = web.Application(client_max_size=128*1024)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/generate", generate)
    app.cleanup_ctx.append(service.lifecycle)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or min(args.devices) < 0 or args.pages < 4:
        parser.error("distinct GPU indices and at least four pages required")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cfg = Config(args.model, num_kvcache_blocks=args.pages, max_model_len=1024,
                 max_num_batched_tokens=4096, max_num_seqs=16)
    cfg.eos = tok.eos_token_id
    with replicas(args.devices, args.model, args.pages) as (conns, _):
        def exchange(i, batch, prefill):
            conns[i].send((batch, prefill))
            if not conns[i].poll(60):
                raise TimeoutError("worker batch timeout")
            return receive(conns[i], "tokens")
        async def infer(i, batch, prefill):
            return await asyncio.to_thread(exchange, i, batch, prefill)
        service = Service(cfg, len(conns), infer)
        web.run_app(application(service, tok), host="127.0.0.1", port=args.port,
                    handler_cancellation=True)


if __name__ == "__main__":
    main()
