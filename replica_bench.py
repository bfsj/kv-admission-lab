"""Open-loop, single-host benchmark for independent nano-vLLM replicas."""

import argparse
import hashlib
import json
import multiprocessing as mp
import random
from contextlib import contextmanager
from itertools import islice
from multiprocessing.connection import wait
from pathlib import Path
from time import monotonic, sleep

from nanovllm.config import Config
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

from nano_worker import worker
from replay import p95
from workload import Request, fingerprint


PAGE = 256


def traffic(count, seed, mixed, interval_ms):
    rng = random.Random(seed)
    trace = []
    for i in range(count):
        common = rng.random() < 0.70
        template = rng.randrange(3)
        hot = mixed and common
        prefix = (100 + template if hot else 1000 + i,) * PAGE
        suffix = tuple(rng.randrange(10, 100) for _ in range(rng.randint(244, 255)))
        trace.append(Request(i, i // 8 * interval_ms, "hot" if hot else "cold",
                             prefix + suffix, rng.randint(8, 16)))
    return trace


class GuardBlocks(BlockManager):
    def __init__(self, config, running):
        super().__init__(config.num_kvcache_blocks, PAGE)
        self.running = running

    def can_allocate(self, seq):
        hits = super().can_allocate(seq)
        reserve = sum(len(s) >= len(s.block_table) * PAGE for s in self.running)
        if hits < 0 or not reserve:
            return hits
        pinned, h = 0, -1
        for index in range(hits):
            h = self.compute_hash(seq.block(index), h)
            pinned += self.hash_to_block_id[h] in self.used_block_ids
        return hits if seq.num_blocks - pinned + reserve <= len(self.free_block_ids) else -1


class MeasuredScheduler(Scheduler):
    def __init__(self, config, guarded):
        super().__init__(config)
        self.guarded = guarded
        self.preemptions = 0
        if guarded:
            self.block_manager = GuardBlocks(config, self.running)

    def preempt(self, seq):
        self.preemptions += 1
        super().preempt(seq)

    def schedule(self):
        if self.guarded and self.waiting:
            head = self.waiting[0]
            if not head.block_table and monotonic() - head.arrival < 0.1:
                candidates = list(islice(self.waiting, 8))
                best = max(candidates, key=self.block_manager.can_allocate)
                self.waiting.remove(best)
                self.waiting.appendleft(best)
        return super().schedule()


def destination(mode, rid, loads, cached):
    if mode == "rr":
        return rid % len(loads)
    least = min(range(len(loads)), key=lambda i: (loads[i], (i-rid) % len(loads)))
    if mode == "affinity":
        matches = [i for i, hit in enumerate(cached) if hit and loads[i] <= loads[least] + 1]
        if matches:
            return min(matches, key=lambda i: (loads[i], (i-rid) % len(loads)))
    return least


def route_request(mode, rid, loads, schedulers, prefix, key, length):
    if mode in ("rr", "least"):
        return destination(mode, rid, loads, ())
    cached = []
    for scheduler in schedulers:
        bm = scheduler.block_manager
        page = bm.hash_to_block_id.get(key) if len(prefix) == PAGE else None
        cached.append(page is not None and bm.blocks[page].token_ids == prefix)
    if mode == "affinity":
        return destination(mode, rid, loads, cached)
    if mode != "work":
        raise ValueError(f"unknown router {mode}")
    # Fixed heuristic in equivalent input tokens. Only current state and declared
    # output caps are visible; the actual stopping position is unknown.
    scores = []
    for scheduler, hit in zip(schedulers, cached):
        work = length - PAGE * hit
        for seq in (*scheduler.waiting, *scheduler.running):
            prefill = max(0, len(seq) - seq.num_cached_tokens)
            decode = min(24, max(0, seq.max_tokens - seq.num_completion_tokens))
            work += prefill + 16 * decode
        scores.append(work)
    return min(range(len(scores)), key=lambda i: (scores[i], loads[i], (i-rid) % len(scores)))


def receive(conn, expected):
    kind, data = conn.recv()
    if kind != expected:
        raise RuntimeError(f"expected {expected}, received {kind}: {data}")
    return data


def verdict(first_ms, max_gap_ms, first_limit, gap_limit):
    if first_ms > first_limit:
        return "both" if max_gap_ms > gap_limit else "ttft_only"
    return "gap_only" if max_gap_ms > gap_limit else "met"


def replay(trace, conns, config, policy, routing, ttft_limit, gap_limit, rows=None):
    schedulers = [MeasuredScheduler(config, policy == "guarded") for _ in conns]
    loads = [0] * len(conns)
    totals = loads.copy()
    pending = {}
    events, outputs = {}, {}
    cursor = hits = recomputed = peak_batch = 0
    # Hash prompt keys before starting the service clock (tokenized ingress).
    key_hash = [BlockManager.compute_hash(r.prompt[:PAGE]) for r in trace]
    start = monotonic()
    while len(outputs) < len(trace):
        now = monotonic()
        if now - start > 300:
            raise TimeoutError("replay exceeded five minutes")
        while cursor < len(trace) and trace[cursor].arrival_ms <= (now-start)*1000:
            r = trace[cursor]
            idx = route_request(routing, r.id, loads, schedulers, list(r.prompt[:PAGE]),
                                key_hash[cursor], len(r.prompt))
            seq = Sequence(list(r.prompt), SamplingParams(max_tokens=r.output_tokens, ignore_eos=True))
            seq.rid = r.id
            seq.arrival = start + r.arrival_ms / 1000
            seq.high_water = 0
            seq.submitted = None
            schedulers[idx].add(seq)
            events[r.id] = []
            loads[idx] += 1
            totals[idx] += 1
            cursor += 1
        for idx, conn in enumerate(conns):
            scheduler = schedulers[idx]
            if conn in pending or scheduler.is_finished():
                continue
            batch, prefill = scheduler.schedule()
            peak_batch = max(peak_batch, len(batch))
            for seq in batch:
                if seq.submitted is None:
                    seq.submitted = monotonic()
                    hits += int(prefill and seq.num_cached_tokens >= PAGE)
                if prefill:
                    recomputed += min(seq.num_scheduled_tokens, max(0, seq.high_water-seq.num_cached_tokens))
                seq.high_water = max(seq.high_water, seq.num_cached_tokens + seq.num_scheduled_tokens)
            conn.send((batch, prefill))
            pending[conn] = (idx, batch, prefill)
        delay = max(0, start + trace[cursor].arrival_ms/1000 - monotonic()) if cursor < len(trace) else 1
        if not pending:
            sleep(min(0.05, delay))
            continue
        for conn in wait(list(pending), timeout=min(delay, 1)):
            idx, batch, prefill = pending.pop(conn)
            ids = receive(conn, "tokens")
            before = [s.num_completion_tokens for s in batch]
            schedulers[idx].postprocess(batch, ids, prefill)
            at = (monotonic()-start)*1000
            for seq, n in zip(batch, before):
                if seq.num_completion_tokens > n:
                    events[seq.rid].append(at)
                if seq.is_finished:
                    loads[idx] -= 1
                    outputs[seq.rid] = (seq.completion_token_ids, (seq.submitted-seq.arrival)*1000)
    elapsed = max(t[-1] for t in events.values())
    first = [events[r.id][0]-r.arrival_ms for r in trace]
    gaps = [[b-a for a,b in zip(times,times[1:])] for times in (events[r.id] for r in trace)]
    categories = dict.fromkeys(("met", "ttft_only", "gap_only", "both"), 0)
    for r, ttft, intervals in zip(trace, first, gaps):
        gap = max(intervals, default=0)
        outcome = verdict(ttft, gap, ttft_limit, gap_limit)
        categories[outcome] += 1
        if rows is not None:
            rows.append(dict(id=r.id, cohort=r.cohort, arrival_ms=r.arrival_ms,
                             input_tokens=len(r.prompt), generated=len(outputs[r.id][0]),
                             wait_ms=outputs[r.id][1], ttft_ms=ttft, max_gap_ms=gap, slo=outcome))
    passed = categories["met"]
    tokens = [outputs[r.id][0] for r in trace]
    if any(len(t) != r.output_tokens for t,r in zip(tokens,trace)):
        raise AssertionError("incomplete generation")
    stats = dict(replicas=len(conns), admission=policy, router=routing, per_replica=totals,
                 elapsed_ms=elapsed, completed=len(outputs), req_s=len(trace)*1000/elapsed,
                 slo_met=passed, goodput_req_s=passed*1000/elapsed, ttft_p95_ms=p95(first),
                 itl_p95_ms=p95([v for g in gaps for v in g]), first_admission_hits=hits,
                 cold_wait_p95_ms=p95([outputs[r.id][1] for r in trace if r.cohort == "cold"]),
                 slo_breakdown=categories, peak_batch=peak_batch, preemptions=sum(s.preemptions for s in schedulers),
                 recomputed_tokens=recomputed)
    return stats, tokens


@contextmanager
def replicas(devices, model, pages):
    ctx = mp.get_context("spawn")
    conns, processes = [], []
    metadata = []
    try:
        for device in devices:
            client, child = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(child, device, model, pages))
            proc.start()
            child.close()
            client.gpu_index, client.worker_pid = device, proc.pid
            conns.append(client)
            processes.append(proc)
        for conn in conns:
            if not conn.poll(300):
                raise TimeoutError("worker initialization timed out")
            metadata.append(receive(conn, "ready"))
        yield conns, metadata
    finally:
        for conn in conns:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
            conn.close()
        for proc in processes:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
                proc.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--devices", type=int, nargs="+", default=[0, 1])
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--pages", type=int, default=16)
    p.add_argument("--interval-ms", type=float, default=110)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (len(set(args.devices)) != len(args.devices) or min(args.devices) < 0 or
            args.pages < 3 or min(args.count, args.repeats, args.interval_ms) <= 0):
        p.error("provide distinct GPUs, at least 3 pages and positive run dimensions")
    config = Config(args.model, max_num_batched_tokens=4096, max_model_len=1024,
                    max_num_seqs=16, num_kvcache_blocks=args.pages)
    report = dict(model=Path(args.model).name, pages_per_gpu=args.pages, page_size=PAGE,
                  seed=args.seed, burst=8, interval_ms=args.interval_ms, ttft_slo_ms=250,
                  max_itl_slo_ms=100, workers=[], cases=[])
    with replicas(args.devices, args.model, args.pages) as (conns, metadata):
        report["workers"] = metadata
        for conn in conns:
            replay(traffic(32, 91, True, 1), [conn], config, "guarded", "rr", 250, 100)
        layouts = [(1, "stock", "rr"), (1, "guarded", "rr")]
        if len(conns) > 1:
            layouts += [(len(conns), "stock", "rr")] + [
                (len(conns), "guarded", mode) for mode in ("rr", "least", "affinity")]
        for mixed in (True, False):
            trace = traffic(args.count, args.seed, mixed, args.interval_ms)
            reference = None
            for rep in range(args.repeats):
                order = layouts if rep % 2 == 0 else layouts[::-1]
                for n, policy, router in order:
                    stats, tokens = replay(trace, conns[:n], config, policy, router, 250, 100)
                    if reference is None:
                        reference = tokens
                    stats.update(profile="mixed" if mixed else "unique", trial=rep+1,
                                 trace_sha256=fingerprint(trace),
                                 output_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                                 exact_vs_first_stock=sum(a == b for a,b in zip(reference,tokens)))
                    report["cases"].append(stats)
                    print(stats["profile"], rep+1, n, policy, router, round(stats["req_s"],1),
                          stats["slo_met"], flush=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
