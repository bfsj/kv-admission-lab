"""Single-worker replay. Backend returns a next token and a service time in ms."""

import argparse
import json
import math
from dataclasses import asdict, dataclass

from admission import Pool, State, choose
from workload import fingerprint, make_trace


@dataclass(frozen=True)
class Config:
    pages: int = 12
    page_size: int = 16
    slots: int = 5
    window: int = 8
    aging_ms: float = 90
    queue_timeout_ms: float = 800
    ttft_slo_ms: float = 160
    itl_slo_ms: float = 20


class CostModel:
    """Illustrative serial timing, independent of hardware."""
    def step(self, s, ids, cached, pages):
        return 42, 0.35 + 0.16 * len(ids)


def p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def run(trace, cfg, policy, backend=None):
    if policy not in ("fcfs", "cache", "guarded"):
        raise ValueError("unknown policy")
    if min(asdict(cfg).values()) <= 0 or not trace:
        raise ValueError("positive settings and a nonempty trace are required")
    if len({r.id for r in trace}) != len(trace):
        raise ValueError("request IDs must be unique")
    if any(r.arrival_ms < 0 or r.output_tokens < 1 or len(r.prompt) <= cfg.page_size for r in trace):
        raise ValueError("each prompt needs a full prefix page and a nonempty suffix")
    backend = backend or CostModel()
    pool = Pool(cfg.pages, cfg.page_size)
    states = [State(r) for r in sorted(trace, key=lambda r: (r.arrival_ms, r.id))]
    active, waiting = [], []
    cursor, now = 0, 0.0

    def forward(s, start):
        nonlocal now
        ids = s.request.prompt + tuple(s.generated)
        token, elapsed = backend.step(s, ids[start:], start, s.pages)
        now += elapsed
        s.recomputed_tokens += max(0, s.computed - start)
        s.computed = len(ids)
        s.generated.append(token)
        s.emitted_ms.append(now)
        if len(s.generated) == s.request.output_tokens:
            pool.release(s)
            active.remove(s)
            s.status = "done"

    while cursor < len(states) or active or waiting:
        if not active and not waiting:
            now = max(now, states[cursor].request.arrival_ms)
        while cursor < len(states) and states[cursor].request.arrival_ms <= now:
            s = states[cursor]
            cursor += 1
            if pool.page_count(len(s.request.prompt) + s.request.output_tokens - 1) > cfg.pages:
                s.status = "rejected"
            else:
                waiting.append(s)
        for s in waiting.copy():
            if s.admitted_ms is None and now - s.request.arrival_ms >= cfg.queue_timeout_ms:
                s.status = "timeout"
                waiting.remove(s)
        # One prefill per round. Decode each previously running request once.
        decoding = active.copy()
        picked = choose(waiting, active, pool, policy, cfg.window, cfg.aging_ms, now)
        if picked is not None and len(active) < cfg.slots:
            waiting.remove(picked)
            cached = pool.attach(picked)
            picked.prefix_hits += int(cached > 0)
            if picked.admitted_ms is None:
                picked.admitted_ms = now
            active.append(picked)
            forward(picked, cached)
        for s in decoding:
            if s not in active:
                continue
            if pool.page_count(s.length) > len(s.pages):
                if not pool.available():
                    # The current request can finish alone (checked at ingress).
                    # Evict youngest competitors until a reclaimable page exists.
                    while not pool.available():
                        victim = next(v for v in reversed(active) if v is not s)
                        pool.release(victim)
                        active.remove(victim)
                        victim.preemptions += 1
                        waiting.append(victim)
                    waiting.sort(key=lambda v: (v.request.arrival_ms, v.request.id))
                s.pages.append(pool.take())
            forward(s, s.length - 1)

    done = [s for s in states if s.status == "done"]
    def ttft(s):
        return s.emitted_ms[0] - s.request.arrival_ms
    def gaps(s):
        return [b - a for a, b in zip(s.emitted_ms, s.emitted_ms[1:])]
    passed = sum(ttft(s) <= cfg.ttft_slo_ms and max(gaps(s), default=0) <= cfg.itl_slo_ms for s in done)
    duration = now - states[0].request.arrival_ms
    metrics = {
        "policy": policy, "offered": len(states), "completed": len(done),
        "timeouts": sum(s.status == "timeout" for s in states),
        "rejected": sum(s.status == "rejected" for s in states),
        "slo_met": passed, "goodput_req_s": passed * 1000 / duration if duration else 0,
        "elapsed_ms": duration, "ttft_p95_ms": p95([ttft(s) for s in done]),
        "itl_p95_ms": p95([gap for s in done for gap in gaps(s)]),
        "wait_p95_ms": {c: p95([s.admitted_ms - s.request.arrival_ms for s in states
                                if s.admitted_ms is not None and s.request.cohort == c])
                        for c in ("hot", "cold")},
        "prefix_hits_including_resumes": sum(s.prefix_hits for s in states),
        "preemptions": sum(s.preemptions for s in states),
        "recomputed_tokens": sum(s.recomputed_tokens for s in states),
        "peak_pinned_pages": pool.peak,
    }
    return metrics, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pages", type=int, default=12)
    args = parser.parse_args()
    cfg = Config(pages=args.pages)
    trace = make_trace(args.count, args.seed, cfg.page_size)
    print(json.dumps({"backend": "synthetic-cost", "seed": args.seed, "config": asdict(cfg),
                      "trace_sha256": fingerprint(trace),
                      "runs": [run(trace, cfg, p)[0] for p in ("fcfs", "cache", "guarded")]},
                     indent=2))
