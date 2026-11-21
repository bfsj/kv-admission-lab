"""Seeded open-loop arrivals. Token IDs make the trace independent of a tokenizer."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Request:
    id: int
    arrival_ms: float
    cohort: str
    prompt: tuple[int, ...]
    output_tokens: int


def make_trace(count=300, seed=17, page_size=16, burst=8, interval_ms=55):
    if min(count, page_size, burst, interval_ms) <= 0:
        raise ValueError("trace dimensions must be positive")
    rng = random.Random(seed)
    trace = []
    for i in range(count):
        hot = rng.random() < 0.65
        prefix = rng.choice([100, 101, 102]) if hot else 1000 + i
        suffix = tuple(rng.randrange(10, 90) for _ in range(rng.randint(4, 18)))
        prompt = (prefix,) * page_size + suffix
        trace.append(Request(i, i // burst * interval_ms, "hot" if hot else "cold",
                             prompt, rng.randint(4, 20)))
    return trace


def fingerprint(trace):
    payload = json.dumps([asdict(r) for r in trace], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
