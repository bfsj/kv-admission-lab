"""One shareable prefix page per prompt, plus private pages that grow on demand."""

from collections import OrderedDict
from dataclasses import dataclass, field

from workload import Request


@dataclass
class State:
    request: Request
    pages: list = field(default_factory=list)
    generated: list = field(default_factory=list)
    emitted_ms: list = field(default_factory=list)
    admitted_ms: float | None = None
    computed: int = 0
    preemptions: int = 0
    recomputed_tokens: int = 0
    prefix_hits: int = 0
    status: str = "waiting"

    @property
    def length(self):
        return len(self.request.prompt) + len(self.generated)


class Pool:
    def __init__(self, capacity, size):
        self.capacity = capacity
        self.size = size
        self.free = list(range(capacity))
        self.prefixes = OrderedDict()
        self.refs = [0] * capacity
        self.peak = 0

    def prefix(self, s):
        return s.request.prompt[:self.size]

    def available(self):
        return len(self.free) + sum(self.refs[p] == 0 for p in self.prefixes.values())

    def page_count(self, tokens):
        return (tokens + self.size - 1) // self.size

    def extra(self, s):
        page = self.prefixes.get(self.prefix(s))
        return self.page_count(s.length) - int(page is not None and self.refs[page] > 0)

    def take(self):
        if not self.free:
            key = next(k for k, p in self.prefixes.items() if self.refs[p] == 0)
            self.free.append(self.prefixes.pop(key))
        page = self.free.pop()
        self.refs[page] = 1
        self.peak = max(self.peak, self.capacity - self.available())
        return page

    def attach(self, s):
        key = self.prefix(s)
        hit = key in self.prefixes
        if hit:
            first = self.prefixes[key]
            self.refs[first] += 1
            self.prefixes.move_to_end(key)
        else:
            first = self.take()
            self.prefixes[key] = first
        s.pages.append(first)
        while len(s.pages) < self.page_count(s.length):
            s.pages.append(self.take())
        self.peak = max(self.peak, self.capacity - self.available())
        return self.size if hit else 0

    def release(self, s):
        self.refs[s.pages[0]] -= 1
        for page in s.pages[1:]:
            self.refs[page] = 0
            self.free.append(page)
        s.pages.clear()


def choose(waiting, active, pool, policy, window, aging_ms, now):
    if not waiting:
        return None
    oldest = waiting[0]
    if policy == "fcfs" or (policy == "guarded" and
                            now - oldest.request.arrival_ms >= aging_ms):
        return oldest if pool.extra(oldest) <= pool.available() else None
    candidates = waiting[:window]
    available = pool.available()
    # Reserve the next decode step for active sequences near a page boundary.
    reserve = sum(s.length > len(s.pages) * pool.size for s in active) if policy == "guarded" else 0
    feasible = [s for s in candidates if pool.extra(s) <= available - reserve]
    return max(feasible, key=lambda s: pool.prefix(s) in pool.prefixes, default=None)
