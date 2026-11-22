"""Verify saved experiment counters against every request in their paired trace."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from replay import p95


def key(case):
    return tuple(case[k] for k in ("rate", "trial", "replicas", "pages", "router"))


def verify(path):
    report = json.loads(path.read_text(encoding="utf8"))
    cases = {key(c): c for c in report["cases"]}
    if not cases or len(cases) != len(report["cases"]):
        raise ValueError("missing or duplicate summary case")
    signatures = {}
    count = 0
    with gzip.open(path.with_suffix(".requests.jsonl.gz"), "rt", encoding="utf8") as detail:
        for line in detail:
            record = json.loads(line)
            c = cases.pop(key(record["case"]))
            rows = record["requests"]
            counters = Counter()
            if not (len(rows) == c["completed"] == sum(c["per_replica"]) == 300):
                raise ValueError("incomplete request set")
            for i,r in enumerate(rows):
                times = [r[k] for k in ("arrival_ms","wait_ms","ttft_ms","max_gap_ms")]
                if (r["id"] != i or r["generated"] < 1 or
                    not all(math.isfinite(t) and t >= 0 for t in times) or r["wait_ms"] > r["ttft_ms"]+1e-5):
                    raise ValueError("invalid request timing or identity")
                miss_first = r["ttft_ms"] > report.get("ttft_limit_ms",250)
                miss_gap = r["max_gap_ms"] > report.get("max_gap_limit_ms",100)
                outcome = {(False,False):"met",(False,True):"gap_only",
                           (True,False):"ttft_only",(True,True):"both"}[miss_first,miss_gap]
                if r["slo"] != outcome:
                    raise ValueError("incorrect per-request SLO label")
                counters[outcome] += 1
            if (counters != Counter(c["slo_breakdown"]) or counters["met"] != c["slo_met"]
                or not math.isclose(c["goodput_req_s"], counters["met"]*1000/c["elapsed_ms"])
                or p95([r["ttft_ms"] for r in rows]) != c["ttft_p95_ms"]
                or p95([r["wait_ms"] for r in rows if r["cohort"]=="cold"]) != c["cold_wait_p95_ms"]):
                raise ValueError("summary does not match request records")
            if signatures.setdefault(c["rate"],c["trace_sha256"]) != c["trace_sha256"]:
                raise ValueError("unpaired trace within a rate")
            count += len(rows)
    if cases:
        raise ValueError("missing detailed case")
    return count


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", type=Path, nargs="+")
    for path in p.parse_args().files:
        print(path, verify(path), "request observations verified")
