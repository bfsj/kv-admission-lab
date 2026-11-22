"""Measure a sampled SLO capacity boundary on matched request traces."""

import argparse
import gzip
import json
import subprocess
from dataclasses import replace
from pathlib import Path

from nanovllm.config import Config
from replica_bench import replay, replicas, traffic
from workload import fingerprint


def other_process_counts(conns):
    readings = []
    for conn in conns:
        try:
            output = subprocess.check_output(["nvidia-smi", "-i", str(conn.gpu_index),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=5)
            readings.append(sum(int(s) != conn.worker_pid for s in output.split() if s.isdigit()))
        except (OSError, subprocess.SubprocessError):
            readings.append(None)
    return readings


def measure(trace, conns, cfg, policy, route, tags, details):
    rows = []
    before = other_process_counts(conns)
    stats, _ = replay(trace, conns, cfg, policy, route, 250, 100, rows)
    stats["other_gpu_processes_before_after"] = [before, other_process_counts(conns)]
    key = dict(**tags, replicas=len(conns), pages=cfg.num_kvcache_blocks, router=route)
    details.write(json.dumps(dict(case=key, requests=rows), separators=(",", ":")) + "\n")
    stats.update(key, trace_sha256=fingerprint(trace))
    print(key, stats["slo_breakdown"], flush=True)
    return stats


def stable(cases, n, rate):
    rates = [c["slo_met"]/c["completed"] for c in cases
             if c["replicas"] == n and c["rate"] == rate]
    return bool(rates) and min(rates) >= 0.95


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--devices", nargs=2, type=int, default=[0, 1])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--paced", action="store_true", help="single-GPU, evenly spaced control")
    args = p.parse_args()
    if len(set(args.devices)) != 2:
        p.error("two distinct GPUs required")
    cases, metadata = [], []
    rates = [12, 24, 36, 48]
    def run_point(n, rate, trial, cfg, conns, details):
        trace = traffic(300, 17, True, 8000/rate)
        if args.paced:
            trace = [replace(r, arrival_ms=r.id*1000/rate) for r in trace]
        cases.append(measure(trace, conns[:n], cfg, "stock", "rr",
                             dict(trial=trial, rate=rate), details))
    with gzip.open(args.output.with_suffix(".requests.jsonl.gz"), "wt", encoding="utf8") as details:
        layouts = [(16, args.devices[:1])] if args.paced else [(16, args.devices), (32, args.devices[:1])]
        for pages, devices in layouts:
            cfg = Config(args.model, max_num_batched_tokens=4096, max_model_len=1024,
                         max_num_seqs=16, num_kvcache_blocks=pages)
            with replicas(devices, args.model, pages) as (conns, cards):
                metadata.append(cards)
                for conn in conns:
                    replay(traffic(32, 91, True, 1), [conn], cfg, "stock", "rr", 250, 100)
                if args.paced:
                    for trial in range(1, 4):
                        for rate in ((12, 18, 24, 36) if trial % 2 else (36, 24, 18, 12)):
                            run_point(1, rate, trial, cfg, conns, details)
                elif pages == 16:
                    for trial in range(1, 4):
                        for rate in (rates if trial % 2 else rates[::-1]):
                            for n in ((1, 2) if trial % 2 else (2, 1)):
                                run_point(n, rate, trial, cfg, conns, details)
                    refinement = {}
                    for n in (1, 2):
                        bad = next((r for r in rates if not stable(cases, n, r)), None)
                        lo = rates[rates.index(bad)-1] if bad and bad > rates[0] else 0
                        refinement[n] = (lo+bad)/2 if bad else 60
                        for trial in range(1, 4):
                            run_point(n, refinement[n], trial, cfg, conns, details)
                else:
                    # Match the dual-card total KV budget at the same measured rates.
                    for rate in sorted({24, 36, refinement[2]}):
                        for trial in range(1, 4):
                            run_point(1, rate, trial, cfg, conns, details)
    args.output.write_text(json.dumps(dict(model=Path(args.model).name, workers=metadata,
        count=300, seed=17, arrival="paced" if args.paced else "burst8",
        ttft_limit_ms=250, max_gap_limit_ms=100, cases=cases), indent=2)+"\n")


if __name__ == "__main__":
    main()
