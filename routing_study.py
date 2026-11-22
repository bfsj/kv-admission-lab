"""Paired cache and load routing comparisons on tokenized support-ticket text."""

import argparse
import gzip
import json
import random
from pathlib import Path

from transformers import AutoTokenizer
from nanovllm.config import Config
from capacity_study import measure
from replica_bench import replicas, traffic, replay
from workload import Request


def text_trace(tokenizer, rate, count=300, seed=17):
    rng = random.Random(seed)
    procedures = ["核对订单与配送时段", "检查退款与支付记录", "检查库存与替换商品"]
    questions = ["请指出下一步需要核查的字段。", "请用一句话向客户说明处理进度。",
                 "哪些记录互相矛盾？", "列出时间线，并指出仍需人工确认的事项。"]
    trace = []
    for i in range(count):
        hot = rng.random() < 0.66
        long_ticket = rng.random() < 0.40
        topic = rng.choice(procedures)
        guide = f"客服工作手册：{topic}。\n"
        if hot:
            guide += "".join(f"核验步骤 {j}：比较原始凭证与状态变更时间。字段缺失时标记为待核实，保留客户原话。\n"
                             for j in range(1, 13))
        else:
            guide = f"独立工单 {i:04d}。{topic}。\n"
        notes = "".join(f"{9+j//6:02d}:{(j*9)%60:02d} 经办人记录：{rng.choice(procedures)}；"
                        f"系统记录编号 {rng.randrange(1000,9999)}。\n"
                        for j in range(14 if long_ticket else 1))
        ids = tokenizer.encode(guide + notes + rng.choice(questions), add_special_tokens=False)
        output = rng.choice((24, 40, 56) if long_ticket else (4, 8, 12))
        if len(ids) + output > 1024:
            raise ValueError("trace exceeds the worker context")
        trace.append(Request(i, i//8 * 8000/rate, "hot" if hot else "cold", tuple(ids), output))
    return trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--devices", nargs=2, type=int, default=[0, 1])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if len(set(args.devices)) != 2:
        p.error("two distinct GPUs required")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    cfg = Config(args.model, max_num_batched_tokens=4096, max_model_len=1024,
                 max_num_seqs=16, num_kvcache_blocks=16)
    cases = []
    modes = ["least", "affinity", "work"]
    with replicas(args.devices, args.model, 16) as (conns, cards), gzip.open(
            args.output.with_suffix(".requests.jsonl.gz"), "wt", encoding="utf8") as details:
        for conn in conns:
            replay(traffic(32, 91, True, 1), [conn], cfg, "guarded", "rr", 250, 100)
        for rate in (18, 36):
            trace = text_trace(tok, rate)
            for trial in range(3):
                for mode in modes[trial:] + modes[:trial]:
                    cases.append(measure(trace, conns, cfg, "guarded", mode,
                                         dict(trial=trial+1, rate=rate), details))
    args.output.write_text(json.dumps(dict(model=Path(args.model).name, workers=cards, seed=17,
            input_range=[min(len(r.prompt) for r in trace), max(len(r.prompt) for r in trace)],
            output_range=[min(r.output_tokens for r in trace), max(r.output_tokens for r in trace)],
            cases=cases), indent=2)+"\n")


if __name__ == "__main__":
    main()
