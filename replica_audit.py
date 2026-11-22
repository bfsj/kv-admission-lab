"""Compare full-context and paged logits at identical token histories on each GPU."""

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import torch
import torch.distributed as dist
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

from nano_worker import ReplicaRunner
from replica_bench import PAGE, traffic


def prefill(runner, tokens, pages, cached=0):
    seq = Sequence(list(tokens))
    seq.block_table = pages[:seq.num_blocks]
    seq.num_cached_tokens = cached
    seq.num_scheduled_tokens = len(seq) - cached
    inputs, positions = runner.prepare_prefill([seq])
    logits = runner.run_model(inputs, positions, True)[0].float()
    reset_context()
    return logits


@torch.inference_mode()
def check(device_model):
    device, model = device_model
    cfg = Config(model, max_num_batched_tokens=1024, max_model_len=1024, max_num_seqs=4)
    runner = ReplicaRunner(cfg, device, 6)
    pairs, similarities, disagreements = [], [], []
    try:
        for request in traffic(16, 17, True, 220):
            tokens = list(request.prompt)
            # Populate the shared page using another prompt with the same first 256 IDs.
            prefill(runner, tokens[:PAGE] + [7, 8], [3, 4])
            for step in range(request.output_tokens):
                ref = prefill(runner, tokens, [0, 1, 2])
                if step == 0:
                    paged = prefill(runner, tokens, [3, 4, 5], cached=PAGE)
                else:
                    seq = Sequence(tokens.copy())
                    seq.block_table = [3, 4, 5][:seq.num_blocks]
                    inputs, positions = runner.prepare_decode([seq])
                    paged = runner.run_model(inputs, positions, False)[0].float()
                    reset_context()
                if not torch.isfinite(ref).all() or not torch.isfinite(paged).all():
                    raise ValueError("non-finite logits")
                baseline, actual = ref.argmax().item(), paged.argmax().item()
                pairs.append((baseline, actual))
                similarities.append(torch.nn.functional.cosine_similarity(ref, paged, dim=0).item())
                if actual != baseline:
                    top2 = torch.topk(ref, 2).values
                    disagreements.append({"request": request.id, "step": step,
                                          "reference_top2_gap": (top2[0]-top2[1]).item()})
                tokens.append(baseline)
        return dict(device=device, prompts=16, compared_steps=len(pairs),
                    matching_argmax=sum(a == b for a,b in pairs), min_logit_cosine=min(similarities),
                    mismatches=disagreements,
                    argmax_digest=hashlib.sha256(json.dumps(pairs).encode()).hexdigest())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices):
        parser.error("provide distinct GPU indices")
    with mp.get_context("spawn").Pool(len(args.devices)) as pool:
        rows = pool.map(check, [(i, args.model) for i in args.devices])
    result = {"workers": rows, "identical_argmax_pairs_across_devices":
              len({r["argmax_digest"] for r in rows}) == 1}
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf8")
