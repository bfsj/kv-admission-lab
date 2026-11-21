"""Replay with real Qwen3 forward passes and an explicitly bounded KV page arena."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoModelForCausalLM, DynamicCache

from replay import Config, run
from workload import fingerprint, make_trace


class QwenWorker:
    def __init__(self, path, cfg, device):
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32,
            attn_implementation="sdpa").to(device).eval()
        model_cfg = self.model.config
        if model_cfg.model_type != "qwen3" or getattr(model_cfg, "use_sliding_window", False):
            raise ValueError("this worker supports full-attention Qwen3 models")
        self.size = cfg.page_size
        self.device = torch.device(device)
        self.arena = torch.empty((2, model_cfg.num_hidden_layers, cfg.pages, self.size,
                                  model_cfg.num_key_value_heads, model_cfg.head_dim),
                                 dtype=torch.float32, device=self.device)

    @torch.inference_mode()
    def step(self, state, ids, cached, pages):
        started = perf_counter()
        past = None
        if cached:
            n = (cached + self.size - 1) // self.size
            indices = torch.tensor(pages[:n], device=self.device)
            # Gather page-major KV into the contiguous layout expected by HF SDPA.
            history = self.arena.index_select(2, indices).flatten(2, 3)[:, :, :cached]
            keys, values = history.transpose(2, 3).unbind(0)
            past = DynamicCache.from_legacy_cache(tuple(
                (k.unsqueeze(0), v.unsqueeze(0)) for k, v in zip(keys, values)))
        inputs = torch.tensor([ids], device=self.device)
        out = self.model(inputs, past_key_values=past, use_cache=True, logits_to_keep=1)
        # Write only newly computed tokens. Shared prefix pages remain read-only on hits.
        kv = out.past_key_values.to_legacy_cache()
        new = torch.stack([torch.stack([layer[i][0, :, cached:] for layer in kv])
                           for i in range(2)]).transpose(2, 3)
        pos, stop = cached, cached + len(ids)
        while pos < stop:
            page, offset = divmod(pos, self.size)
            span = min(self.size - offset, stop - pos)
            self.arena[:, :, pages[page], offset:offset + span].copy_(new[:, :, pos-cached:pos-cached+span])
            pos += span
        token = out.logits[0, -1].argmax().item()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return token, (perf_counter() - started) * 1000

    @torch.inference_mode()
    def reference(self, request):
        ids = list(request.prompt)
        for _ in range(request.output_tokens):
            logits = self.model(torch.tensor([ids], device=self.device), use_cache=False,
                                logits_to_keep=1).logits
            ids.append(logits[0, -1].argmax().item())
        return ids[len(request.prompt):]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config(pages=12, page_size=8, slots=4, aging_ms=500,
                 queue_timeout_ms=120000, ttft_slo_ms=2000, itl_slo_ms=500)
    trace = [replace(r, output_tokens=4 + r.output_tokens % 5) for r in
             make_trace(args.count, args.seed, cfg.page_size, burst=4, interval_ms=250)]
    worker = QwenWorker(args.model, cfg, args.device)
    reference = {r.id: worker.reference(r) for r in trace}
    results = []
    for policy in ("fcfs", "cache", "guarded"):
        stats, states = run(trace, cfg, policy, worker)
        matches = sum(s.generated == reference[s.request.id] for s in states)
        stats["exact_sequences"] = matches
        results.append(stats)
    digest = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    report = {"backend": "real-model-service-clock", "model": args.model.name,
              "dtype": "float32", "device": args.device, "threads": args.threads,
              "torch": str(torch.__version__), "transformers": transformers.__version__,
              "platform": platform.system() + " " + platform.machine(),
              "config": asdict(cfg), "seed": args.seed, "trace_sha256": fingerprint(trace),
              "arena_bytes": worker.arena.numel() * worker.arena.element_size(),
              "reference_tokens": sum(map(len, reference.values())),
              "reference_sha256": digest, "runs": results}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    print(output)
    if any(s["exact_sequences"] != len(trace) for s in results):
        raise SystemExit("generated sequences differ from the uncached reference or did not finish")


if __name__ == "__main__":
    main()
