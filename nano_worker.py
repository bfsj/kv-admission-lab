"""One independent Qwen3 replica. Uses nano-vLLM's real prefill/decode kernels."""

import traceback

import torch
import torch.distributed as dist
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model


class ReplicaRunner(ModelRunner):
    def __init__(self, config, device, pages):
        self.config = config
        self.rank = 0
        self.world_size = 1
        self.block_size = config.kvcache_block_size
        self.enforce_eager = True
        self.sampler = lambda logits, temperatures: logits.argmax(dim=-1)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", store=dist.HashStore(), rank=0, world_size=1)
        hf = config.hf_config
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(hf.dtype)
            with torch.device("cuda"):
                self.model = Qwen3ForCausalLM(hf)
                load_model(self.model, config.model)
                self.warmup_model()
                self.kv_cache = torch.empty(2, hf.num_hidden_layers, pages, self.block_size,
                                            hf.num_key_value_heads, hf.head_dim)
                layers = [m for m in self.model.modules() if hasattr(m, "k_cache")]
                if len(layers) != hf.num_hidden_layers:
                    raise ValueError("unexpected Qwen3 attention layout")
                for index, layer in enumerate(layers):
                    layer.k_cache, layer.v_cache = self.kv_cache[:, index].unbind()
        finally:
            torch.set_default_dtype(previous)


def worker(conn, device, model, pages):
    try:
        config = Config(model, max_num_batched_tokens=4096, max_model_len=1024, max_num_seqs=16)
        runner = ReplicaRunner(config, device, pages)
        props = torch.cuda.get_device_properties(device)
        conn.send(("ready", {"device": device, "gpu": props.name, "torch": str(torch.__version__),
                             "dtype": str(config.hf_config.dtype),
                             "kv_bytes": runner.kv_cache.nbytes}))
        while True:
            message = conn.recv()
            if message is None:
                break
            seqs, prefill = message
            for seq in seqs:
                seq.temperature = 1.0
            conn.send(("tokens", runner.run(seqs, prefill)))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        conn.close()
