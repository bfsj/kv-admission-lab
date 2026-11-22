# KV Admission Lab

[RTX 4090 单卡与双副本实测](results/replicas.md) · [36 组原始数据](results/rtx4090-replicas.json)

[容量边界与逐请求超时归因](results/capacity.md) · [长短中文请求的路由对照](results/routing.md) · [流式 API 与断开回收](serving.md)

一个面向单个 LLM 推理 worker 的接纳调度实验。多条请求可以复用同一段提示词的 KV，但命中缓存的请求插队也会增加冷请求的等待时间。这个实验固定到达流量、生成长度和 KV 页数，比较三种排队规则。

| 策略 | 何时接纳下一条请求 |
| --- | --- |
| FCFS | 只考察队首，页数足够就接纳 |
| cache | 看队首起的有限窗口，优先接纳已有首个前缀页的请求，同分按到达顺序 |
| guarded | 在 cache 基础上，给即将跨页的 decode 预留空间。队首等待超过阈值后优先处理队首 |

每轮最多接纳一个请求，再推进上一轮仍活跃的请求。页池按引用计数共享首个完整前缀页，私有尾部按需增长。空间不足时先回收闲置前缀，再抢占活跃请求。恢复时使用已有首个前缀页，重算丢失的上下文。每个请求都带有已知的输出长度上限。

## 复现调度实验

Python 3.10+，默认实验仅使用标准库。在仓库根目录运行：

```sh
python -m unittest -v
python replay.py --seed 17 --count 300 --pages 12
```

默认流量每 55 ms 同时到达 8 条请求。热组从三个固定首段中抽样，冷组各有独立首段。组别由流量生成器定义，实际命中取决于当时的页池。输出包含输入轨迹的 SHA-256、三种策略的配置和结果。

合成后端按每次调用 0.35 ms 加每个输入 token 0.16 ms 推进时钟。goodput 是同时满足首 token 延迟和每一次后续 token 间隔上限的请求数，除以从首次到达到最后一次完成的秒数。等待时间统计首次成功接纳的请求，超时和拒绝单列。重算量只包含重复执行过的上下文 token。默认输出的时间指标来自该成本模型。

## 实验结果

固定默认配置，种子 17–21，每个种子 300 条请求。每种策略处理相同的 1,500 条请求。首 token 目标 160 ms，每一次后续 token 间隔目标 20 ms，首次接纳超时 800 ms。表中 goodput 用五次运行的达标总数除以总回放秒数。[逐次结果和轨迹摘要](results/synthetic.json)

| 策略 | 同时满足两项延迟目标 | 合成 goodput（请求/秒） | 抢占次数 | 重算 token | 接纳超时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FCFS | 314 | 22.21 | 174 | 2,743 | 7 |
| cache | 335 | 24.08 | 145 | 2,531 | 0 |
| guarded | 318 | 22.63 | 130 | 2,133 | 3 |

五组汇总中，cache 的达标吞吐最高。guarded 把重算量降得更多，但为后续 decode 预留页也限制了接纳新请求的机会。保护规则对冷请求的收益取决于流量：种子 18 的冷组等待 P95 从 FCFS 的 709.0 ms 降到 682.6 ms，种子 17 则从 729.1 ms 升到 730.6 ms。

## 真实模型校验

准备本地 Qwen/Qwen3-0.6B 模型目录，安装下列依赖后运行。默认使用 CPU、FP32、4 线程和 greedy 解码。模型权重文件的 SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

```sh
python -m pip install -r requirements-model.txt
python model_replay.py --model /path/to/Qwen3-0.6B
```

这条命令先用完整上下文逐步前向，生成参考 token，然后对三种策略逐请求核对整个输出。模型后端将各层真实 K/V 写入固定的页池，复用命中的前缀页，抢占后用已生成的 token 恢复上下文。HF attention 所需的连续历史由页池临时拼装，页池大小只统计持久 K/V，共享首段限一页。

本次 CPU 校验使用 16 个请求、共 101 个输出 token，页池 12 页，每页 8 个 token，持久 K/V 共 21 MiB。模型调用串行执行，回放时钟累计真实前向和 KV 搬运耗时，并按指定到达时间插入空闲期。该实验校验输出正确性；GPU continuous batching 和 HTTP 端到端性能需要单独测量。模型入口也接受 `--device cuda`。[CPU 实测记录](results/qwen3-cpu.json)

## 单卡与多副本 GPU 推理

同一个调度进程接收预分词请求，每张 GPU 上有一个独立的 Qwen3 模型和固定大小的 KV 页池。请求从接纳到完成一直留在初次分配的卡上。各卡独立做 continuous batching，经进程间管道异步返回生成结果。单卡和多卡复用同一套 nano-vLLM 模型、FlashAttention 和分页 KV 执行路径。

单卡比较上游调度器（stock）和带保护的接纳策略（guarded）。guarded 在长等待队首出现前，从前八个候选中优先选可接纳且命中块最多的请求，并在分配时给将要跨页的运行中序列留出 KV 块。双卡增加三种路由：轮询、最少未完成请求、缓存亲和。亲和路由只考虑已有首个完整前缀页、且未完成请求数至多比最空闲卡多一条的副本。所有策略共用相同的自然淘汰和抢占机制。

Linux CUDA 环境需安装与 PyTorch 匹配的 FlashAttention 2，并使用固定的 nano-vLLM 版本。在已有模型和所需依赖的环境中运行：

```sh
git clone https://github.com/GeeeekExplorer/nano-vllm.git ../nano-vllm-pinned
git -C ../nano-vllm-pinned checkout bb823b3e06983d71485a8e1f23715ebd87d98ef8
export PYTHONPATH="$PWD/../nano-vllm-pinned"
python replica_bench.py --model /path/to/Qwen3-0.6B --devices 0 1 --count 300 --repeats 3 --interval-ms 220 --output replicas.json
```

`--devices 0` 只运行单卡对照。每卡各有 16 个 256-token 的物理 KV 块，双卡总缓存随副本数翻倍。每个场景启动前清空调度和缓存元数据，保留已预热的模型。硬件时钟从计划到达计到控制进程收到 token，包括排队、调度、管道传输和模型执行。流量采用成批到达的合成 token，生成固定长度并忽略 EOS。两类流量共用长度和到达时间；mixed 中约 70% 的请求从三个热前缀中选择，unique 中每个请求的首块均不同。正式对比统计 250 ms 首 token 和每个后续间隔 100 ms 双重上限下的达标吞吐。
