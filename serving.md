# 流式生成接口

Linux 单机服务支持一个或多个 nano-vLLM 副本。每张卡保有独立的模型和页池，控制进程按可观测 token 工作量分流。使用 README 固定的 nano-vLLM、PyTorch、FlashAttention 与本地 Qwen3-0.6B，再安装 HTTP 依赖。

```sh
python -m pip install -r requirements-serving.txt
python stream_api.py --model /path/to/Qwen3-0.6B --devices 0 1 --pages 16 --port 8080
```

服务默认绑定回环地址。向 `POST /generate` 提交 JSON，响应为 UTF-8 SSE。每个 `token` 事件带 token ID 和可立即显示的文本片段，`done` 事件带停止原因、生成长度、最终文本和最后需要补上的 `text_tail`。UTF-8 字符跨 token 时，中间文本片段可能为空。

```sh
curl -N -H 'Content-Type: application/json' \
  -d '{"prompt":"请解释推理请求为什么需要排队。","max_tokens":32}' \
  http://127.0.0.1:8080/generate
curl http://127.0.0.1:8080/metrics
```

每请求最多生成 256 个 token，prompt 和生成上限之和须在模型上下文和单卡页预算内。无效请求返回 400，未完成的请求已达 128 个时返回 429。GPU worker 出错会终止在途流，后续请求收到 503。服务重启后可重新加载模型。

客户端断开时，调度器立即移除排队请求。已派发的批次先完成本轮 GPU 写入，再释放被取消请求引用的页，其他请求可以继续使用共享页。`/metrics` 报告每卡排队、运行、固定占用和可分配页数，同时给出全局完成、取消和健康状态。

## 验证

```sh
python -m unittest discover -v
python stream_probe.py --url http://127.0.0.1:8080 --count 24 --output stream-http.json
```

单测使用真实调度器和可控制返回时机的推理替身，覆盖 HTTP 输入、排队上限、EOS 和长度停止、跨 token 文本、等待中与在途取消、共享页引用、多副本独立推进和 worker 失败。客户端探针面向空闲服务，用独立 HTTP 客户端从发送开始计时至解析到第一个 token；四并发请求后，主动断开一条正在输出的连接，检查取消计数和页回收，再发一条请求验证服务可用。客户端与 GPU 在同一台主机，计时包含 HTTP、分词、排队、模型执行及 SSE 传输。

实际双 4090 验证中，24 条四并发请求全部正常结束，每条流拼接后的文本均与最终文本一致。客户端收到另一个请求的第一个 token 后断开，服务端取消计数增加 1、活跃请求归零，两卡各 16 页全部可再分配。后续请求顺利生成 16 个 token。24 条请求的客户端首 token 中位数为 36.8 ms，P95 为 644.5 ms；最高的两个值均出现在第一次四并发。探针只有一次串行预热，未对这些首次长延迟做内核级归因。[逐请求 HTTP 记录](results/stream-http.json)

为观察首次并发与重复运行的区别，又重新启动服务，连续运行同一探针两次，中间保留进程与缓存。两次各 24 条请求，客户端首 token P95 分别为 995.9 ms 和 48.9 ms，两次断开后资源回收和后续生成均通过。三组共 72 条计时请求、3 次真实连接取消，每组样本均为 24 条。[连续两轮客户端实测](results/stream-repeat.json)
