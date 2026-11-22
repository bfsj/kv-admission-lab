"""Client-side loopback probe for an otherwise idle running stream_api."""

import argparse
import asyncio
import json
from pathlib import Path
from time import perf_counter

import aiohttp
from replay import p95


async def frames(response):
    response.raise_for_status()
    event = data = None
    async for line in response.content:
        text = line.decode().rstrip("\r\n")
        if text.startswith("event: "):
            event = text[7:]
        elif text.startswith("data: "):
            data = json.loads(text[6:])
        elif not text and event is not None:
            yield event, data
            event = data = None


async def probe(url, count):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async def metrics():
            async with session.get(url+"/metrics") as r:
                r.raise_for_status()
                return await r.json()
        async def request(i, limit=16, abort=False):
            prompt = f"请用中文逐项解释订单 {i} 的退款审核流程，并给出四个具体示例。"
            started = perf_counter()
            async with session.post(url+"/generate", json=dict(prompt=prompt,max_tokens=limit)) as r:
                tokens, chunks, first_ms = [], [], None
                async for event, data in frames(r):
                    if event == "token":
                        if first_ms is None:
                            first_ms = (perf_counter()-started)*1000
                        tokens.append(data["token_id"])
                        chunks.append(data["text"])
                        if abort:
                            r.close()
                            return {"received":len(tokens), "ttft_ms":first_ms}
                    elif event == "done":
                        if len(tokens) != data["generated"] or not (0 < len(tokens) <= limit):
                            raise AssertionError("invalid output length")
                        return dict(id=i, ttft_ms=first_ms, elapsed_ms=(perf_counter()-started)*1000,
                                    tokens=len(tokens), reason=data["reason"],
                                    text_matches="".join(chunks)+data["text_tail"]==data["text"])
                    else:
                        raise RuntimeError(str(data))
                raise RuntimeError("stream ended before done")
        before = await metrics()
        if before["active"] or any(g["pinned"] for g in before["replicas"]):
            raise ValueError("probe expects idle replicas")
        await request(-1)
        rows = []
        for offset in range(0, count, 4):
            rows.extend(await asyncio.gather(*(request(i) for i in range(offset,min(offset+4,count)))))
        cancelled = await request(count, limit=256, abort=True)
        for _ in range(200):
            after_abort = await metrics()
            if after_abort["active"] == 0 and after_abort["cancelled"] == before["cancelled"]+1:
                break
            await asyncio.sleep(0.025)
        else:
            raise AssertionError("disconnect was not reclaimed")
        followup = await request(count+1)
        end = await metrics()
        if (not end["healthy"] or end["active"] or
                end["completed"] != before["completed"]+count+2 or
                any(g["pinned"] for g in end["replicas"]) or
                not all(r["text_matches"] for r in rows+[followup])):
            raise AssertionError("post-disconnect service verification failed")
        return dict(concurrency=4, requests=rows, ttft_p95_ms=p95([r["ttft_ms"] for r in rows]),
                    abort=cancelled, before=before, after_abort=after_abort, followup=followup, after=end)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.count < 1:
        p.error("positive count required")
    args.output.write_text(json.dumps(asyncio.run(probe(args.url.rstrip("/"), args.count)),indent=2)+"\n")
