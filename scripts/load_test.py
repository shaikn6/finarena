"""Concurrent load test against a running finarena: latency percentiles and error counts per strategy."""
import asyncio
import os
import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N, CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 60, int(sys.argv[3]) if len(sys.argv) > 3 else 8
TEXTS = ["$NVDA crushes earnings and raises guidance", "Fed leaves rates unchanged", "Bank shares tumble as losses mount",
         "Mixed signals: revenue up, margins squeezed, outlook unclear", "Company says it may or may not meet targets",
         "Offshore operator signs a non-binding letter of intent to lease a floating storage unit for up to 11 years, which could cut operating costs by 15-25% versus its current contract, though maintaining or replacing the existing vessel would still require substantial capital investment over the next several quarters. " * 2]
APP = {"limit_bal": 50000, "education": 2, "marriage": 1, "age": 35, "pay_status": [2, 2, 1, 0, 0, 0],
       "bill_amt": [30000] * 6, "pay_amt": [500] * 6}


async def one(client, path, body, lat, errs):
    t = time.perf_counter()
    try:
        r = await client.post(path, json=body, timeout=60)
        if r.status_code != 200:
            errs.append(r.status_code)
    except httpx.HTTPError as e:
        errs.append(type(e).__name__)
    lat.append((time.perf_counter() - t) * 1e3)


async def bench(name, path, make_body):
    lat, errs, sem = [], [], asyncio.Semaphore(CONC)
    async with httpx.AsyncClient(base_url=BASE, headers={"x-api-key": os.environ.get("FINARENA_API_KEY", "")}) as c:
        async def guarded(i):
            async with sem:
                await one(c, path, make_body(i), lat, errs)
        t = time.perf_counter()
        await asyncio.gather(*(guarded(i) for i in range(N)))
        wall = time.perf_counter() - t
    lat.sort()
    def q(p):
        return lat[min(len(lat) - 1, int(p * len(lat)))]

    print(f"{name:22s} n={N} conc={CONC} p50={q(.5):7.0f}ms p95={q(.95):7.0f}ms max={lat[-1]:7.0f}ms "
          f"rps={N / wall:5.1f} errors={len(errs)} {sorted(set(map(str, errs)))}", flush=True)


async def main():
    for strategy in ("fast", "auto", "accurate"):
        await bench(f"sentiment {strategy}", "/v1/sentiment", lambda i, s=strategy: {"texts": [TEXTS[i % 6]], "strategy": s})
    await bench("credit accurate", "/v1/credit/score", lambda i: {"applications": [APP], "model": "accurate"})

asyncio.run(main())
