"""End-to-end check: score held-out tweets through the real API and compare with the benchmark numbers."""
import json
import os
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from fastapi.testclient import TestClient

from finarena.config import Settings
from finarena.loader import load_registry
from finarena.main import create_app

N = int(os.environ.get("N", 400))
s = Settings(env="dev", artifact_dir=Path("artifacts"), max_batch=16)
reg = load_registry(s)
assert reg.sentiment and reg.sentiment.accurate, "LLM did not load"
client = TestClient(create_app(s, reg))
d = load_dataset("zeroshot/twitter-financial-news-sentiment")["validation"].shuffle(seed=0).select(range(N))
texts, y = d["text"], np.array(d["label"])
names = ["Bearish", "Bullish", "Neutral"]
out = {}
for strat in ["fast", "accurate", "auto"]:
    preds, esc, t = [], 0, time.perf_counter()
    for i in range(0, N, 16):
        r = client.post("/v1/sentiment", json={"texts": texts[i:i + 16], "strategy": strat}).json()
        preds += [names.index(x["label"]) for x in r["results"]]
        esc += sum(x["escalated"] for x in r["results"])
    out[strat] = dict(accuracy=round(float((np.array(preds) == y).mean()), 4), escalated=round(esc / N, 3),
                      seconds_per_100=round((time.perf_counter() - t) / N * 100, 2))
    print(strat, out[strat], flush=True)
json.dump(out, open("artifacts/api_verification.json", "w"), indent=2)
