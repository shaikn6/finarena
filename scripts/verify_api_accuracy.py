"""End-to-end check: score held-out text through the real API code path and compare with the fin-lora benchmark."""
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

N_TWEETS = int(os.environ.get("N", 400))
s = Settings(env="dev", artifact_dir=Path("artifacts"), max_batch=16, rate_per_minute=0)
reg = load_registry(s)
assert reg.sentiment and reg.sentiment.accurate, "LLM did not load"
client = TestClient(create_app(s, reg))
names = ["Bearish", "Bullish", "Neutral"]

tw = load_dataset("zeroshot/twitter-financial-news-sentiment")["validation"].shuffle(seed=0).select(range(N_TWEETS))
jb = load_dataset("Jean-Baptiste/financial_news_sentiment")["test"]  # manually labeled headlines; this test split is never trained on
jb_map = {0: 0, 2: 1, 1: 2}  # negative -> Bearish, positive -> Bullish, neutral -> Neutral
sets = {"tweets (sample)": (list(tw["text"]), list(tw["label"])), "manual headlines": (list(jb["title"]), [jb_map[int(x)] for x in jb["labels"]])}

out = {}
for set_name, (texts, y) in sets.items():
    for strat in ("fast", "accurate", "auto"):
        preds, esc, t = [], 0, time.perf_counter()
        for i in range(0, len(texts), 16):
            r = client.post("/v1/sentiment", json={"texts": texts[i:i + 16], "strategy": strat}).json()
            preds += [names.index(x["label"]) for x in r["results"]]
            esc += sum(x["escalated"] for x in r["results"])
        out[f"{set_name} / {strat}"] = dict(n=len(texts), accuracy=round(float((np.array(preds) == np.array(y)).mean()), 4),
                                            escalated=round(esc / len(texts), 3), seconds_per_100=round((time.perf_counter() - t) / len(texts) * 100, 2))
        print(f"{set_name:18s} {strat:9s}", out[f"{set_name} / {strat}"], flush=True)
json.dump(out, open("artifacts/api_verification.json", "w"), indent=2)
