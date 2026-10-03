"""Collect the measured benchmark results into artifacts/arena.json for the UI and /v1/arena."""
import json
import sys
from pathlib import Path

import numpy as np

home = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home()
sent = json.loads((home / "fin-lora/exports/arena_sentiment.json").read_text())
credit = json.loads((home / "credit-arena/results.json").read_text())
trade = json.loads((home / "trade-arena/results.json").read_text())["per_asset"]

assets = list(trade)
strategies = list(trade[assets[0]])
arena = {
    "sentiment": {
        "dataset": sent["dataset"],
        "models": [{"name": m["name"], "accuracy": m["accuracy"], "latency_ms": m["latency_ms"],
                    "detail": " · ".join(f"{k.split(' (')[0]} {v * 100:.0f}" for k, v in m["by_set"].items())} for m in sent["models"]],
        "cascade": sent["cascade"],
    },
    "credit": {
        "dataset": "30,000 card accounts, 6,000 held-out",
        "models": [{"name": k, "auc": v["auc"], "ci": v["auc_ci95"], "cost": v["cost_per_account"]}
                   for k, v in credit["models"].items()],
    },
    "trading": {
        "dataset": "10 assets, walk-forward 2015-2026, net of costs",
        "strategies": [{"name": s, "sharpe_net": round(float(np.mean([trade[a][s]["sharpe"] for a in assets])), 2),
                        "sharpe_gross": round(float(np.mean([trade[a][s]["sharpe_gross"] for a in assets])), 2)} for s in strategies],
    },
}
Path("artifacts/arena.json").write_text(json.dumps(arena, indent=2))
print("wrote artifacts/arena.json:", {k: len(v.get("models") or v.get("strategies")) for k, v in arena.items()})
