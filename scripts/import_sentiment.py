"""Install the sentiment artifacts produced by a fin-lora checkout, and record its benchmark in the model cards.

usage: python scripts/import_sentiment.py ~/fin-lora      (after fin-lora's train_news.py and export_for_finarena.py)

Everything is checked and copied into a temporary directory first, so a failed import never leaves a half-installed model.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

src, out = Path(sys.argv[1]).expanduser(), Path("artifacts")
needed = {"fast model": src / "exports/sentiment_tfidf.joblib", "benchmark": src / "exports/arena_sentiment.json",
          "prompt": src / "exports/prompt.json", "adapter config": src / "adapters/news-0.5b/adapter_config.json",
          "adapter weights": src / "adapters/news-0.5b/adapter_model.safetensors"}
missing = [f"{what}: {path}" for what, path in needed.items() if not path.exists()]
if missing:
    sys.exit("cannot import, missing:\n  " + "\n  ".join(missing))

bench = json.loads(needed["benchmark"].read_text())
cards = json.loads((out / "model_cards.json").read_text())
cards["sentiment"] = dict(
    data="trained on NOSIBLE news snippets, tweets and the training split of the manually labeled headlines; tested on 3 held-out sets (the headline test split is never trained on)",
    benchmark={m["name"]: m["by_set"] for m in bench["models"]},
    cascade_threshold=bench["chosen_threshold"],
    cascade=next(r for r in bench["cascade"] if r["threshold"] == bench["chosen_threshold"]),
    latency_ms=bench["latency_ms"])

with tempfile.TemporaryDirectory(dir=out) as tmp:
    staged = Path(tmp) / "sentiment_lora"
    staged.mkdir()
    for f in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(src / "adapters/news-0.5b" / f, staged / f)
    shutil.copy(needed["prompt"], staged / "prompt.json")
    for f in staged.iterdir():
        f.chmod(0o644)
    shutil.copy(needed["fast model"], Path(tmp) / "sentiment_tfidf.joblib")
    shutil.rmtree(out / "sentiment_lora", ignore_errors=True)
    shutil.move(str(staged), out / "sentiment_lora")
    shutil.move(str(Path(tmp) / "sentiment_tfidf.joblib"), out / "sentiment_tfidf.joblib")
(out / "model_cards.json").write_text(json.dumps(cards, indent=2))
print("installed sentiment artifacts; cascade threshold", bench["chosen_threshold"])
