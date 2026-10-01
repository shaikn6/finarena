"""Build sentiment artifacts: the TF-IDF fast model, and copy in the LoRA adapter trained in fin-lora."""
import json
import shutil
import sys
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

out = Path("artifacts")
adapter_src = Path(sys.argv[1]) if len(sys.argv) > 1 else None
d = load_dataset("zeroshot/twitter-financial-news-sentiment")
pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(max_iter=2000, C=5))
pipe.fit(d["train"]["text"], d["train"]["label"])
pred = pipe.predict(d["validation"]["text"])
joblib.dump(pipe, out / "sentiment_tfidf.joblib", compress=3)
card = dict(sentiment=dict(data="zeroshot/twitter-financial-news-sentiment; test = official validation split (2,388)",
                           tfidf_accuracy=round(accuracy_score(d["validation"]["label"], pred), 4),
                           tfidf_macro_f1=round(f1_score(d["validation"]["label"], pred, average="macro"), 4)))
if adapter_src:
    shutil.copytree(adapter_src, out / "sentiment_lora", dirs_exist_ok=True)
cards = json.loads((out / "model_cards.json").read_text())
(out / "model_cards.json").write_text(json.dumps(cards | card, indent=2))
print(json.dumps(card, indent=2))
