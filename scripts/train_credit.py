"""Train the credit artifacts (HGB + logistic regression + cost-optimal threshold) from the UCI dataset."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from finarena.models.credit import FEATURES, engineer

out = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
out.mkdir(exist_ok=True)
d = fetch_openml(data_id=42477, as_frame=True, parser="auto")
raw = d.data.drop(columns=["x2"])  # sex is never a feature
rename = {"x1": "limit_bal", "x3": "education", "x4": "marriage", "x5": "age"}
rename |= {f"x{5 + i}": f"pay_status_{i}" for i in range(1, 7)} | {f"x{11 + i}": f"bill_amt_{i}" for i in range(1, 7)} | {f"x{17 + i}": f"pay_amt_{i}" for i in range(1, 7)}
X = engineer(raw.rename(columns=rename)[FEATURES])
y = d.target.astype(int).values
Xtr, Xrest, ytr, yrest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
Xva, Xte, yva, yte = train_test_split(Xrest, yrest, test_size=0.5, stratify=yrest, random_state=0)
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)).fit(Xtr, ytr)

pva = hgb.predict_proba(Xva)[:, 1]
ts = np.unique(np.quantile(pva, np.linspace(0.05, 0.95, 91)))
def cost(t, p, yy):
    return (((p < t) & (yy == 1)).sum() * 5 + ((p >= t) & (yy == 0)).sum()) / len(yy)


threshold = float(ts[np.argmin([cost(t, pva, yva) for t in ts])])
card = dict(credit=dict(
    data="UCI Default of Credit Card Clients (30,000 accounts, 22% default), 60/20/20 split",
    test_auc_hgb=round(roc_auc_score(yte, hgb.predict_proba(Xte)[:, 1]), 4),
    test_auc_logreg=round(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]), 4),
    threshold=round(threshold, 4), sex_used_as_feature=False))
joblib.dump(dict(hgb=hgb, lr=lr, threshold=threshold, columns=list(X.columns)), out / "credit.joblib", compress=3)
cards = json.loads((out / "model_cards.json").read_text()) if (out / "model_cards.json").exists() else {}
(out / "model_cards.json").write_text(json.dumps(cards | card, indent=2))
print(json.dumps(card, indent=2))
