"""Credit default scorer: gradient boosting (accurate) and logistic regression with reason codes (explainable)."""
import numpy as np
import pandas as pd

FEATURES = ["limit_bal", "education", "marriage", "age"] + [f"pay_status_{i}" for i in range(1, 7)] + \
           [f"bill_amt_{i}" for i in range(1, 7)] + [f"pay_amt_{i}" for i in range(1, 7)]
REASON_TEXT = {
    "util": "High credit utilisation", "pay_ratio": "Low repayment relative to balances",
    "max_delinq": "Severe past delinquency", "n_late": "Multiple late payments", "limit_bal": "Low credit limit",
    "age": "Borrower age", "education": "Education level", "marriage": "Marital status",
}


def to_frame(apps):
    """List of CreditApplication -> model feature frame (same engineering as training)."""
    rows = []
    for a in apps:
        r = dict(limit_bal=a.limit_bal, education=a.education, marriage=a.marriage, age=a.age)
        for i in range(6):
            r[f"pay_status_{i + 1}"], r[f"bill_amt_{i + 1}"], r[f"pay_amt_{i + 1}"] = a.pay_status[i], a.bill_amt[i], a.pay_amt[i]
        rows.append(r)
    return engineer(pd.DataFrame(rows, columns=FEATURES))


def engineer(df):
    X = df.copy()
    bills, pays = [f"bill_amt_{i}" for i in range(1, 7)], [f"pay_amt_{i}" for i in range(1, 7)]
    delinq = [f"pay_status_{i}" for i in range(1, 7)]
    X["util"] = df[bills].mean(axis=1) / df["limit_bal"].clip(lower=1)
    X["pay_ratio"] = df[pays].sum(axis=1) / df[bills].sum(axis=1).clip(lower=1)
    X["max_delinq"] = df[delinq].max(axis=1)
    X["n_late"] = (df[delinq] > 0).sum(axis=1)
    return X


class CreditScorer:
    """Wraps the trained artifacts: {'hgb', 'lr', 'threshold', 'columns'}."""

    def __init__(self, bundle):
        self.hgb, self.lr = bundle["hgb"], bundle["lr"]
        self.threshold, self.columns = float(bundle["threshold"]), list(bundle["columns"])

    def score(self, apps, model="accurate"):
        X = to_frame(apps)[self.columns]
        if model == "explainable":
            name, p, codes = "logistic-regression", self.lr.predict_proba(X)[:, 1], self._reason_codes(X)
        else:
            name, p, codes = "hist-gradient-boosting", self.hgb.predict_proba(X)[:, 1], [None] * len(X)
        return [dict(probability_of_default=float(pi), decision="decline" if pi >= self.threshold else "approve",
                     model=name, reason_codes=c) for pi, c in zip(p, codes)]

    def _reason_codes(self, X, top=3):
        """Top risk-increasing contributions: coefficient x standardised value, mapped to plain language."""
        scaler, clf = self.lr.steps[0][1], self.lr.steps[-1][1]
        contrib = scaler.transform(X) * clf.coef_[0]
        out = []
        for row in contrib:
            picks = []
            for j in np.argsort(row)[::-1]:
                if row[j] <= 0 or len(picks) == top:
                    break
                col = self.columns[j]
                key = "max_delinq" if col.startswith("pay_status") else "util" if col.startswith("bill_amt") else \
                      "pay_ratio" if col.startswith("pay_amt") else col
                text = REASON_TEXT.get(key, col)
                if text not in picks:
                    picks.append(text)
            out.append(picks)
        return out
