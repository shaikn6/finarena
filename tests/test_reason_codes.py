"""Unit tests for credit reason-code generation (CreditScorer._reason_codes).

The method is driven by a hand-built scaler/classifier so the expected plain-language output is derivable by hand,
with no dependency on the trained `artifacts/credit.joblib`. Contribution = standardised value x coefficient; the
top risk-increasing (positive) contributions are mapped to plain language, de-duplicated, and capped.
"""
from types import SimpleNamespace

import numpy as np

from finarena.models.credit import CreditScorer


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


def scorer(columns, coef):
    """A CreditScorer stand-in exposing only what _reason_codes reads: a 2-step `lr` pipeline and `columns`."""
    clf = SimpleNamespace(coef_=np.array([coef], dtype=float))
    lr = SimpleNamespace(steps=[("scaler", IdentityScaler()), ("clf", clf)])
    return SimpleNamespace(lr=lr, columns=list(columns))


def one(self, row):
    """Reason codes for a single application (the method returns one list per row)."""
    return CreditScorer._reason_codes(self, np.array([row], dtype=float))[0]


def test_orders_by_contribution_dedupes_delinquency_and_caps_at_three():
    s = scorer(
        ["pay_status_1", "pay_status_2", "bill_amt_1", "pay_amt_1", "age"],
        coef=[1, 1, 1, 1, 1],
    )
    # Descending contributions: pay_status_1 (5), pay_status_2 (4, same label -> dropped), bill_amt_1 (3),
    # pay_amt_1 (2) -> three distinct picks, cap reached before `age`.
    assert one(s, [5, 4, 3, 2, 1]) == [
        "Severe past delinquency",             # pay_status_* -> max_delinq
        "High credit utilisation",             # bill_amt_* -> util
        "Low repayment relative to balances",  # pay_amt_* -> pay_ratio
    ]


def test_only_positive_contributions_count():
    s = scorer(["limit_bal", "age"], coef=[1, 1])
    # Only limit_bal is risk-increasing (positive); age is protective here (negative) and must not appear.
    assert one(s, [2, -3]) == ["Low credit limit"]


def test_an_all_negative_row_returns_no_reasons():
    s = scorer(["limit_bal", "age"], coef=[1, 1])
    assert one(s, [-1, -2]) == []


def test_unknown_column_falls_back_to_its_own_name():
    s = scorer(["mystery_feature"], coef=[1])
    assert one(s, [3]) == ["mystery_feature"]


def test_coefficient_sign_decides_which_direction_is_risky():
    # A negative coefficient means a low value raises default risk: contribution = value x coef.
    s = scorer(["limit_bal", "age"], coef=[-1, 1])
    assert one(s, [-5, -1]) == ["Low credit limit"]  # (-5)*(-1)=5 positive; age (-1)*1 negative -> dropped


def test_returns_one_list_of_codes_per_row_in_order():
    s = scorer(["limit_bal", "age"], coef=[1, 1])
    out = CreditScorer._reason_codes(s, np.array([[5, 0], [-1, -1]], dtype=float))
    assert out == [["Low credit limit"], []]
