"""Credit features must be rejected at the boundary, never reach the model."""

import pytest
from conftest import H

BASE = {
    "limit_bal": 50_000,
    "education": 2,
    "marriage": 1,
    "age": 30,
    "pay_status": [0, 0, 0, 0, 0, 0],
    "bill_amt": [100, 100, 100, 100, 100, 100],
    "pay_amt": [50, 50, 50, 50, 50, 50],
}


def _with(field, value):
    app = dict(BASE)
    app[field] = value
    return {"applications": [app]}


@pytest.mark.parametrize(
    "field,value",
    [
        ("bill_amt", ["Infinity"] * 6),  # non-standard JSON constant as a string
        ("bill_amt", ["NaN"] * 6),
        ("bill_amt", [1e308] * 6),  # finite but over the bound
        ("pay_amt", ["Infinity"] * 6),
        ("pay_amt", [1e308] * 6),
        ("limit_bal", "Infinity"),
    ],
)
def test_non_finite_or_overflow_credit_features_rejected(make_client, field, value):
    # Before the bound, these flowed into StandardScaler and surfaced as a 500;
    # now they are a clean 422 at validation.
    resp = make_client().post("/v1/credit/score", json=_with(field, value), headers=H)
    assert resp.status_code == 422


def test_bare_infinity_json_token_is_422_not_500(make_client):
    # A bare (non-string) Infinity token is accepted by Python's JSON parser and
    # rejected by Pydantic, but the raw float would crash the default error
    # serializer -> 500. The safe validation handler must keep it a 422.
    raw = (
        '{"applications":[{"limit_bal":50000,"education":2,"marriage":1,"age":30,'
        '"pay_status":[0,0,0,0,0,0],'
        '"bill_amt":[Infinity,Infinity,Infinity,Infinity,Infinity,Infinity],'
        '"pay_amt":[50,50,50,50,50,50]}]}'
    )
    resp = make_client().post(
        "/v1/credit/score", content=raw, headers={**H, "content-type": "application/json"}
    )
    assert resp.status_code == 422
