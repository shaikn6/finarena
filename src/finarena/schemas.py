"""Request/response contracts. Validation lives here so handlers only see well-formed data."""
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, field_validator

Strategy = Literal["fast", "accurate", "auto"]
Label = Literal["Bearish", "Bullish", "Neutral"]
SENTIMENT_LABELS = get_args(Label)


class SentimentRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    strategy: Strategy = "auto"

    @field_validator("texts")
    @classmethod
    def non_blank(cls, v):
        if any(not t.strip() for t in v):
            raise ValueError("texts must not contain blank strings")
        return v


class SentimentItem(BaseModel):
    label: Label
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model: str
    escalated: bool


class SentimentResponse(BaseModel):
    results: list[SentimentItem]
    strategy: Strategy
    escalated_fraction: float


class CreditApplication(BaseModel):
    """Account snapshot. Sex is deliberately not accepted: it is not a model input (fair-lending)."""
    limit_bal: float = Field(gt=0, le=10_000_000)
    education: int = Field(ge=0, le=6)
    marriage: int = Field(ge=0, le=3)
    age: int = Field(ge=18, le=120)
    pay_status: list[Annotated[int, Field(ge=-2, le=9)]] = Field(min_length=6, max_length=6, description="Repayment status, Sep..Apr (-2..9)")
    bill_amt: list[float] = Field(min_length=6, max_length=6)
    pay_amt: list[Annotated[float, Field(ge=0)]] = Field(min_length=6, max_length=6, description="Payments made, Sep..Apr (>= 0)")


class CreditRequest(BaseModel):
    applications: list[CreditApplication] = Field(min_length=1)
    model: Literal["accurate", "explainable"] = "accurate"


class CreditItem(BaseModel):
    probability_of_default: float = Field(ge=0, le=1)
    decision: Literal["approve", "decline"]
    model: str
    reason_codes: list[str] | None = None


class CreditResponse(BaseModel):
    results: list[CreditItem]
    threshold: float
    cost_assumption: str


class Detection(BaseModel):
    xyxy: list[float]
    confidence: float


class SignatureResponse(BaseModel):
    detections: list[Detection]
    image_width: int
    image_height: int
