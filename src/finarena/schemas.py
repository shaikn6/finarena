"""Request/response contracts. Validation lives here so handlers only see well-formed data."""
from typing import Annotated, Literal, get_args

from pydantic import AfterValidator, BaseModel, Field

Strategy = Literal["fast", "accurate", "auto"]
Label = Literal["Bearish", "Bullish", "Neutral"]
SENTIMENT_LABELS = get_args(Label)


def _not_blank(v):
    if not v.strip():
        raise ValueError("must not be blank")
    return v


NonBlankStr = Annotated[str, AfterValidator(_not_blank)]


class SentimentRequest(BaseModel):
    texts: list[NonBlankStr] = Field(min_length=1)
    strategy: Strategy = "auto"


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


class SentimentTask(BaseModel):
    task: Literal["sentiment"]
    text: NonBlankStr


class CreditTask(BaseModel):
    task: Literal["credit"]
    application: CreditApplication


class AnalyzeRequest(BaseModel):
    items: list[Annotated[SentimentTask | CreditTask, Field(discriminator="task")]] = Field(min_length=1)
    sentiment_strategy: Strategy = "auto"
    credit_model: Literal["accurate", "explainable"] = "accurate"


class AnalyzeResult(BaseModel):
    task: Literal["sentiment", "credit"]
    ok: bool
    result: dict | None = None
    error: str | None = None


class AnalyzeResponse(BaseModel):
    results: list[AnalyzeResult]
