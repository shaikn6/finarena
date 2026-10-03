import numpy as np
import pytest
from fastapi.testclient import TestClient

from finarena.config import Settings
from finarena.main import Registry, create_app
from finarena.models.sentiment import SentimentService


class StubModel:
    def __init__(self, name, probs):
        self.name, self._p = name, np.array(probs, dtype=float)

    def proba(self, texts):
        return np.tile(self._p, (len(texts), 1))


def build_client(keys=("secret",), fast=(0.9, 0.05, 0.05), accurate=(0.1, 0.8, 0.1), with_llm=True, sentiment=True,
                 credit=None, signature=None, arena=None, rate_per_minute=0, max_batch=4, max_text_chars=50):
    """A TestClient around a real app with stub models. `fast`/`accurate` are probability tuples or model objects."""
    s = Settings(env="dev", api_keys=frozenset(keys), max_batch=max_batch, max_text_chars=max_text_chars, max_image_bytes=1000,
                 cascade_threshold=0.8, rate_per_minute=rate_per_minute)
    model = lambda name, spec: StubModel(name, spec) if isinstance(spec, tuple) else spec  # noqa: E731
    svc = SentimentService(model("fast", fast), model("llm", accurate) if with_llm else None, s.cascade_threshold) if sentiment else None
    registry = Registry(sentiment=svc, credit=credit, signature=signature, cards={"x": 1}, arena=arena)
    return TestClient(create_app(s, registry), raise_server_exceptions=False)


@pytest.fixture
def make_client():
    return build_client


H = {"x-api-key": "secret"}
APP = dict(limit_bal=50000, education=2, marriage=1, age=35, pay_status=[0, 0, 0, 0, 0, 0],
           bill_amt=[1000] * 6, pay_amt=[500] * 6)
