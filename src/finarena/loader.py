"""Build the model registry from artifacts on disk. Missing optional models degrade to 'not loaded', never a crash."""
import json
import logging

import joblib

from finarena.config import Settings
from finarena.main import Registry
from finarena.models.concurrency import Serialized
from finarena.models.credit import CreditScorer
from finarena.models.sentiment import FastModel, LLMModel, SentimentService

log = logging.getLogger("finarena")
BASE_LLM = "Qwen/Qwen2.5-0.5B-Instruct"


def load_registry(s: Settings) -> Registry:
    d = s.artifact_dir
    reg = Registry(cards=json.loads((d / "model_cards.json").read_text()) if (d / "model_cards.json").exists() else {})
    if (d / "arena.json").exists():
        reg.arena = json.loads((d / "arena.json").read_text())
    if (d / "credit.joblib").exists():
        reg.credit = CreditScorer(joblib.load(d / "credit.joblib"))
    if (d / "sentiment_tfidf.joblib").exists():
        fast, accurate = FastModel(joblib.load(d / "sentiment_tfidf.joblib")), None
        if s.enable_llm and (d / "sentiment_lora").exists():
            try:
                accurate = Serialized(LLMModel(BASE_LLM, d / "sentiment_lora"), s.model_wait_seconds)
            except Exception as e:
                if s.require_llm:
                    raise
                if isinstance(e, ImportError):  # expected on the base image: one line, not a traceback
                    log.warning("LLM sentiment disabled: torch/transformers/peft are not installed (use the [llm] image); serving fast model only")
                else:
                    log.exception("LLM sentiment model failed to load; serving fast model only")
        reg.sentiment = SentimentService(fast, accurate, s.cascade_threshold)
    if s.enable_signature and (d / "signature.pt").exists():
        from finarena.models.signature import SignatureDetector
        reg.signature = Serialized(SignatureDetector(d / "signature.pt"), s.model_wait_seconds)
    return reg
