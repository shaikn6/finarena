"""Financial-tweet sentiment: a fast TF-IDF model, an accurate LoRA-tuned LLM, and a confidence cascade."""
import numpy as np

from finarena.models.concurrency import ModelBusy, Serialized  # noqa: F401  (re-exported)
from finarena.schemas import SENTIMENT_LABELS

PROMPT = ("Classify the sentiment of this financial news tweet as Bearish, Bullish, or Neutral.\n"
          "Tweet: {text}\nSentiment:")


class FastModel:
    name = "tfidf-logreg"

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def proba(self, texts):
        return self.pipeline.predict_proba(texts)


class LLMModel:
    """Qwen2.5-0.5B-Instruct + LoRA adapter, scored over the three label tokens (no free-text generation)."""
    name = "qwen2.5-0.5b-lora"

    def __init__(self, base_id, adapter_dir, device=None, max_len=96):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(base_id)
        self.tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16 if self.device != "cpu" else torch.float32)
        self.model = PeftModel.from_pretrained(base, adapter_dir).eval().to(self.device)
        self.ids = [self.tok(" " + label, add_special_tokens=False).input_ids[0] for label in SENTIMENT_LABELS]
        self.max_len = max_len

    def proba(self, texts, bs=16):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = self.tok([PROMPT.format(text=t) for t in texts[i:i + bs]], return_tensors="pt", padding=True,
                               truncation=True, max_length=self.max_len).to(self.device)
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :][:, self.ids].float()
                out.append(self.torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(out)


class SentimentService:
    def __init__(self, fast, accurate=None, threshold=0.8):
        self.fast, self.accurate, self.threshold = fast, accurate, threshold

    def predict(self, texts, strategy="auto"):
        if strategy == "auto" and self.accurate is None:
            strategy = "fast"  # no LLM on this deployment: the cascade degrades to the fast model
        if strategy == "accurate" and self.accurate is None:
            raise LookupError("accurate model is not available on this deployment")
        n = len(texts)
        if strategy == "accurate":
            probs, models, esc = self.accurate.proba(texts), [self.accurate.name] * n, np.zeros(n, bool)
        else:
            probs, models, esc = self.fast.proba(texts), [self.fast.name] * n, np.zeros(n, bool)
            if strategy == "auto":
                esc = probs.max(1) < self.threshold
                for i in np.flatnonzero(esc):
                    models[i] = self.accurate.name
                if esc.any():
                    probs[esc] = self.accurate.proba([texts[i] for i in np.flatnonzero(esc)])
        return [dict(label=SENTIMENT_LABELS[int(p.argmax())], confidence=float(p.max()),
                     probabilities={lab: float(v) for lab, v in zip(SENTIMENT_LABELS, p)}, model=m, escalated=bool(e))
                for p, m, e in zip(probs, models, esc)]
