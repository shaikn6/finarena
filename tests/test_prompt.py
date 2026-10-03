"""The serving prompt must be exactly what the adapter was trained with, and long inputs must keep the prompt's final cue."""
import json
from pathlib import Path

import pytest

from finarena.models.sentiment import TOKEN_MARGIN, fit_text_to_window, load_prompt_config

TEMPLATE = "Classify the sentiment of this text as Bearish, Bullish, or Neutral.\nText: {text}\nSentiment:"
ADAPTER = Path(__file__).parents[1] / "artifacts" / "sentiment_lora"


class WordTokenizer:
    """One token per whitespace-separated word: enough to test the windowing logic without downloading a model."""

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": texts.split()}
        return {"input_ids": [t.split() for t in texts]}

    def decode(self, ids):
        return " ".join(ids)


def write(tmp_path, **cfg):
    (tmp_path / "prompt.json").write_text(json.dumps(cfg))
    return tmp_path


def test_prompt_config_is_read_from_the_adapter_directory(tmp_path):
    assert load_prompt_config(write(tmp_path, template=TEMPLATE, max_len=256)) == (TEMPLATE, 256)


def test_adapter_without_its_prompt_file_refuses_to_load(tmp_path):
    with pytest.raises(FileNotFoundError, match="trained on"):
        load_prompt_config(tmp_path)


@pytest.mark.parametrize("cfg", [dict(template="no placeholder", max_len=256), dict(template=TEMPLATE, max_len=0)])
def test_malformed_prompt_file_is_rejected(tmp_path, cfg):
    with pytest.raises(ValueError, match="malformed"):
        load_prompt_config(write(tmp_path, **cfg))


@pytest.mark.skipif(not ADAPTER.exists(), reason="adapter not installed")
def test_the_shipped_adapter_carries_a_valid_prompt_file():
    template, max_len = load_prompt_config(ADAPTER)
    assert template.endswith("Sentiment:") and max_len == 256


def test_short_texts_are_returned_untouched():
    texts = ["Shares plunge after SEC probe", "Dividend notice"]
    assert fit_text_to_window(WordTokenizer(), texts, TEMPLATE, 256) == texts


def test_long_text_is_cut_but_the_prompt_still_ends_with_its_cue():
    tok, long_text = WordTokenizer(), " ".join(f"w{i}" for i in range(3000))
    (fitted,) = fit_text_to_window(tok, [long_text], TEMPLATE, 64)
    prompt = TEMPLATE.format(text=fitted)
    assert prompt.endswith("\nSentiment:")  # the model answers from the last position, so this must survive
    assert len(tok(prompt)["input_ids"]) <= 64  # and the whole prompt fits the window
    assert fitted.startswith("w0 w1")  # the beginning of the text is what is kept


def test_a_text_exactly_at_the_budget_is_not_cut():
    tok = WordTokenizer()
    budget = 64 - len(tok(TEMPLATE.format(text=""))["input_ids"]) - TOKEN_MARGIN
    exact = " ".join(f"w{i}" for i in range(budget))
    assert fit_text_to_window(tok, [exact], TEMPLATE, 64) == [exact]
    assert fit_text_to_window(tok, [exact + " extra"], TEMPLATE, 64) != [exact + " extra"]
