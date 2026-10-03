"""The one place that runs the specialist models, shared by the single-task endpoints and the mixed-batch router."""
import logging

from finarena.models.concurrency import ModelBusy

log = logging.getLogger("finarena")


class Unavailable(RuntimeError):
    """A model (or a strategy) this deployment does not have."""


def text_too_long_message(settings):
    return f"text too long (max {settings.max_text_chars} chars)"


def run_sentiment(registry, texts, strategy):
    if registry.sentiment is None:
        raise Unavailable("sentiment models not loaded")
    try:
        return registry.sentiment.predict(texts, strategy)
    except LookupError as e:
        raise Unavailable(str(e)) from e


def run_credit(registry, applications, model):
    if registry.credit is None:
        raise Unavailable("credit model not loaded")
    return registry.credit.score(applications, model)


def analyze(items, registry, settings, sentiment_strategy, credit_model, metrics):
    """Return one result per item, in order: {"task", "ok", "result" | "error"}.

    Items of the same task go to their model in one call. Whatever goes wrong with one task (model missing, a busy
    model, an unexpected exception) fails only that task's items; an item that is individually invalid (a text that
    is too long) fails alone.
    """
    results = [None] * len(items)

    def fail(i, error):
        results[i] = {"task": items[i].task, "ok": False, "error": error}

    def run_group(task, indices, run):
        """Run one task's batch, turning its failures into per-item errors."""
        if not indices:
            return
        try:
            outputs = run([items[i] for i in indices])
        except Unavailable as e:
            outputs, error = None, str(e)
        except ModelBusy as e:
            metrics.inc("finarena_model_busy_total")
            outputs, error = None, str(e)
        except Exception:  # noqa: BLE001 - one task's crash must not fail the other tasks' items
            log.exception("%s failed inside /v1/analyze", task)
            outputs, error = None, f"{task} failed"
        for n, i in enumerate(indices):
            if outputs is None:
                fail(i, error)
            else:
                results[i] = {"task": task, "ok": True, "result": outputs[n]}

    sentiment, credit = [], []
    for i, item in enumerate(items):
        if item.task == "credit":
            credit.append(i)
        elif len(item.text) > settings.max_text_chars:
            fail(i, text_too_long_message(settings))
        else:
            sentiment.append(i)

    run_group("sentiment", sentiment, lambda batch: run_sentiment(registry, [it.text for it in batch], sentiment_strategy))
    run_group("credit", credit, lambda batch: run_credit(registry, [it.application for it in batch], credit_model))
    return results
