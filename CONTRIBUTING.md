# Contributing

Thanks for taking the time to contribute.

## Development setup

```bash
make install                       # .venv with the package and dev tools
make test                          # pytest
.venv/bin/ruff check src tests scripts
FINARENA_ENV=dev make run          # http://localhost:8000, auth off for local use only
```

The trained models in `artifacts/` are committed, so the tests need no training
step and no network. The LLM tier is optional: `.venv/bin/pip install -e ".[dev,llm]"`.

## Ground rules

- **One logical change per PR.** Keep diffs reviewable.
- **Tests are not optional.** New behaviour needs a test; bug fixes need a test
  that fails before the fix.
- **The build must be green:** `ruff check src tests scripts` and `make test`
  pass, and `make docker` still builds.
- **Model changes come from their source repo.** Sentiment models are trained in
  fin-lora and imported with `scripts/import_sentiment.py`; the credit model is
  rebuilt with `make train`. A changed metric in the README or model cards must
  come from the script that produced it.
- **Optional models stay optional.** A missing model degrades its endpoint; it
  must not stop the service unless the operator asked for that.

## Commit messages

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).

## Reporting bugs / proposing features

Use the issue templates. For anything security-related, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
