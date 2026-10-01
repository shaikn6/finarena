.PHONY: install train test run docker
install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
train:
	.venv/bin/python scripts/train_credit.py
test:
	.venv/bin/python -m pytest -q
run:
	FINARENA_ENV=dev .venv/bin/uvicorn finarena.asgi:app --reload
docker:
	docker build -t finarena .
