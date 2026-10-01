"""Production entrypoint: `uvicorn finarena.asgi:app`."""
import logging

from finarena.config import Settings
from finarena.loader import load_registry
from finarena.main import create_app

logging.basicConfig(level=logging.INFO, format="%(message)s")
_settings = Settings.from_env()
app = create_app(_settings, load_registry(_settings))
