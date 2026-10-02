"""The README's configuration table must not drift from the real defaults in Settings."""
import re
from pathlib import Path

import pytest

from finarena.config import Settings

README = (Path(__file__).parents[1] / "README.md").read_text()
DEFAULTS = Settings()
CASES = {
    "FINARENA_ENV": DEFAULTS.env, "FINARENA_ARTIFACT_DIR": str(DEFAULTS.artifact_dir),
    "FINARENA_RATE_PER_MINUTE": DEFAULTS.rate_per_minute, "FINARENA_MAX_BATCH": DEFAULTS.max_batch,
    "FINARENA_MAX_TEXT_CHARS": DEFAULTS.max_text_chars, "FINARENA_MAX_IMAGE_BYTES": DEFAULTS.max_image_bytes,
    "FINARENA_CASCADE_THRESHOLD": DEFAULTS.cascade_threshold, "FINARENA_MODEL_WAIT_SECONDS": int(DEFAULTS.model_wait_seconds),
}


@pytest.mark.parametrize("var,default", CASES.items())
def test_readme_default_matches_settings(var, default):
    row = re.search(rf"\| `{var}` \| `([^`]+)` \|", README)
    assert row, f"{var} is missing from the README configuration table"
    assert row.group(1) == str(default)
