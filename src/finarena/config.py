"""Runtime configuration, read once from environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path


def _int(name, default):
    return int(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class Settings:
    env: str = "prod"
    api_keys: frozenset = field(default_factory=frozenset)
    artifact_dir: Path = Path("artifacts")
    max_batch: int = 64
    max_text_chars: int = 2000
    max_image_bytes: int = 10 * 1024 * 1024
    cascade_threshold: float = 0.8
    enable_llm: bool = True
    enable_signature: bool = False
    model_wait_seconds: float = 20.0
    require_llm: bool = False
    rate_per_minute: int = 120  # per client; 0 disables

    @classmethod
    def from_env(cls):
        keys = frozenset(k.strip() for k in os.environ.get("FINARENA_API_KEYS", "").split(",") if k.strip())
        env = os.environ.get("FINARENA_ENV", "prod")
        if env == "prod" and not keys:
            raise RuntimeError("FINARENA_API_KEYS must be set when FINARENA_ENV=prod (use FINARENA_ENV=dev to disable auth locally)")
        return cls(env=env, api_keys=keys, artifact_dir=Path(os.environ.get("FINARENA_ARTIFACT_DIR", "artifacts")),
                   max_batch=_int("FINARENA_MAX_BATCH", 64), max_text_chars=_int("FINARENA_MAX_TEXT_CHARS", 2000),
                   max_image_bytes=_int("FINARENA_MAX_IMAGE_BYTES", 10 * 1024 * 1024),
                   cascade_threshold=float(os.environ.get("FINARENA_CASCADE_THRESHOLD", "0.8")),
                   enable_llm=os.environ.get("FINARENA_ENABLE_LLM", "1") == "1",
                   enable_signature=os.environ.get("FINARENA_ENABLE_SIGNATURE", "0") == "1",
                   model_wait_seconds=float(os.environ.get("FINARENA_MODEL_WAIT_SECONDS", "20")),
                   require_llm=os.environ.get("FINARENA_REQUIRE_LLM", "0") == "1",
                   rate_per_minute=_int("FINARENA_RATE_PER_MINUTE", 120))
