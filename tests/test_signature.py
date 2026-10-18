import io
import os
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from conftest import H, build_client
from fastapi.testclient import TestClient

from finarena.config import Settings
from finarena.main import Registry, create_app

# The YOLO weights are AGPL-linked and not committed (see README, Licensing). The route, decoding and error handling are
# tested here against a stand-in model; the tests at the bottom run real inference when weights are supplied.
Image = pytest.importorskip("PIL.Image")
from finarena.models.signature import SignatureDetector  # noqa: E402


def png(w=64, h=48):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


class FakeYolo:
    """Mimics the part of ultralytics' result the detector reads: boxes with xyxy and conf."""

    def predict(self, img, conf, verbose):
        box = type("Box", (), dict(xyxy=np.array([[1.04, 2.0, 30.0, 20.0]]), conf=np.float64(0.91234567)))()
        return [type("Result", (), dict(boxes=[box]))()]


class FakeDetector(SignatureDetector):
    def __init__(self):
        self.model = FakeYolo()


@pytest.fixture(scope="module")
def stub_client():
    s = Settings(env="dev", api_keys=frozenset({"secret"}), max_image_bytes=3_000_000)
    return TestClient(create_app(s, Registry(signature=FakeDetector())))


def test_returns_detections_with_image_size(stub_client):
    r = stub_client.post("/v1/signature/detect", files={"file": ("a.png", png(64, 48), "image/png")}, headers=H)
    assert r.status_code == 200
    assert r.json() == {"detections": [{"xyxy": [1.0, 2.0, 30.0, 20.0], "confidence": 0.9123}], "image_width": 64, "image_height": 48}


def test_wrong_content_type_is_415(stub_client):
    assert stub_client.post("/v1/signature/detect", files={"file": ("a.txt", b"hi", "text/plain")}, headers=H).status_code == 415


def test_corrupt_image_is_422_not_500(stub_client):
    r = stub_client.post("/v1/signature/detect", files={"file": ("a.jpg", b"not an image at all", "image/jpeg")}, headers=H)
    assert r.status_code == 422


def test_oversize_image_is_413(stub_client):
    big = b"\xff\xd8" + b"0" * 3_100_000
    assert stub_client.post("/v1/signature/detect", files={"file": ("a.jpg", big, "image/jpeg")}, headers=H).status_code == 413


def test_requires_api_key(stub_client):
    assert stub_client.post("/v1/signature/detect", files={"file": ("a.png", png(), "image/png")}).status_code == 401


def test_unconfigured_detector_is_503_with_a_clear_message():
    r = build_client().post("/v1/signature/detect", files={"file": ("a.png", png(), "image/png")}, headers=H)
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


def test_ready_reports_signature_loaded():
    assert build_client(signature=FakeDetector()).get("/ready").json()["models"]["signature"] is True


def test_loader_leaves_signature_off_when_enabled_without_weights(tmp_path, caplog):
    from finarena.loader import load_registry

    reg = load_registry(Settings(env="dev", artifact_dir=tmp_path, enable_signature=True))
    assert reg.signature is None and "weights not found" in caplog.text


def test_loader_honours_the_weights_path_setting(tmp_path, monkeypatch):
    import finarena.models.signature as sig
    from finarena.loader import load_registry

    weights = tmp_path / "elsewhere.pt"
    weights.write_bytes(b"")
    seen = []
    monkeypatch.setattr(sig, "SignatureDetector", lambda path: seen.append(path) or FakeDetector())
    reg = load_registry(Settings(env="dev", artifact_dir=tmp_path, enable_signature=True, signature_weights=weights))
    assert reg.signature is not None and seen == [weights]


# Real inference: set FINARENA_SIGNATURE_WEIGHTS to a YOLO11 signature checkpoint (needs `pip install ".[signature]"`).
# FINARENA_SIGNATURE_SAMPLE optionally points at a document photo that contains a signature.
WEIGHTS = pathlib.Path(os.environ.get("FINARENA_SIGNATURE_WEIGHTS", "artifacts/signature.pt"))
SAMPLE = pathlib.Path(os.environ["FINARENA_SIGNATURE_SAMPLE"]) if "FINARENA_SIGNATURE_SAMPLE" in os.environ else None
real = pytest.mark.skipif(not WEIGHTS.exists(), reason="signature weights not supplied (FINARENA_SIGNATURE_WEIGHTS)")


@pytest.fixture(scope="module")
def real_client():
    pytest.importorskip("ultralytics")
    s = Settings(env="dev", api_keys=frozenset({"secret"}), max_image_bytes=3_000_000)
    return TestClient(create_app(s, Registry(signature=SignatureDetector(WEIGHTS))))


@real
def test_real_model_answers_200_with_in_bounds_boxes(real_client):
    data = SAMPLE.read_bytes() if SAMPLE else png(320, 240)
    r = real_client.post("/v1/signature/detect", files={"file": ("a.jpg", data, "image/jpeg" if SAMPLE else "image/png")}, headers=H)
    assert r.status_code == 200
    body = r.json()
    for d in body["detections"]:
        x1, y1, x2, y2 = d["xyxy"]
        assert 0 <= x1 < x2 <= body["image_width"] and 0 <= y1 < y2 <= body["image_height"]


@real
@pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(), reason="FINARENA_SIGNATURE_SAMPLE not set")
def test_real_model_finds_the_signature_in_a_document_photo(real_client):
    r = real_client.post("/v1/signature/detect", files={"file": ("a.jpg", SAMPLE.read_bytes(), "image/jpeg")}, headers=H)
    assert r.status_code == 200 and r.json()["detections"][0]["confidence"] > 0.5


@real
@pytest.mark.skipif(SAMPLE is None or not SAMPLE.exists(), reason="FINARENA_SIGNATURE_SAMPLE not set")
def test_real_model_concurrent_requests_all_succeed(real_client):
    data = SAMPLE.read_bytes()

    def call(_):
        return real_client.post("/v1/signature/detect", files={"file": ("a.jpg", data, "image/jpeg")}, headers=H).status_code

    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(call, range(24))) == {200}
