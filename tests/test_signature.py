import io
import pathlib

import pytest
from conftest import H
from fastapi.testclient import TestClient

from finarena.config import Settings
from finarena.main import Registry, create_app

ART = pathlib.Path(__file__).parents[1] / "artifacts" / "signature.pt"
IMG = pathlib.Path.home() / "datasets/signature/images/val/Frame_108.jpg"
pytestmark = pytest.mark.skipif(not (ART.exists() and IMG.exists()), reason="signature weights or sample image missing")


@pytest.fixture(scope="module")
def client():
    from finarena.models.signature import SignatureDetector

    s = Settings(env="dev", api_keys=frozenset({"secret"}), max_image_bytes=3_000_000)
    return TestClient(create_app(s, Registry(signature=SignatureDetector(ART))))


def test_detects_the_signature_in_a_real_document_photo(client):
    r = client.post("/v1/signature/detect", files={"file": ("a.jpg", IMG.read_bytes(), "image/jpeg")}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert len(body["detections"]) >= 1 and body["detections"][0]["confidence"] > 0.5
    x1, y1, x2, y2 = body["detections"][0]["xyxy"]
    assert 0 <= x1 < x2 <= body["image_width"] and 0 <= y1 < y2 <= body["image_height"]


def test_wrong_content_type_is_415(client):
    assert client.post("/v1/signature/detect", files={"file": ("a.txt", b"hi", "text/plain")}, headers=H).status_code == 415


def test_corrupt_image_is_422_not_500(client):
    r = client.post("/v1/signature/detect", files={"file": ("a.jpg", b"not an image at all", "image/jpeg")}, headers=H)
    assert r.status_code == 422


def test_oversize_image_is_413(client):
    big = io.BytesIO(b"\xff\xd8" + b"0" * 3_100_000)
    assert client.post("/v1/signature/detect", files={"file": ("a.jpg", big.getvalue(), "image/jpeg")}, headers=H).status_code == 413


def test_requires_api_key(client):
    assert client.post("/v1/signature/detect", files={"file": ("a.jpg", IMG.read_bytes(), "image/jpeg")}).status_code == 401


def test_concurrent_requests_all_succeed_without_crashing(client):
    from concurrent.futures import ThreadPoolExecutor

    data = IMG.read_bytes()

    def call(_):
        return client.post("/v1/signature/detect", files={"file": ("a.jpg", data, "image/jpeg")}, headers=H).status_code

    with ThreadPoolExecutor(8) as pool:
        assert set(pool.map(call, range(24))) == {200}
