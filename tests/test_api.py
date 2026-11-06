"""API behaviour, including the security fixes (path traversal, upload limits)."""
import io

import pytest

pytest.importorskip("torch")  # PyTorch tests are skipped in the serving-only (Vercel) environment
import torch  # noqa: E402
from fastapi.testclient import TestClient
from PIL import Image

import server
from src.config import CLASS_NAMES
from src.inference import InferenceService
from src.models import BrainTumorCNNv2, save_checkpoint
from src.preprocessing import DEFAULT_PREPROCESS
from tests.conftest import make_scan


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("app")
    models_dir, results_dir, public_dir = root / "models", root / "results", root / "public"
    samples_dir = public_dir / "samples"
    torch.manual_seed(0)
    save_checkpoint(models_dir / "custom_cnn_final_model.pth", BrainTumorCNNv2().eval(), "custom_cnn",
                    "custom_cnn_v2", DEFAULT_PREPROCESS, temperature=1.3, val_metrics={"val_acc": 0.9})
    for i, cls in enumerate(CLASS_NAMES):
        (samples_dir / cls).mkdir(parents=True)
        make_scan(100 + i, i).save(samples_dir / cls / f"{cls}_1.jpg", quality=95)
    results_dir.mkdir()
    (public_dir / "index.html").write_text("<!doctype html><title>NeuroScan</title>")
    settings = server.Settings(models_dir=models_dir, public_dir=public_dir, results_dir=results_dir,
                               max_upload_mb=1, serve_static=True)
    service = InferenceService(models_dir=models_dir, device=torch.device("cpu"), benchmark_path=None,
                               backend="torch").load()
    with TestClient(server.create_app(settings, service)) as test_client:
        yield test_client


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_status_only_list_real_checkpoints(client):
    assert client.get("/api/health").json() == {"status": "ok", "backend": "torch", "models_loaded": 1}
    status = client.get("/api/status").json()
    rows = {m["key"]: m for m in status["models"]}
    assert rows["custom_cnn"]["is_trained"] and rows["custom_cnn"]["temperature"] == pytest.approx(1.3)
    assert not rows["vit"]["is_trained"] and "not found" in rows["vit"]["unavailable_reason"]
    assert "models_dir" not in status  # no filesystem paths leaked


def test_predict_upload_returns_calibrated_ensemble_and_gradcam(client):
    res = client.post("/api/predict", files={"file": ("scan.png", _png_bytes(make_scan(5, 3)), "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] and data["total_models"] == 1
    assert sum(data["ensemble_probabilities"].values()) == pytest.approx(100.0, abs=0.1)
    assert data["confidence_level"] in {"high", "moderate", "low"}
    assert data["gradcam"]["heatmap_b64"].startswith("data:image/png;base64,")
    assert {m["model_key"] for m in data["unavailable_models"]} == {"resnet18", "mobilenet_v3", "vit"}
    assert "not a medical device" in data["disclaimer"]


def test_predict_known_sample(client):
    samples = client.get("/api/samples").json()["samples"]
    assert len(samples) == 4
    assert client.post("/api/predict", data={"sample_id": samples[0]["id"]}).status_code == 200


@pytest.mark.parametrize("sample_id", ["../../server.py", "glioma/../../../etc/passwd", "/etc/passwd", "nope/x.jpg"])
def test_sample_path_traversal_is_rejected(client, sample_id):
    res = client.post("/api/predict", data={"sample_id": sample_id})
    assert res.status_code == 404


def test_invalid_uploads_get_clear_errors(client):
    assert client.post("/api/predict").status_code == 400
    bad = client.post("/api/predict", files={"file": ("x.jpg", b"definitely not an image", "image/jpeg")})
    assert bad.status_code == 415 and "image" in bad.json()["error"]
    gif = io.BytesIO()
    Image.new("RGB", (64, 64)).save(gif, format="GIF")
    assert client.post("/api/predict", files={"file": ("x.gif", gif.getvalue(), "image/gif")}).status_code == 415
    tiny = client.post("/api/predict", files={"file": ("t.png", _png_bytes(Image.new("RGB", (8, 8))), "image/png")})
    assert tiny.status_code == 400


def test_oversized_upload_is_rejected(client):
    big = b"0" * (2 * 1024 * 1024)
    assert client.post("/api/predict", files={"file": ("big.jpg", big, "image/jpeg")}).status_code == 413


def test_metrics_never_fabricated(client):
    data = client.get("/api/metrics").json()
    assert data["status"] == "unavailable" and data["metrics"] == {}


def test_security_headers_and_frontend(client):
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["x-content-type-options"] == "nosniff"
    assert "script-src 'self'" in res.headers["content-security-policy"]


def test_reload_is_allowed_from_loopback_only(client):
    # TestClient's host is "testclient", which the server treats as local.
    assert client.post("/api/refresh").status_code == 200
