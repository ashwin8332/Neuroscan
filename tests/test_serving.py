"""Serving stack tests: run with ONLY the deployment dependencies (no PyTorch).

These use the real ONNX models in models/onnx/ and the real public/ folder,
i.e. exactly what is deployed to Vercel.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from src.config import CLASS_NAMES, MODELS_DIR, PUBLIC_DIR, RESULTS_DIR  # noqa: E402
from src.preprocessing import to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = json.loads((PUBLIC_DIR / "samples" / "index.json").read_text())["samples"]
pytestmark = pytest.mark.skipif(not (MODELS_DIR / "onnx" / "manifest.json").exists(), reason="no ONNX export")


def strict_json(response):
    """Parse a response body as strict JSON (NaN/Infinity are invalid in browsers)."""
    def reject(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(response.text, parse_constant=reject)


def png_bytes(img: Image.Image, **save_kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", **save_kwargs)
    return buf.getvalue()


def encode(img: Image.Image, fmt: str, **save_kwargs) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def sample_path(sample_id: str) -> Path:
    return PUBLIC_DIR / "samples" / sample_id


@pytest.fixture(scope="module")
def client():
    settings = server.Settings(max_upload_mb=4, serve_static=True)
    with TestClient(server.create_app(settings)) as c:
        yield c


def predict(client, data: bytes, name: str = "scan.png", mime: str = "image/png"):
    return client.post("/api/predict", files={"file": (name, data, mime)})


# --------------------------------------------------------------------------- #
# Models and known results
# --------------------------------------------------------------------------- #
def test_onnx_backend_serves_all_exported_models(client):
    health = client.get("/api/health").json()
    manifest = json.loads((MODELS_DIR / "onnx" / "manifest.json").read_text())
    assert health == {"status": "ok", "backend": "onnx", "models_loaded": len(manifest["models"])}
    status = client.get("/api/status").json()
    assert status["backend"] == "onnx" and status["max_upload_mb"] == 4
    trained = {row["key"]: row for row in status["models"] if row["is_trained"]}
    assert set(trained) == set(manifest["models"])
    for row in status["models"]:
        if row["key"] in manifest.get("excluded", {}):
            assert row["unavailable_reason"].startswith("not deployed")
    assert abs(sum(row["ensemble_weight"] for row in trained.values()) - 1.0) < 1e-3


@pytest.mark.parametrize("sample", SAMPLES, ids=[s["id"] for s in SAMPLES])
def test_every_sample_scan_gets_a_valid_result(client, sample):
    res = predict(client, sample_path(sample["id"]).read_bytes(), sample["filename"], "image/jpeg")
    assert res.status_code == 200, res.text
    data = strict_json(res)
    assert data["backend"] == "onnx" and data["consensus_prediction"] in CLASS_NAMES
    assert abs(sum(data["ensemble_probabilities"].values()) - 100.0) < 0.1
    assert data["confidence_level"] in {"high", "moderate", "low"}
    assert data["gradcam"] and data["gradcam"]["heatmap_b64"].startswith("data:image/png;base64,")
    assert all(0.0 <= p <= 100.0 for row in data["model_results"] for p in row["confidences"].values())


def test_known_results_do_not_regress(client):
    """Fixed outputs of the shipped models (guards preprocessing/ONNX regressions)."""
    pituitary = strict_json(predict(client, sample_path("pituitary/Te-pi_10.jpg").read_bytes()))
    assert pituitary["consensus_prediction"] == "pituitary" and pituitary["confidence_level"] == "high"
    # The glioma scan the original app called "healthy" must stay inconclusive.
    glioma = strict_json(predict(client, sample_path("glioma/Te-gl_1.jpg").read_bytes()))
    assert glioma["inconclusive"] and glioma["gradcam"]["target_class"] == "glioma"


# --------------------------------------------------------------------------- #
# Image edge cases
# --------------------------------------------------------------------------- #
def _base_scan() -> Image.Image:
    """A real sample as 8-bit greyscale with an exact 0..255 range."""
    arr = np.asarray(Image.open(sample_path("meningioma/Te-aug-me_1.jpg")).convert("L")).copy()
    arr[0, 0], arr[0, 1] = 0, 255
    return Image.fromarray(arr, mode="L")


def _probs(client, data: bytes):
    res = predict(client, data)
    assert res.status_code == 200, res.text
    body = strict_json(res)
    return body["ensemble_probabilities"], body


def test_exif_rotated_photo_is_turned_upright(client):
    base = _base_scan().convert("RGB")
    rotated = base.rotate(90, expand=True)  # stored sideways ...
    exif = Image.Exif()
    exif[0x0112] = 6                        # ... with "rotate 90 CW to view" orientation
    upright, _ = _probs(client, png_bytes(base))
    tagged, _ = _probs(client, png_bytes(rotated, exif=exif))
    assert tagged == upright


def test_16_bit_png_is_scaled_not_clipped(client):
    base = _base_scan()
    wide = Image.fromarray((np.asarray(base, dtype=np.uint16) * 257).astype(np.uint16))
    assert wide.mode.startswith("I")
    as_16bit, body = _probs(client, png_bytes(wide))
    as_8bit, _ = _probs(client, png_bytes(base))
    assert as_16bit == as_8bit
    assert not any("contrast" in w for w in body["warnings"])


def test_transparency_is_composited_on_black():
    rgba = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
    rgba.paste((200, 200, 200, 255), (16, 16, 48, 48))
    arr = np.asarray(to_rgb(rgba))
    assert arr[0, 0].tolist() == [0, 0, 0] and arr[32, 32].tolist() == [200, 200, 200]


def test_opaque_rgba_matches_rgb(client):
    base = _base_scan().convert("RGB")
    opaque = base.copy()
    opaque.putalpha(255)
    assert _probs(client, png_bytes(opaque))[0] == _probs(client, png_bytes(base))[0]


@pytest.mark.parametrize("variant", ["palette", "cmyk_jpeg", "greyscale", "multi_frame_tiff", "webp", "bmp"])
def test_other_formats_and_modes_are_accepted(client, variant):
    base = _base_scan()
    if variant == "palette":
        data, name = png_bytes(base.convert("RGB").convert("P", palette=Image.ADAPTIVE)), "p.png"
    elif variant == "cmyk_jpeg":
        data, name = encode(base.convert("CMYK"), "JPEG"), "c.jpg"
    elif variant == "greyscale":
        data, name = png_bytes(base), "g.png"
    elif variant == "multi_frame_tiff":
        data, name = encode(base, "TIFF", save_all=True, append_images=[base.rotate(45)]), "m.tif"
    elif variant == "webp":
        data, name = encode(base.convert("RGB"), "WEBP", quality=95), "w.webp"
    else:
        data, name = encode(base.convert("RGB"), "BMP"), "b.bmp"
    res = predict(client, data, name)
    assert res.status_code == 200, res.text
    assert strict_json(res)["success"] is True


def test_blank_and_photo_inputs_are_flagged(client):
    blank = strict_json(predict(client, png_bytes(Image.new("L", (256, 256), 0))))
    assert any("head/brain region" in w for w in blank["warnings"])
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8))
    assert any("colour" in w for w in strict_json(predict(client, png_bytes(photo)))["warnings"])


def test_extreme_aspect_ratio(client):
    strip = Image.new("L", (2000, 60), 30)
    assert predict(client, png_bytes(strip)).status_code == 200


@pytest.mark.parametrize("payload,status", [
    (b"", 400),                                  # empty file
    (b"\x89PNG\r\n\x1a\n not really a png", 415),  # corrupt header
    (b"%PDF-1.7 fake", 415),                     # wrong type, renamed .png
])
def test_bad_uploads_are_rejected_cleanly(client, payload, status):
    res = predict(client, payload)
    assert res.status_code == status and "error" in strict_json(res)


def test_truncated_jpeg_is_rejected(client):
    data = sample_path("glioma/Te-gl_10.jpg").read_bytes()
    res = predict(client, data[: len(data) // 2], "cut.jpg", "image/jpeg")
    assert res.status_code == 415


def test_gif_tiny_and_bomb_images(client):
    assert predict(client, encode(Image.new("L", (64, 64)), "GIF"), "a.gif").status_code == 415
    assert predict(client, png_bytes(Image.new("L", (16, 16)))).status_code == 400
    # 7000x7000 (49 MP) is above the 40 MP limit but below Pillow's 2x hard error.
    bomb = png_bytes(Image.new("1", (7000, 7000)))
    assert len(bomb) < 4 * 1024 * 1024
    res = predict(client, bomb)
    assert res.status_code == 413 and "too large" in strict_json(res)["error"]


def test_oversized_upload_is_rejected(client):
    res = predict(client, b"0" * (5 * 1024 * 1024))
    assert res.status_code == 413


# --------------------------------------------------------------------------- #
# API behaviour
# --------------------------------------------------------------------------- #
def test_api_errors_are_json(client):
    for res, code in [
        (client.get("/api/does-not-exist"), 404),
        (client.get("/api/predict"), 405),
        (client.post("/api/predict"), 400),
        (client.post("/api/predict", data={"sample_id": "../server.py"}), 404),
    ]:
        assert res.status_code == code
        assert isinstance(strict_json(res)["error"], str)


def test_file_takes_precedence_over_sample_id(client):
    data = sample_path("notumor/Te-no_1.jpg").read_bytes()
    res = client.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")},
                      data={"sample_id": "pituitary/Te-pi_10.jpg"})
    assert strict_json(res)["consensus_prediction"] == "notumor"


def test_security_and_cache_headers(client):
    res = client.get("/api/health")
    assert res.headers["x-content-type-options"] == "nosniff"
    assert res.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in res.headers["content-security-policy"]


def test_local_server_serves_the_static_site(client):
    page = client.get("/")
    assert page.status_code == 200 and "NeuroScan" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/samples/index.json").json()["samples"] == SAMPLES


def test_metrics_with_nan_are_strict_json(tmp_path):
    (tmp_path / "model_comparison_metrics.json").write_text(
        '{"resnet18": {"display_name": "ResNet-18", "auc_macro": NaN, "ece": Infinity, "test_accuracy_pct": 95.0}}')
    (tmp_path / "resnet18_history.json").write_text('{"model_name": "resnet18", "val_loss": [0.5, NaN]}')
    app = server.create_app(server.Settings(results_dir=tmp_path, serve_static=False))
    res = TestClient(app).get("/api/metrics")
    body = strict_json(res)
    assert res.status_code == 200 and body["status"] == "available"
    assert body["metrics"]["resnet18"]["auc_macro"] is None and body["histories"]["resnet18"]["val_loss"] == [0.5, None]


def test_real_metrics_file_is_served(client):
    body = strict_json(client.get("/api/metrics"))
    if (RESULTS_DIR / "model_comparison_metrics.json").exists():
        assert body["status"] == "available" and body["metrics"]


def test_models_load_lazily_without_lifespan():
    """Serverless platforms may skip ASGI lifespan events."""
    app = server.create_app(server.Settings(serve_static=False))
    no_lifespan = TestClient(app)  # not used as a context manager -> lifespan never runs
    assert no_lifespan.get("/api/health").json()["backend"] == "onnx"


def test_vercel_mode_leaves_static_files_to_the_cdn():
    app = server.create_app(server.Settings(serve_static=False))
    c = TestClient(app)
    assert c.get("/").status_code == 404
    assert c.get("/static/app.js").status_code == 404
    assert c.get("/api/health").status_code == 200


# --------------------------------------------------------------------------- #
# The deployed import graph needs no PyTorch / SciPy / scikit-learn
# --------------------------------------------------------------------------- #
BLOCKER = textwrap.dedent("""
    import importlib.abc, sys
    BLOCKED = {"torch", "torchvision", "scipy", "sklearn", "matplotlib", "pandas", "seaborn", "tqdm"}
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"{name} is not available on Vercel")
            return None
    sys.meta_path.insert(0, Block())
""")


def test_vercel_runtime_works_without_training_dependencies():
    script = BLOCKER + textwrap.dedent("""
        import json
        from fastapi.testclient import TestClient
        import server
        s = server.Settings()
        assert s.max_upload_mb == 4.0 and s.serve_static is False, (s.max_upload_mb, s.serve_static)
        c = TestClient(server.app)
        data = open("public/samples/pituitary/Te-pi_10.jpg", "rb").read()
        r = c.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["backend"] == "onnx" and body["consensus_prediction"] == "pituitary", body
        assert c.get("/").status_code == 404   # the CDN serves public/ on Vercel
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
        print(json.dumps({"ok": True, "heavy_modules": loaded}))
    """)
    env = {**os.environ, "VERCEL": "1", "NEUROSCAN_MAX_UPLOAD_MB": "10"}
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"ok": True, "heavy_modules": []}


def test_json_writer_is_strict(tmp_path):
    import numpy as np

    from src.jsonutil import write_json

    write_json(tmp_path / "x.json", {"a": float("nan"), "b": [np.float32(1.5), float("inf")], "c": np.int64(3)})
    assert strict_json(type("R", (), {"text": (tmp_path / "x.json").read_text()})()) == {"a": None, "b": [1.5, None], "c": 3}


def test_head_request_on_health(client):
    assert client.head("/api/health").status_code == 200


# --------------------------------------------------------------------------- #
# Deployment settings: a typo in an environment variable must not take the app down
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("", True), ("1", True), ("Yes", True), ("ON", True),
    ("0", False), ("false", False), ("No", False), (" off ", False),
    ("maybe", True),  # invalid -> default
])
def test_env_flag(monkeypatch, raw, expected):
    from src.config import env_flag

    monkeypatch.setenv("NEUROSCAN_TEST_FLAG", raw)
    assert env_flag("NEUROSCAN_TEST_FLAG", True) is expected


@pytest.mark.parametrize("raw,expected", [
    ("", 10.0), ("2.5", 2.5), (" 3 ", 3.0),
    ("4MB", 10.0), ("0", 10.0), ("-1", 10.0), ("nan", 10.0), ("inf", 10.0), ("1e9", 10.0),
])
def test_env_number(monkeypatch, raw, expected):
    from src.config import env_number

    monkeypatch.setenv("NEUROSCAN_TEST_NUMBER", raw)
    assert env_number("NEUROSCAN_TEST_NUMBER", 10.0, minimum=0.1, maximum=1000) == expected


@pytest.mark.parametrize("raw,expected", [
    ("", 20), ("debug", 10), ("Warning", 30), ("WARN", 30), ("error", 40), ("15", 15), ("verbose", 20),
])
def test_env_log_level(monkeypatch, raw, expected):
    from src.config import env_log_level

    monkeypatch.setenv("NEUROSCAN_LOG_LEVEL", raw)
    assert env_log_level() == expected


def test_settings_ignore_invalid_values(monkeypatch):
    for name, raw in {"NEUROSCAN_MAX_UPLOAD_MB": "4MB", "NEUROSCAN_PORT": "abc",
                      "NEUROSCAN_TTA": "maybe", "NEUROSCAN_HOST": "  "}.items():
        monkeypatch.setenv(name, raw)
    s = server.Settings()
    assert (s.max_upload_mb, s.port, s.tta, s.host) == (4.0 if server.ON_VERCEL else 10.0, 8000, True, "127.0.0.1")
    monkeypatch.setenv("NEUROSCAN_TTA", "false")
    monkeypatch.setenv("NEUROSCAN_MAX_UPLOAD_MB", "0")
    s = server.Settings()
    assert s.tta is False and s.max_upload_mb > 0


@pytest.mark.parametrize("raw,expected", [
    ("", "auto"), (" ONNX ", "onnx"), ("onnxruntime", "onnx"), ("PyTorch", "torch"), ("bogus", "auto"),
])
def test_backend_setting_is_tolerant(monkeypatch, raw, expected):
    from src.inference import InferenceService

    monkeypatch.setenv("NEUROSCAN_BACKEND", raw)
    assert InferenceService().backend_preference == expected


def test_explicit_backend_argument_stays_strict():
    from src.inference import InferenceService

    with pytest.raises(ValueError):
        InferenceService(backend="bogus")


def test_unusable_backend_setting_falls_back_to_onnx(monkeypatch):
    """NEUROSCAN_BACKEND=torch on Vercel (no PyTorch) must still serve the ONNX models."""
    import src.inference as inference

    monkeypatch.setenv("NEUROSCAN_BACKEND", "torch")
    monkeypatch.setattr(inference, "_torch_available", lambda: False)
    svc = inference.InferenceService(benchmark_path=None).load()
    manifest = json.loads((MODELS_DIR / "onnx" / "manifest.json").read_text())
    assert svc.backend == "onnx" and set(svc.models) == set(manifest["models"])


def test_health_reports_degraded_without_models(tmp_path):
    app = server.create_app(server.Settings(models_dir=tmp_path, results_dir=tmp_path, serve_static=False))
    body = TestClient(app).get("/api/health").json()
    assert body["status"] == "degraded" and body["models_loaded"] == 0


def test_vercel_function_survives_invalid_settings():
    """Every optional setting mistyped at once, exactly as the function starts on Vercel."""
    script = BLOCKER + textwrap.dedent("""
        import json
        from fastapi.testclient import TestClient
        import server
        c = TestClient(server.app)
        health = c.get("/api/health").json()
        status = c.get("/api/status").json()
        data = open("public/samples/pituitary/Te-pi_10.jpg", "rb").read()
        r = c.post("/api/predict", files={"file": ("x.jpg", data, "image/jpeg")})
        s = server.Settings()
        print(json.dumps({"health": health, "max_upload_mb": status["max_upload_mb"], "predict": r.status_code,
                          "prediction": r.json().get("consensus_prediction"), "tta": s.tta, "port": s.port}))
    """)
    bad = {"NEUROSCAN_LOG_LEVEL": "verbose", "NEUROSCAN_MAX_UPLOAD_MB": "4MB", "NEUROSCAN_PORT": "abc",
           "NEUROSCAN_TTA": "maybe", "NEUROSCAN_ORT_THREADS": "two", "NEUROSCAN_BACKEND": "ONNXRuntime "}
    env = {**os.environ, "VERCEL": "1", **bad}
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-3000:]
    manifest = json.loads((MODELS_DIR / "onnx" / "manifest.json").read_text())
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "health": {"status": "ok", "backend": "onnx", "models_loaded": len(manifest["models"])},
        "max_upload_mb": 4.0, "predict": 200, "prediction": "pituitary", "tta": True, "port": 8000,
    }
    for name in ("NEUROSCAN_LOG_LEVEL", "NEUROSCAN_MAX_UPLOAD_MB", "NEUROSCAN_ORT_THREADS"):
        assert f"Ignoring {name}=" in result.stderr, f"no warning logged for {name}"
