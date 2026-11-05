"""NeuroScan AI web server (FastAPI).

    python server.py                      # http://127.0.0.1:8000
    NEUROSCAN_PORT=9000 python server.py

Configuration (environment variables):
    NEUROSCAN_HOST            bind address (default 127.0.0.1)
    NEUROSCAN_PORT            port (default 8000)
    NEUROSCAN_MAX_UPLOAD_MB   upload size limit (default 10)
    NEUROSCAN_CORS_ORIGINS    comma-separated allowed origins (default: none, same-origin only)
    NEUROSCAN_TTA             1/0 - flip test-time augmentation (default 1)
    NEUROSCAN_LOG_LEVEL       logging level (default INFO)
"""
from __future__ import annotations

import io
import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.config import CLASS_NAMES, FRONTEND_DIR, MODELS_DIR, RESULTS_DIR, SAMPLE_SCANS_DIR, get_device
from src.inference import InferenceService

logger = logging.getLogger("neuroscan.server")

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}
Image.MAX_IMAGE_PIXELS = 40_000_000  # ~6300x6300; larger files are rejected as decompression bombs


@dataclass
class Settings:
    host: str = os.environ.get("NEUROSCAN_HOST", "127.0.0.1")
    port: int = int(os.environ.get("NEUROSCAN_PORT", "8000"))
    max_upload_mb: float = float(os.environ.get("NEUROSCAN_MAX_UPLOAD_MB", "10"))
    cors_origins: List[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("NEUROSCAN_CORS_ORIGINS", "").split(",") if o.strip()
    ])
    tta: bool = os.environ.get("NEUROSCAN_TTA", "1") != "0"
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    samples_dir: Path = SAMPLE_SCANS_DIR
    frontend_dir: Path = FRONTEND_DIR

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


class APIError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def index_samples(samples_dir: Path) -> Dict[str, Path]:
    """Whitelist of sample scans: ``sample_id -> path``. Only these can be requested."""
    index: Dict[str, Path] = {}
    if not samples_dir.is_dir():
        return index
    for cls in CLASS_NAMES:
        folder = samples_dir / cls
        if folder.is_dir():
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stat().st_size > 500:
                    index[f"{cls}/{path.name}"] = path
    return index


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise APIError(400, "The uploaded file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except Image.DecompressionBombError:
        raise APIError(413, "Image resolution is too large.")
    except (UnidentifiedImageError, OSError, ValueError):
        raise APIError(415, "The file is not a readable image. Upload a JPEG or PNG export of an MRI slice.")
    if fmt not in ALLOWED_FORMATS:
        raise APIError(415, f"Unsupported image format '{fmt}'. Use JPEG or PNG.")
    if min(img.size) < 32:
        raise APIError(400, f"Image is too small ({img.size[0]}x{img.size[1]}px).")
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    return img.convert("RGB")


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise APIError(413, f"File is larger than the {limit // (1024 * 1024)} MB limit.")
    return data


def _is_loopback(request: Request) -> bool:
    try:
        return request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return request.client is not None and request.client.host in {"localhost", "testclient"}


def _load_json(path: Path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def create_app(settings: Optional[Settings] = None, service: Optional[InferenceService] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            app.state.service = InferenceService(
                models_dir=settings.models_dir, device=get_device(), tta=settings.tta,
                benchmark_path=settings.results_dir / "model_comparison_metrics.json",
            ).load()
        else:
            app.state.service = service
        app.state.samples = index_samples(settings.samples_dir)
        logger.info("Ready: %d model(s) loaded, %d sample scans", len(app.state.service.models), len(app.state.samples))
        yield

    app = FastAPI(title="NeuroScan AI", version="2.0.0", lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def guard_and_secure(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 64 * 1024:
            return JSONResponse(status_code=413, content={"error": f"Request exceeds the {settings.max_upload_mb:g} MB limit."})
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
            "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'",
        )
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    # ------------------------------------------------------------------ API
    @app.get("/api/health")
    def health(request: Request):
        return {"status": "ok", "models_loaded": len(request.app.state.service.models)}

    @app.get("/api/status")
    def status(request: Request):
        return {**request.app.state.service.status(), "max_upload_mb": settings.max_upload_mb}

    @app.post("/api/refresh")
    def refresh(request: Request):
        if not _is_loopback(request):
            raise APIError(403, "Model reload is only allowed from the server machine.")
        request.app.state.service.load()
        request.app.state.samples = index_samples(settings.samples_dir)
        return {"status": "refreshed", "loaded_models": list(request.app.state.service.models)}

    @app.get("/api/samples")
    def samples(request: Request):
        return {"samples": [
            {"id": sample_id, "class": sample_id.split("/")[0], "filename": path.name,
             "path": f"/samples/{sample_id}"}
            for sample_id, path in request.app.state.samples.items()
        ]}

    @app.get("/api/metrics")
    def metrics():
        data = _load_json(settings.results_dir / "model_comparison_metrics.json")
        histories = {}
        for path in sorted(settings.results_dir.glob("*_history.json")):
            hist = _load_json(path) or {}
            key = hist.get("model_name") or path.stem.replace("_history", "")
            histories[key] = {k: hist.get(k) for k in ("display_name", "train_loss", "val_loss", "train_acc", "val_acc", "best_epoch")}
        if not data:
            return {"status": "unavailable", "metrics": {}, "histories": histories,
                    "message": "No evaluation results yet. Run `python evaluate.py`."}
        return {"status": "available", "metrics": data, "histories": histories,
                "meta": _load_json(settings.results_dir / "evaluation_meta.json") or {}}

    @app.post("/api/predict")
    def predict(request: Request, file: Optional[UploadFile] = File(None), sample_id: Optional[str] = Form(None)):
        # Sync endpoint: FastAPI runs it in a worker thread, so CPU-bound
        # inference never blocks the event loop.
        if file is not None and file.filename:
            image = decode_image(_read_limited(file, settings.max_upload_bytes))
        elif sample_id:
            path = request.app.state.samples.get(sample_id)
            if path is None:
                raise APIError(404, "Unknown sample scan.")
            with Image.open(path) as img:
                image = img.convert("RGB")
        else:
            raise APIError(400, "Provide an MRI image file or a sample_id.")

        service: InferenceService = request.app.state.service
        if not service.models:
            raise APIError(503, "No trained model checkpoints are available. Train models or add checkpoints to models/.")
        return service.predict(image)

    # ------------------------------------------------------------- static
    if settings.samples_dir.is_dir():
        app.mount("/samples", StaticFiles(directory=str(settings.samples_dir)), name="samples")
    if settings.frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(settings.frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        index_file = settings.frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"error": "Frontend not found"}, status_code=404)

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("NEUROSCAN_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Settings()
    uvicorn.run("server:app", host=cfg.host, port=cfg.port, reload=False)
