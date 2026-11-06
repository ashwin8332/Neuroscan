"""NeuroScan AI web server (FastAPI).

    python server.py                      # http://127.0.0.1:8000
    NEUROSCAN_PORT=9000 python server.py

On Vercel this module is the function entrypoint (see pyproject.toml
``[tool.vercel] entrypoint``); ``public/`` is served by Vercel's CDN and only
``/api/*`` reaches this app. Models run on ONNX Runtime (no PyTorch needed).

Configuration (environment variables):
    NEUROSCAN_HOST            bind address (default 127.0.0.1)
    NEUROSCAN_PORT            port (default 8000)
    NEUROSCAN_MAX_UPLOAD_MB   upload size limit (default 10, capped at 4 on Vercel)
    NEUROSCAN_CORS_ORIGINS    comma-separated allowed origins (default: none, same-origin only)
    NEUROSCAN_TTA             1/0 - flip test-time augmentation (default 1)
    NEUROSCAN_BACKEND         auto | onnx | torch (default auto: ONNX when exported)
    NEUROSCAN_LOG_LEVEL       logging level (default INFO)
"""
from __future__ import annotations

import io
import ipaddress
import json
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config import CLASS_NAMES, MODELS_DIR, ON_VERCEL, PUBLIC_DIR, RESULTS_DIR
from src.inference import InferenceService
from src.jsonutil import json_safe
from src.runtime import normalize_image

logger = logging.getLogger("neuroscan.server")
if ON_VERCEL and not logging.getLogger().handlers:  # surface INFO logs in the Vercel dashboard
    logging.basicConfig(level=os.environ.get("NEUROSCAN_LOG_LEVEL", "INFO"),
                        format="%(levelname)s %(name)s: %(message)s")

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}
Image.MAX_IMAGE_PIXELS = 40_000_000  # ~6300x6300; larger files are rejected as decompression bombs
VERCEL_BODY_LIMIT_MB = 4.0           # Vercel rejects request bodies above 4.5 MB before the app runs
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; "
        "base-uri 'self'; form-action 'self'"
    ),
}


def _default_upload_mb() -> float:
    configured = float(os.environ.get("NEUROSCAN_MAX_UPLOAD_MB", "10"))
    return min(configured, VERCEL_BODY_LIMIT_MB) if ON_VERCEL else configured


@dataclass
class Settings:
    host: str = field(default_factory=lambda: os.environ.get("NEUROSCAN_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("NEUROSCAN_PORT", "8000")))
    max_upload_mb: float = field(default_factory=_default_upload_mb)
    cors_origins: List[str] = field(default_factory=lambda: [
        o.strip() for o in os.environ.get("NEUROSCAN_CORS_ORIGINS", "").split(",") if o.strip()
    ])
    tta: bool = field(default_factory=lambda: os.environ.get("NEUROSCAN_TTA", "1") != "0")
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    public_dir: Path = PUBLIC_DIR
    serve_static: bool = not ON_VERCEL   # on Vercel the CDN serves public/

    @property
    def samples_dir(self) -> Path:
        return self.public_dir / "samples"

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
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.is_file() and path.stat().st_size > 500:
                    index[f"{cls}/{path.name}"] = path
    return index


def decode_image(data: bytes) -> Image.Image:
    """Validate an upload by content (never by file name or Content-Type)."""
    if not data:
        raise APIError(400, "The uploaded file is empty.")
    try:
        with warnings.catch_warnings():
            # Pillow only *warns* between 1x and 2x MAX_IMAGE_PIXELS; treat it as fatal.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            fmt = img.format
            if fmt not in ALLOWED_FORMATS:
                raise APIError(415, f"Unsupported image format '{fmt}'. Use JPEG or PNG.")
            img.load()
    except APIError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise APIError(413, "Image resolution is too large.")
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
        raise APIError(415, "The file is not a readable image. Upload a JPEG or PNG export of an MRI slice.")
    if min(img.size) < 32:
        raise APIError(400, f"Image is too small ({img.size[0]}x{img.size[1]}px).")
    try:
        return normalize_image(img)  # first frame, EXIF rotation, 16-bit / alpha handling
    except (OSError, ValueError) as exc:
        raise APIError(415, f"The image could not be decoded ({exc.__class__.__name__}).")


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise APIError(413, f"File is larger than the {limit / (1024 * 1024):g} MB limit.")
    return data


def _is_loopback(request: Request) -> bool:
    try:
        return request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return request.client is not None and request.client.host in {"localhost", "testclient"}


def _load_json(path: Path) -> Any:
    try:
        with open(path) as f:
            return json_safe(json.load(f))
    except (OSError, ValueError):
        return None


def create_app(settings: Optional[Settings] = None, service: Optional[InferenceService] = None) -> FastAPI:
    settings = settings or Settings()
    service_lock = threading.Lock()

    def get_service(app: FastAPI) -> InferenceService:
        """Load models on first use. Works with or without ASGI lifespan events
        (serverless platforms do not always run them)."""
        current = getattr(app.state, "service", None)
        if current is not None:
            return current
        with service_lock:
            current = getattr(app.state, "service", None)
            if current is None:
                current = service or InferenceService(
                    models_dir=settings.models_dir, tta=settings.tta,
                    benchmark_path=settings.results_dir / "model_comparison_metrics.json",
                ).load()
                app.state.service = current
                logger.info("Ready: backend=%s, %d model(s) loaded", current.backend, len(current.models))
            return current

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            get_service(app)  # warm start when the platform runs lifespan events
        except Exception:     # never block startup; the first request retries
            logger.exception("Model warm-up failed")
        yield

    app = FastAPI(title="NeuroScan AI", version="2.1.0", lifespan=lifespan,
                  docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.get_service = lambda: get_service(app)

    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.middleware("http")
    async def guard_and_secure(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_upload_bytes + 64 * 1024:
            response = JSONResponse(status_code=413, content={"error": f"Request exceeds the {settings.max_upload_mb:g} MB limit."})
        else:
            response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(_request: Request, exc: APIError):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(_request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(status_code=exc.status_code, content={"error": message}, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(_request: Request, _exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"error": "Invalid request. Send a multipart form with a 'file' field."})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Internal server error."})

    # ------------------------------------------------------------------ API
    @app.api_route("/api/health", methods=["GET", "HEAD"])
    def health(request: Request):
        svc = get_service(request.app)
        return {"status": "ok", "backend": svc.backend, "models_loaded": len(svc.models)}

    @app.get("/api/status")
    def status(request: Request):
        return {**get_service(request.app).status(), "max_upload_mb": settings.max_upload_mb,
                "platform": "vercel" if ON_VERCEL else "server"}

    @app.post("/api/refresh")
    def refresh(request: Request):
        if not _is_loopback(request):
            raise APIError(403, "Model reload is only allowed from the server machine.")
        svc = get_service(request.app).load()
        return {"status": "refreshed", "backend": svc.backend, "loaded_models": list(svc.models)}

    @app.get("/api/samples")
    def samples():
        return {"samples": [
            {"id": sample_id, "class": sample_id.split("/")[0], "filename": path.name, "path": f"/samples/{sample_id}"}
            for sample_id, path in index_samples(settings.samples_dir).items()
        ]}

    @app.get("/api/metrics")
    def metrics():
        data = _load_json(settings.results_dir / "model_comparison_metrics.json")
        histories = {}
        for path in sorted(settings.results_dir.glob("*_history.json")):
            hist = _load_json(path)
            if not isinstance(hist, dict):
                continue
            key = hist.get("model_name") or path.stem.replace("_history", "")
            histories[key] = {k: hist.get(k) for k in ("display_name", "train_loss", "val_loss", "train_acc", "val_acc", "best_epoch")}
        if not isinstance(data, dict) or not data:
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
            path = index_samples(settings.samples_dir).get(sample_id)
            if path is None:
                raise APIError(404, "Unknown sample scan.")
            with Image.open(path) as img:
                image = normalize_image(img)
        else:
            raise APIError(400, "Provide an MRI image file or a sample_id.")

        svc = get_service(request.app)
        if not svc.models:
            raise APIError(503, "No trained models are available. Train models and export them with scripts/export_onnx.py.")
        return json_safe(svc.predict(image))

    # ------------------------------------------------------------- static
    # Local/self-hosted only. On Vercel, public/ is served by the CDN.
    if settings.serve_static and settings.public_dir.is_dir():
        if (settings.public_dir / "static").is_dir():
            app.mount("/static", StaticFiles(directory=str(settings.public_dir / "static")), name="static")
        if settings.samples_dir.is_dir():
            app.mount("/samples", StaticFiles(directory=str(settings.samples_dir)), name="samples")

        @app.get("/", include_in_schema=False)
        def index():
            index_file = settings.public_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            raise APIError(404, "Frontend not found.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=os.environ.get("NEUROSCAN_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Settings()
    uvicorn.run("server:app", host=cfg.host, port=cfg.port, reload=False)
