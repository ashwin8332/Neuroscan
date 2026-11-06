"""Inference service shared by the FastAPI server, the Gradio app and the CLI.

Backends
--------
* **onnx** (default when ``models/onnx/manifest.json`` exists): ONNX Runtime,
  no PyTorch needed. This is what the Vercel deployment runs.
* **torch**: the PyTorch checkpoints in ``models/`` (local development).

Choose with ``InferenceService(backend=...)`` or ``NEUROSCAN_BACKEND``
(``auto`` | ``onnx`` | ``torch``). This module imports no PyTorch at load time.

Behaviour (unchanged across backends): only models with a real checkpoint are
used, each with its own preprocessing and calibration temperature, flip TTA,
a reliability-weighted ensemble with an explicit "inconclusive" state, and
Grad-CAM from the most reliable CNN explaining that model's own prediction.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from .config import CLASS_METADATA, CLASS_NAMES, MEDICAL_DISCLAIMER, MODEL_SPECS, MODELS_DIR, ONNX_DIR, RESULTS_DIR
from .ensemble import assess_uncertainty, reliability_weight, weighted_average
from .preprocessing import assess_image_quality, preprocess_image
from .runtime import (  # noqa: F401  (re-exported helpers)
    ImageInput,
    ServingModel,
    colorize,
    heatmap_rgba,
    load_image,
    load_onnx_models,
    onnx_runtime_available,
    overlay,
    to_data_uri,
)

__all__ = [
    "InferenceService", "ImageInput", "ServingModel", "colorize", "heatmap_rgba", "load_image",
    "overlay", "to_data_uri", "to_pil",
]

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "onnx", "torch")
_LEGACY_TORCH_API = {"compute_gradcam", "prepare_image", "predict_single_image",
                     "generate_gradcam_overlay", "get_available_trained_models", "TorchModel"}


def __getattr__(name: str):
    """Lazily expose the old PyTorch-based helpers without importing torch eagerly."""
    if name in _LEGACY_TORCH_API:
        from . import torch_backend

        return getattr(torch_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def to_pil(image_input: ImageInput) -> Image.Image:  # backwards-compatible name
    return load_image(image_input)


def _torch_available() -> bool:
    # find_spec avoids importing PyTorch (seconds) just to check that it exists.
    return importlib.util.find_spec("torch") is not None


class InferenceService:
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        device: Any = None,
        benchmark_path: Optional[Path] = RESULTS_DIR / "model_comparison_metrics.json",
        tta: bool = True,
        backend: Optional[str] = None,
        onnx_dir: Optional[Path] = None,
    ):
        self.models_dir = Path(models_dir)
        self.onnx_dir = Path(onnx_dir) if onnx_dir else (self.models_dir / "onnx" if models_dir != MODELS_DIR else ONNX_DIR)
        self.device = device
        self.benchmark_path = benchmark_path
        self.tta = tta
        self.backend_preference = (backend or os.environ.get("NEUROSCAN_BACKEND", "auto")).lower()
        if self.backend_preference not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend_preference!r}")
        self.backend = "none"
        self.models: Dict[str, ServingModel] = {}
        self.unavailable: Dict[str, str] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- loading
    def _benchmark(self) -> Dict[str, Any]:
        if self.benchmark_path and Path(self.benchmark_path).exists():
            try:
                with open(self.benchmark_path) as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except (OSError, json.JSONDecodeError):
                logger.warning("Could not read %s", self.benchmark_path)
        return {}

    def _select_backend(self) -> str:
        has_onnx = (self.onnx_dir / "manifest.json").is_file()
        if self.backend_preference == "onnx":
            return "onnx"
        if self.backend_preference == "torch":
            return "torch"
        if has_onnx and onnx_runtime_available():
            return "onnx"
        if _torch_available():
            return "torch"
        return "onnx" if has_onnx else "none"

    def load(self) -> "InferenceService":
        """(Re)load all models. Builds a new registry, then swaps it in atomically
        so in-flight requests never see a half-loaded state. Never raises for
        missing or broken model files - they are reported as unavailable."""
        backend = self._select_backend()
        if backend == "onnx":
            if onnx_runtime_available():
                models, unavailable = load_onnx_models(self.onnx_dir)
            else:
                models, unavailable = {}, {k: "onnxruntime is not installed" for k in MODEL_SPECS}
        elif backend == "torch":
            try:
                from .torch_backend import load_torch_models
            except ImportError as exc:  # PyTorch missing or broken
                logger.error("PyTorch backend unavailable: %s", exc)
                models, unavailable = {}, {k: "PyTorch is not installed" for k in MODEL_SPECS}
            else:
                models, unavailable = load_torch_models(self.models_dir, self.device)
        else:
            models, unavailable = {}, {k: "no model backend available (install onnxruntime)" for k in MODEL_SPECS}

        benchmark = self._benchmark()
        for key, model in models.items():
            model.weight = reliability_weight(model.val_metrics, benchmark.get(key))
            logger.info("Loaded %s via %s (%s, T=%.2f, weight=%.2f)", model.display_name, backend,
                        model.arch, model.temperature, model.weight)
        with self._lock:
            self.models, self.unavailable, self.backend = models, unavailable, backend
        return self

    # ---------------------------------------------------------------- status
    def status(self) -> Dict[str, Any]:
        with self._lock:
            loaded, unavailable, backend = dict(self.models), dict(self.unavailable), self.backend
        total_weight = sum(m.weight for m in loaded.values()) or 1.0
        rows = []
        for key, spec in MODEL_SPECS.items():
            row: Dict[str, Any] = {
                "key": key,
                "display_name": spec["display_name"],
                "description": spec["description"],
                "checkpoint": spec["checkpoint"],
                "is_trained": key in loaded,
            }
            if key in loaded:
                m = loaded[key]
                row.update({"ensemble_weight": round(m.weight / total_weight, 4), **m.public_info()})
            else:
                row["unavailable_reason"] = unavailable.get(key, "not loaded")
            rows.append(row)
        device = "cpu" if backend == "onnx" else str(self.device or "cpu")
        return {
            "backend": backend,
            "device": device,
            "tta": self.tta,
            "classes": CLASS_NAMES,
            "class_metadata": CLASS_METADATA,
            "models": rows,
            "loaded_models": len(loaded),
            "total_models": len(MODEL_SPECS),
            "disclaimer": MEDICAL_DISCLAIMER,
        }

    # ---------------------------------------------------------------- predict
    @staticmethod
    def _gradcam_model(models: Dict[str, ServingModel]) -> Optional[ServingModel]:
        candidates = [m for m in models.values() if m.supports_gradcam]
        return max(candidates, key=lambda m: m.weight) if candidates else None

    def predict(self, image: ImageInput, with_gradcam: bool = True) -> Dict[str, Any]:
        start = time.perf_counter()
        pil = load_image(image)
        quality = assess_image_quality(pil)
        with self._lock:
            models = dict(self.models)
            unavailable = dict(self.unavailable)
            if not models:
                raise RuntimeError("No trained models are available on the server.")

            per_model, prob_rows, weights, votes = [], [], [], []
            own_prediction: Dict[str, int] = {}
            for key, m in models.items():
                probs = np.asarray(m.analyze(pil, self.tta), dtype=np.float64)
                if probs.shape != (len(CLASS_NAMES),) or not np.all(np.isfinite(probs)):
                    logger.error("%s returned invalid probabilities %s", key, probs)
                    unavailable[key] = "returned invalid output"
                    continue
                idx = int(np.argmax(probs))
                own_prediction[key] = idx
                meta = CLASS_METADATA[CLASS_NAMES[idx]]
                prob_rows.append(probs)
                weights.append(m.weight)
                votes.append(idx)
                per_model.append({
                    "model_key": key,
                    "display_name": m.display_name,
                    "predicted_class": CLASS_NAMES[idx],
                    "confidence_pct": round(float(probs[idx]) * 100.0, 2),
                    "confidences": {c: round(float(p) * 100.0, 2) for c, p in zip(CLASS_NAMES, probs)},
                    "color": meta["color"],
                    "severity": meta["severity"],
                    "description": meta["simple_summary"],
                    "weight": m.weight,
                    "temperature": round(m.temperature, 3),
                    "calibrated": abs(m.temperature - 1.0) > 1e-6,
                    "legacy_checkpoint": m.is_legacy,
                    "is_checkpoint_loaded": True,
                })
            if not per_model:
                raise RuntimeError("No model produced a valid prediction.")

            ensemble = weighted_average(prob_rows, weights)
            verdict = assess_uncertainty(ensemble, votes, weights)
            total_weight = sum(weights)
            for row in per_model:
                row["weight"] = round(row["weight"] / total_weight, 4)

            gradcam = None
            cam_model = self._gradcam_model({k: m for k, m in models.items() if k in own_prediction}) if with_gradcam else None
            if cam_model is not None:
                # Explain what this model itself predicted: a heatmap for a class
                # the model did not choose is hard to interpret.
                gradcam = self._gradcam_payload(cam_model, pil, own_prediction[cam_model.key])

        return self._build_response(pil, quality, per_model, ensemble, verdict, gradcam, unavailable,
                                    (time.perf_counter() - start) * 1000.0)

    def _gradcam_payload(self, m: ServingModel, pil: Image.Image, class_idx: int) -> Optional[Dict[str, Any]]:
        try:
            heat = m.explain(pil, class_idx)
        except Exception:  # explainability must never break a prediction
            logger.exception("Grad-CAM failed for %s", m.key)
            return None
        if heat is None:
            return None
        model_view = preprocess_image(pil, m.preprocess)
        return {
            "model_key": m.key,
            "display_name": m.display_name,
            "target_class": CLASS_NAMES[class_idx],
            "input_b64": to_data_uri(model_view),
            "heatmap_b64": to_data_uri(heatmap_rgba(heat), fmt="PNG"),
            "overlay_b64": to_data_uri(overlay(model_view, heat)),
        }

    def _build_response(self, pil, quality, per_model, ensemble, verdict, gradcam, unavailable, elapsed_ms):
        top = verdict["predicted_class"]
        meta = CLASS_METADATA[top]
        total = verdict["total_models"]
        agree = verdict["agreement_count"]
        warnings = list(quality["warnings"])

        if verdict["inconclusive"]:
            title = "Inconclusive — expert review needed"
            summary = (
                f"The AI models could not reach a confident result for this scan "
                f"({'; '.join(verdict['reasons'])}). The most likely pattern is "
                f"'{meta['short_label']}' at {verdict['probability'] * 100:.0f}%, but this should "
                "not be relied on. Please have the scan reviewed by a radiologist."
            )
        else:
            title = meta["display_name"]
            agreement = "All" if agree == total else f"{agree} of"
            summary = f"{agreement} {total} AI model{'s' if total != 1 else ''} matched this scan to the '{meta['short_label']}' pattern. "
            summary += f"{meta['simple_summary']} {meta['what_it_means']}"
        if warnings:
            summary += " Note: " + " ".join(warnings)

        ensemble_pct = {c: round(float(p) * 100.0, 2) for c, p in zip(CLASS_NAMES, ensemble)}
        guide = []
        for cls in CLASS_NAMES:
            m = CLASS_METADATA[cls]
            guide.append({
                "class_key": cls,
                "display_name": m["display_name"],
                "color": m["color"],
                "icon": m["icon"],
                "severity": m["severity"],
                "severity_level": m["severity_level"],
                "simple_summary": m["simple_summary"],
                "plain_details": m["plain_details"],
                "what_it_means": m["what_it_means"],
                "average_confidence_pct": ensemble_pct[cls],
                "is_detected": cls == top,
                "model_votes": sum(1 for r in per_model if r["predicted_class"] == cls),
                "total_models": total,
            })

        fallback_view = to_data_uri(pil.resize((224, 224)))
        return {
            "success": True,
            "backend": self.backend,
            "consensus_prediction": top,
            "consensus_display_name": title,
            "consensus_confidence": round(verdict["probability"] * 100.0, 2),
            "confidence_level": verdict["confidence_level"],
            "inconclusive": verdict["inconclusive"],
            "uncertainty": verdict,
            "agreement_count": agree,
            "total_models": total,
            "agreement_text": f"{agree} of {total} models agree",
            "plain_summary": summary,
            "severity": meta["severity"],
            "severity_level": "unknown" if verdict["inconclusive"] else meta["severity_level"],
            "ensemble_probabilities": ensemble_pct,
            "categories_guide": guide,
            "model_results": per_model,
            "unavailable_models": [
                {"model_key": k, "display_name": MODEL_SPECS[k]["display_name"], "reason": reason}
                for k, reason in unavailable.items()
            ],
            "gradcam": gradcam,
            "original_image_b64": gradcam["input_b64"] if gradcam else fallback_view,
            "gradcam_image_b64": gradcam["overlay_b64"] if gradcam else None,
            "image_quality": quality,
            "warnings": warnings,
            "tta": self.tta,
            "disclaimer": MEDICAL_DISCLAIMER,
            "inference_ms": round(elapsed_ms, 1),
        }
