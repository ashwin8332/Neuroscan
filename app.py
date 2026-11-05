"""Alternative Gradio interface (the FastAPI app in server.py is the main UI).

    pip install -r requirements-gradio.txt
    python app.py                      # http://127.0.0.1:7861
"""
from __future__ import annotations

import base64
import io
import sys

import pandas as pd
from PIL import Image

try:
    import gradio as gr
except ImportError:  # gradio is an optional extra
    sys.exit("Gradio is not installed. Run: pip install -r requirements-gradio.txt")

from src.config import CLASS_METADATA, MEDICAL_DISCLAIMER, RESULTS_DIR, SAMPLE_SCANS_DIR
from src.inference import InferenceService

service = InferenceService().load()


def _decode(data_uri):
    if not data_uri:
        return None
    return Image.open(io.BytesIO(base64.b64decode(data_uri.split(",", 1)[1]))).convert("RGB")


def analyze(image: Image.Image):
    if image is None:
        return "Please upload an MRI slice.", {}, None
    if not service.models:
        return "No trained checkpoints are available in models/.", {}, None
    result = service.predict(image)
    lines = [
        f"## {result['consensus_display_name']}",
        f"**Confidence:** {result['confidence_level']} · {result['agreement_text']}",
        "",
        result["plain_summary"],
        "",
        "| Model | Prediction | Confidence | Weight |",
        "|---|---|---|---|",
    ]
    for row in result["model_results"]:
        lines.append(f"| {row['display_name']} | {CLASS_METADATA[row['predicted_class']]['short_label']} "
                     f"| {row['confidence_pct']:.1f}% | {row['weight'] * 100:.0f}% |")
    for missing in result["unavailable_models"]:
        lines.append(f"| {missing['display_name']} | not used | - | - |")
    lines += ["", f"_{MEDICAL_DISCLAIMER}_"]
    probabilities = {CLASS_METADATA[c]["short_label"]: p / 100 for c, p in result["ensemble_probabilities"].items()}
    overlay = _decode(result["gradcam"]["overlay_b64"]) if result["gradcam"] else None
    return "\n".join(lines), probabilities, overlay


def benchmark_table() -> pd.DataFrame:
    csv_file = RESULTS_DIR / "model_comparison_metrics.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return pd.DataFrame([{"Status": "No evaluation results yet - run python evaluate.py"}])


examples = []
for cls in CLASS_METADATA:
    scans = sorted((SAMPLE_SCANS_DIR / cls).glob("*.jpg"))
    if scans:
        examples.append(str(scans[0]))

with gr.Blocks(title="NeuroScan AI") as demo:
    gr.Markdown("# NeuroScan AI\nCalibrated multi-model brain MRI classification with Grad-CAM. "
                f"**Research use only.** {MEDICAL_DISCLAIMER}")
    with gr.Tab("Analyze"):
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(type="pil", label="MRI slice (JPEG/PNG)")
                button = gr.Button("Analyze scan", variant="primary")
                if examples:
                    gr.Examples(examples=examples, inputs=image_in)
            with gr.Column():
                report = gr.Markdown()
                probs = gr.Label(num_top_classes=4, label="Ensemble probabilities")
                cam = gr.Image(label="Grad-CAM (most reliable CNN)")
        button.click(analyze, inputs=image_in, outputs=[report, probs, cam])
    with gr.Tab("Benchmark"):
        table = gr.Dataframe(value=benchmark_table, interactive=False)
        gr.Button("Refresh").click(benchmark_table, outputs=table)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
