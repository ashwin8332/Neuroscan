import os
from pathlib import Path
from typing import Dict, List, Tuple
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.config import (
    CLASS_METADATA,
    CLASS_NAMES,
    MODELS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    TEST_DIR,
    get_device,
)
from src.inference import (
    generate_gradcam_overlay,
    get_available_trained_models,
    predict_single_image,
)
from src.models import MODEL_REGISTRY, build_model, load_checkpoint

device = get_device()
print(f"Starting Gradio App with device: {device}")

# Pre-load available models into memory
LOADED_MODELS = get_available_trained_models(device)
if not LOADED_MODELS:
    print("Notice: No saved checkpoints found in models/. Initializing untrained architectures for demo.")
    for key, spec in MODEL_REGISTRY.items():
        m = build_model(key, num_classes=len(CLASS_NAMES), pretrained=True)
        m.to(device).eval()
        LOADED_MODELS[spec["display_name"]] = m


def analyze_mri_image(image: Image.Image, show_gradcam: bool):
    """
    Run multi-model prediction and optional Grad-CAM visualizer.
    """
    if image is None:
        return "Please upload an MRI image.", {}, None

    predictions_markdown = "### Multi-Model Diagnostic Predictions\n\n"
    combined_confidences = {}

    gradcam_img = None

    for model_name, model in LOADED_MODELS.items():
        res = predict_single_image(model, image, device)
        pred_class = res["predicted_class"]
        conf = res["confidence"] * 100.0
        meta = CLASS_METADATA.get(pred_class, {})
        color = meta.get("color", "#333333")
        desc = meta.get("description", "")

        predictions_markdown += (
            f"<div style='border-left: 5px solid {color}; padding: 8px 12px; margin-bottom: 12px; background: #f8f9fa; border-radius: 4px;'>"
            f"<strong style='font-size: 1.1em;'>{model_name}</strong><br>"
            f"Prediction: <span style='color: {color}; font-weight: bold; font-size: 1.15em;'>{pred_class.upper()}</span> "
            f"({conf:.2f}% Confidence)<br>"
            f"<small style='color: #666;'>{desc}</small>"
            f"</div>"
        )

        for cls_name, prob in res["confidences"].items():
            key = f"{model_name[:12]}: {cls_name}"
            combined_confidences[key] = float(prob)

        # Generate Grad-CAM for the first supported CNN
        if show_gradcam and gradcam_img is None:
            cam_result = generate_gradcam_overlay(model, image, device)
            if cam_result is not None:
                gradcam_img = cam_result

    return predictions_markdown, combined_confidences, gradcam_img


def load_benchmark_table():
    """Load evaluation CSV if present."""
    csv_file = RESULTS_DIR / "model_comparison_metrics.csv"
    if csv_file.exists():
        return pd.read_csv(csv_file)
    return pd.DataFrame(
        [
            {
                "Model": spec["display_name"],
                "Status": "Checkpoint present" if (MODELS_DIR / spec["checkpoint"]).exists() else "Needs training",
            }
            for spec in MODEL_REGISTRY.values()
        ]
    )


# Gather sample images for test examples in UI
EXAMPLE_IMAGES = []
for cls_name in CLASS_NAMES:
    cls_folder = TEST_DIR / cls_name
    if cls_folder.exists():
        samples = list(cls_folder.glob("*.jpg"))
        if samples:
            EXAMPLE_IMAGES.append(str(samples[0]))


# Gradio UI definition
with gr.Blocks(title="Brain Tumor MRI Detection & Model Comparison Suite", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 Brain Tumor Detection & Multi-Model Comparative Benchmark
        ### Machine Learning & Deep Learning Comparative Study
        Upload an MRI scan to perform real-time classification across multiple deep learning architectures 
        (**Custom CNN**, **ResNet-18**, **MobileNetV3**, and **Vision Transformer**), inspect prediction confidence distributions, 
        and visualize **Grad-CAM tumor localization heatmaps**.
        """
    )

    with gr.Tabs():
        with gr.TabItem("🔍 Live Multi-Model Diagnostic"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="pil", label="Upload Brain MRI Scan")
                    gradcam_toggle = gr.Checkbox(value=True, label="Generate Grad-CAM Explainability Heatmap")
                    predict_btn = gr.Button("Analyze Scan Across All Models", variant="primary")

                    if EXAMPLE_IMAGES:
                        gr.Examples(
                            examples=EXAMPLE_IMAGES,
                            inputs=input_image,
                            label="Sample Dataset MRI Scans",
                        )

                with gr.Column(scale=1):
                    output_html = gr.HTML(label="Diagnostic Assessment")
                    gradcam_output = gr.Image(label="Grad-CAM Tumor Attention Localization")
                    conf_plot = gr.Label(num_top_classes=4, label="Class Probabilities")

            predict_btn.click(
                fn=analyze_mri_image,
                inputs=[input_image, gradcam_toggle],
                outputs=[output_html, conf_plot, gradcam_output],
            )

        with gr.TabItem("📊 Comparative Benchmark & Metrics"):
            gr.Markdown(
                """
                ### Model Evaluation Summary & Performance Metrics
                Comparison of test set accuracy, error rates, cross-entropy loss, macro F1-score, and inference latency.
                """
            )
            refresh_btn = gr.Button("Refresh Metrics")
            metrics_table = gr.Dataframe(value=load_benchmark_table, interactive=False)
            refresh_btn.click(fn=load_benchmark_table, outputs=metrics_table)

            gr.Markdown("### Publication Figures")
            with gr.Row():
                acc_err_plot = PLOTS_DIR / "accuracy_and_error_comparison.png"
                f1_auc_plot = PLOTS_DIR / "f1_and_auc_comparison.png"
                cm_plot = PLOTS_DIR / "confusion_matrices_grid.png"
                class_f1_plot = PLOTS_DIR / "per_class_f1_heatmap.png"

                if acc_err_plot.exists():
                    gr.Image(value=str(acc_err_plot), label="Accuracy vs Error Rate")
                if f1_auc_plot.exists():
                    gr.Image(value=str(f1_auc_plot), label="Macro F1 vs ROC-AUC")

            with gr.Row():
                if cm_plot.exists():
                    gr.Image(value=str(cm_plot), label="Confusion Matrices")
                if class_f1_plot.exists():
                    gr.Image(value=str(class_f1_plot), label="Per-Class F1 Breakdown")

        with gr.TabItem("ℹ️ Pathology & Dataset Explorer"):
            gr.Markdown(
                """
                ### Classification Classes & Pathology Information

                | Class | Category | Clinical Description | Typical MRI Appearance |
                |---|---|---|---|
                | **Glioma** | Malignant / Invasive | Arises from glial cells; can be low-grade or high-grade glioblastoma. | Infiltrative mass with irregular hyperintense margins on T2/FLAIR. |
                | **Meningioma** | Typically Benign | Arises from the meningeal coverings of the brain and spinal cord. | Well-circumscribed, extra-axial dural-based mass with vivid contrast enhancement. |
                | **Pituitary** | Sellar Tumor | Neoplasm of the pituitary gland near optic chiasm. | Sellar / suprasellar mass with sellar remodeling and optic compression risk. |
                | **No Tumor** | Healthy Normal | Brain MRI without detectable neoplasms or mass effect. | Normal ventricular symmetry and gray-white matter junction differentiation. |

                #### Dataset Split
                - **Training Split**: 5,600 MRI images (balanced ~1,400 per class)
                - **Testing Split**: 1,600 MRI images (balanced ~400 per class)
                """
            )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
