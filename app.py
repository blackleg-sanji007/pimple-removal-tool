"""Acne removal demo: YOLOv8s masks with DeepFillv2 and OpenCV inpainting."""

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import gdown
import gradio as gr
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.environ.get("ACNE_YOLO_MODEL_PATH", ROOT / "models" / "acne04_60_best.pt")
).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH

DEEPFILL_DIR = ROOT / "deepfillv2_colab"
DEEPFILL_WEIGHTS = DEEPFILL_DIR / "model" / "deepfillv2_WGAN.pth"
DEEPFILL_REPO = "https://github.com/vrindaprabhu/deepfillv2_colab.git"
DEEPFILL_FILE_ID = "1uMghKl883-9hDLhSiI8lRbHCzCmmRwV-"
IMAGE_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def detector():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH)).to(DEVICE)


@lru_cache(maxsize=1)
def ensure_deepfill():
    """Fetch the upstream inference code and its pretrained generator once."""
    inpaint_script = DEEPFILL_DIR / "inpaint.py"
    if not inpaint_script.is_file():
        if DEEPFILL_DIR.exists():
            raise RuntimeError(f"Incomplete DeepFill directory: {DEEPFILL_DIR}")
        subprocess.run(["git", "clone", DEEPFILL_REPO, str(DEEPFILL_DIR)],
                       check=True, cwd=ROOT)

    # The upstream script hardcodes eight workers for a one-image dataset.
    # Its config already sets NUM_WORKERS=0, which avoids Windows spawn delays.
    source = inpaint_script.read_text(encoding="utf-8")
    if "num_workers=8," in source:
        inpaint_script.write_text(
            source.replace("num_workers=8,", "num_workers=NUM_WORKERS,"),
            encoding="utf-8",
        )

    if not DEEPFILL_WEIGHTS.is_file():
        DEEPFILL_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(id=DEEPFILL_FILE_ID,
                                output=str(DEEPFILL_WEIGHTS), quiet=False)
        if not result or not DEEPFILL_WEIGHTS.is_file():
            raise RuntimeError("DeepFill checkpoint download failed")


def acne_mask(image_bgr, confidence, kernel_size, iterations):
    height, width = image_bgr.shape[:2]
    boxes = detector().predict(
        image_bgr, conf=float(confidence), device=DEVICE, verbose=False
    )[0].boxes.xyxy
    mask = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = (int(value) for value in box.tolist())
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    if len(boxes):
        size = max(3, int(kernel_size) | 1)
        kernel = np.ones((size, size), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=int(iterations))
    return mask, len(boxes)


def run_deepfill(image_bgr, mask):
    ensure_deepfill()
    image_path = DEEPFILL_DIR / "input" / "input_img.png"
    mask_path = DEEPFILL_DIR / "input" / "mask.png"
    output_path = DEEPFILL_DIR / "output" / "inpainted_img.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(image_path), image_bgr):
        raise RuntimeError("Could not save the DeepFill input image")
    if not cv2.imwrite(str(mask_path), mask):
        raise RuntimeError("Could not save the DeepFill mask")
    output_path.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, "inpaint.py"], cwd=DEEPFILL_DIR,
                       check=True)
        result_bgr = cv2.imread(str(output_path))
        if result_bgr is None:
            raise RuntimeError("DeepFill produced no output image")
        return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    finally:
        image_path.unlink(missing_ok=True)
        mask_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


def compare_methods(image_rgb, confidence, kernel_size, iterations):
    if image_rgb is None:
        raise gr.Error("Upload a face image first.")
    image_rgb = np.asarray(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    mask, detections = acne_mask(image_bgr, confidence, kernel_size, iterations)
    if detections == 0:
        gr.Info("No acne regions detected; showing the original image.")
        return image_rgb, image_rgb, (image_rgb, image_rgb), (image_rgb, image_rgb)

    resized_original = cv2.resize(image_rgb, IMAGE_SIZE)
    resized_bgr = cv2.resize(image_bgr, IMAGE_SIZE)
    resized_mask = cv2.resize(mask, IMAGE_SIZE)
    opencv_bgr = cv2.inpaint(resized_bgr, resized_mask, 3, cv2.INPAINT_TELEA)
    opencv_rgb = cv2.cvtColor(opencv_bgr, cv2.COLOR_BGR2RGB)
    deepfill_rgb = run_deepfill(image_bgr, mask)
    return (
        deepfill_rgb,
        opencv_rgb,
        (resized_original, deepfill_rgb),
        (opencv_rgb, deepfill_rgb),
    )


def create_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Pimple Removal Tool (YOLOv8s + DeepFillv2)")
        gr.Markdown("Upload a face image and click **Remove Pimples**.")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="numpy", image_mode="RGB", label="Input Face Image"
                )
                confidence = gr.Slider(
                    0.05, 0.5, value=0.1, step=0.01, label="Confidence Threshold"
                )
                kernel_size = gr.Slider(
                    3, 15, value=9, step=2, label="Mask Dilation Kernel Size (px)"
                )
                iterations = gr.Slider(
                    1, 5, value=3, step=1, label="Mask Dilation Iterations"
                )
                submit = gr.Button("Remove Pimples", variant="primary")
            with gr.Column():
                deepfill_output = gr.Image(type="numpy", label="DeepFillv2 Output")
                opencv_output = gr.Image(type="numpy", label="OpenCV Telea Output")
                before_after = gr.ImageSlider(label="Before / After DeepFillv2")
                methods = gr.ImageSlider(label="OpenCV / DeepFillv2")

        submit.click(
            fn=compare_methods,
            inputs=[image_input, confidence, kernel_size, iterations],
            outputs=[deepfill_output, opencv_output, before_after, methods],
            concurrency_limit=1,
        )
    return demo


if __name__ == "__main__":
    if not MODEL_PATH.is_file():
        raise SystemExit(f"YOLO checkpoint not found: {MODEL_PATH}")
    ensure_deepfill()
    share = os.environ.get("GRADIO_SHARE", "0").lower() in {"1", "true", "yes"}
    create_gradio_app().launch(
        server_name="127.0.0.1", share=share, theme=gr.themes.Soft()
    )
