"""
Minimal deployment demo: a Gradio app that loads a trained MultiCheXNet
checkpoint and lets a user upload a chest X-ray to get classification,
detection and segmentation results in one shot.

Run locally / in Colab:
    python app/gradio_app.py --checkpoint /path/to/mtl_best.pt

In a notebook cell you can also just `import` and call `build_app(...)`
then `demo.launch(share=True)`.
"""

import argparse
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.inference import load_model, preprocess
from src.utils.bbox_utils import decode_predictions


def build_app(checkpoint_path, device=config.DEVICE):
    model = load_model(checkpoint_path, device=device)

    def infer(image):
        # `image` comes in as an HxWx3 RGB uint8 numpy array from Gradio
        resized = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        x = preprocess(resized).to(device)

        import torch
        with torch.no_grad():
            out = model(x)

        vis = resized.copy()
        class_probs = {}
        if "class" in out:
            probs = torch.softmax(out["class"], dim=1)[0].cpu().numpy()
            class_probs = {name: float(p) for name, p in zip(config.CLASS_NAMES, probs)}

        if "det" in out:
            boxes = decode_predictions(out["det"][0], conf_threshold=0.3)
            for b in boxes:
                x1, y1, x2, y2 = map(int, b["box"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, f"{b['score']:.2f}", (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        mask_vis = None
        if "seg" in out:
            mask = out["seg"][0, 0].cpu().numpy()
            mask_vis = (mask * 255).astype(np.uint8)
            overlay = vis.copy()
            overlay[mask > 0.5] = [255, 0, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        return vis, mask_vis, class_probs

    demo = gr.Interface(
        fn=infer,
        inputs=gr.Image(label="Chest X-ray"),
        outputs=[
            gr.Image(label="Detections / segmentation overlay"),
            gr.Image(label="Raw segmentation mask"),
            gr.Label(label="Classification"),
        ],
        title="MultiCheXNet (PyTorch) - Chest X-ray Multi-Task Demo",
        description="Classifies (normal / pneumothorax / pneumonia), "
                     "detects pneumonia opacities, and segments pneumothorax "
                     "regions in a single forward pass.",
    )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_app(args.checkpoint)
    demo.launch(share=args.share)
