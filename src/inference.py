"""
End-to-end inference pipeline: load a trained MultiCheXNet checkpoint and run
it on a single chest X-ray image (jpg/png/dcm), producing:
    - predicted class (+ probability)
    - detected bounding boxes (if any)
    - a segmentation mask (if any)

This is the PyTorch equivalent of demo.py in the original repo.
"""

import numpy as np
import torch
import cv2

from . import config
from .models.mtl_model import MultiCheXNet
from .utils.bbox_utils import decode_predictions


def load_model(checkpoint_path, device=config.DEVICE, add_classifier=True,
               add_detector=True, add_segmenter=True):
    model = MultiCheXNet(pretrained_encoder=False, add_classifier=add_classifier,
                          add_detector=add_detector, add_segmenter=add_segmenter)
    model.load(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def read_image(path, dim=(config.IMG_SIZE, config.IMG_SIZE)):
    """Reads jpg/png/dcm and returns (resized_rgb_uint8, original_rgb_uint8)."""
    if str(path).lower().endswith(".dcm"):
        import pydicom
        arr = pydicom.dcmread(path).pixel_array
        arr = cv2.cvtColor(np.asarray(arr, dtype=np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    original = arr.copy()
    resized = cv2.resize(arr, dim)
    return resized, original


def preprocess(img_uint8):
    img = img_uint8.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


@torch.no_grad()
def predict(model, image_path, device=config.DEVICE, conf_threshold=0.3, seg_threshold=0.5):
    resized, original = read_image(image_path)
    x = preprocess(resized).to(device)

    out = model(x)
    result = {"image": resized}

    if "class" in out:
        probs = torch.softmax(out["class"], dim=1)[0].cpu().numpy()
        result["class_probs"] = {name: float(p) for name, p in zip(config.CLASS_NAMES, probs)}
        result["predicted_class"] = config.CLASS_NAMES[int(probs.argmax())]

    if "det" in out:
        boxes = decode_predictions(out["det"][0], conf_threshold=conf_threshold)
        result["boxes"] = boxes

    if "seg" in out:
        mask = out["seg"][0, 0].cpu().numpy()
        result["mask"] = mask
        result["mask_binary"] = (mask > seg_threshold).astype(np.uint8)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    res = predict(model, args.image)
    print("Predicted class:", res.get("predicted_class"))
    print("Class probabilities:", res.get("class_probs"))
    print("Boxes:", res.get("boxes"))
