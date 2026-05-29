"""
Face Demographics Pipeline — LitServe deployment server.

Two-stage pipeline:
  1. YOLOv8 (fine-tuned on WIDER FACE) detects faces and returns bboxes
  2. ResNet18 (multi-task, fine-tuned on UTKFace) predicts age + gender per face

Endpoint: POST /predict   (multipart/form-data with field "request" containing an image)
Returns:  { "detections": [ {bbox, face_conf, age, gender, gender_conf}, ... ] }
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from fastapi import UploadFile
from litserve import LitAPI, LitServer
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
from fastapi import Request

# ── Configuration ─────────────────────────────────────────────────────────────
YOLO_WEIGHTS  = Path("checkpoints/yolo_face_best.pt")
CLASSIFIER_WEIGHTS = Path("checkpoints/demographic_resnet18.pt")

YOLO_CONF_THRESH      = 0.25
IMG_SIZE              = 768
DEFAULT_CROP_PADDING  = 0.5  # 50% extra context around each face bbox

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Multi-task classifier (same architecture as training notebook) ────────────
class MultiTaskResNet18(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_feat = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)
        self.age_head    = nn.Linear(in_feat, 1)
        self.gender_head = nn.Linear(in_feat, 1)

    def forward(self, x):
        feat = self.dropout(self.backbone(x))
        age = self.age_head(feat).squeeze(-1)
        gender_logit = self.gender_head(feat).squeeze(-1)
        return age, gender_logit


# ── Utility: crop a face bbox with extra context margin ───────────────────────
def crop_with_padding(image_pil: Image.Image, bbox, padding: float) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = padding * w, padding * h
    W, H = image_pil.size
    x1c, y1c = max(0, x1 - px), max(0, y1 - py)
    x2c, y2c = min(W, x2 + px), min(H, y2 + py)
    return image_pil.crop((x1c, y1c, x2c, y2c))


# ── LitServe API ──────────────────────────────────────────────────────────────
class FaceDemographicsAPI(LitAPI):
    """Two-stage face analysis pipeline served via LitServe."""

    def setup(self, device):
        print(f"[setup] device = {device}")

        # Stage 1: YOLOv8 face detector
        print(f"[setup] loading YOLO weights from {YOLO_WEIGHTS}")
        if not YOLO_WEIGHTS.exists():
            raise FileNotFoundError(f"YOLO weights not found at {YOLO_WEIGHTS}")
        self.yolo = YOLO(str(YOLO_WEIGHTS))
        self.yolo.to(device)

        # Stage 2: ResNet18 multi-task classifier
        print(f"[setup] loading classifier weights from {CLASSIFIER_WEIGHTS}")
        if not CLASSIFIER_WEIGHTS.exists():
            raise FileNotFoundError(f"Classifier weights not found at {CLASSIFIER_WEIGHTS}")
        self.classifier = MultiTaskResNet18().to(device)
        state = torch.load(str(CLASSIFIER_WEIGHTS), map_location=device)
        self.classifier.load_state_dict(state)
        self.classifier.eval()

        # Preprocessing transform (same as eval_tf in the training notebook)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self.device = device
        print("[setup] both models loaded and ready")

    def decode_request(self, request: UploadFile) -> dict:
        image = Image.open(io.BytesIO(request.file.read())).convert("RGB")
        crop_padding = DEFAULT_CROP_PADDING
        try:
            headers = request.headers
            crop_padding = float(headers.get("crop-padding", DEFAULT_CROP_PADDING))
        except Exception:
            pass
        return {
            "image": image,
            "crop_padding": crop_padding,
        }

    def predict(self, inputs: dict) -> dict:
        """Run the full pipeline on a single image.

        Returns a dict with the list of detections so encode_response is trivial.
        """
        # ── Stage 1: face detection ──
        image = inputs["image"]
        crop_padding = inputs["crop_padding"]
        np_img = np.array(image)
        results = self.yolo.predict(
            np_img, conf=YOLO_CONF_THRESH, imgsz=IMG_SIZE,
            save=False, verbose=False,
        )[0]

        if results.boxes is None or len(results.boxes) == 0:
            return {"detections": [], "num_faces": 0}

        boxes = results.boxes.xyxy.cpu().numpy()
        face_confs = results.boxes.conf.cpu().numpy()

        # ── Stage 2: demographic classification (batched) ──
        crops = [crop_with_padding(image, b.tolist(), padding=crop_padding) for b in boxes]
        batch = torch.stack([self.preprocess(c) for c in crops]).to(self.device)

        with torch.no_grad():
            pred_ages, pred_g_logits = self.classifier(batch)

        ages = np.clip(pred_ages.cpu().numpy(), 0, 100)
        g_probs = torch.sigmoid(pred_g_logits).cpu().numpy()  # P(female)

        detections = []
        for i, b in enumerate(boxes):
            is_female = g_probs[i] > 0.5
            detections.append({
                "bbox":        [round(float(x), 2) for x in b.tolist()],
                "face_conf":   round(float(face_confs[i]), 4),
                "age":         round(float(ages[i]), 1),
                "gender":      "Female" if is_female else "Male",
                "gender_conf": round(float(g_probs[i] if is_female else 1 - g_probs[i]), 4),
            })

        return {"detections": detections, "num_faces": len(detections)}

    def encode_response(self, prediction: dict) -> dict:
        return prediction

if __name__ == "__main__":
    print("=" * 60)
    print("  Face Demographics Pipeline — LitServe")
    print("=" * 60)
    server = LitServer(FaceDemographicsAPI(), accelerator="auto")
    server.run(port=8000)
