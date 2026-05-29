"""
Face Demographics Pipeline — client.

Sends an image to the LitServe endpoint and renders each detected face with:
  - bbox (blue if Male, pink if Female)
  - age estimate (years)
  - face detection confidence + gender confidence

Usage:
  python client.py -i path/to/image.jpg
  python client.py -i path/to/image.jpg --url http://localhost:8000/predict
  python client.py -i path/to/image.jpg --out custom_output.jpg
"""

import argparse
import logging
import os

import requests
from PIL import Image, ImageDraw, ImageFont

SERVER_URL = "http://localhost:8000/predict"

# Colors for visualization (RGB tuples)
MALE_COLOR   = (30, 144, 255)   # dodgerblue
FEMALE_COLOR = (255, 20, 147)   # deeppink
TEXT_COLOR   = (255, 255, 255)


def get_font(size: int):
    """Try to load a TTF font, fall back to PIL's default if unavailable."""
    candidate_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """Draw bboxes + labels on the image using PIL only (no extra deps)."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = get_font(size=max(14, annotated.width // 50))

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        color = FEMALE_COLOR if d["gender"] == "Female" else MALE_COLOR

        # Bbox
        line_width = max(2, annotated.width // 400)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)

        # Label text
        label = f"{d['gender']}, {d['age']:.0f}y"
        sub   = f"f:{d['face_conf']:.2f} g:{d['gender_conf']:.2f}"

        # Compute label box
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        pad = 4

        label_y = max(0, y1 - th - 2 * pad)
        draw.rectangle(
            [(x1, label_y), (x1 + tw + 2 * pad, label_y + th + 2 * pad)],
            fill=color,
        )
        draw.text((x1 + pad, label_y + pad), label, fill=TEXT_COLOR, font=font)

        # Sub-label (smaller) inside the bbox at the bottom
        sub_font = get_font(size=max(10, annotated.width // 70))
        sub_bbox = draw.textbbox((0, 0), sub, font=sub_font)
        sw = sub_bbox[2] - sub_bbox[0]
        sh = sub_bbox[3] - sub_bbox[1]
        draw.rectangle(
            [(x1, y2 - sh - 2 * pad), (x1 + sw + 2 * pad, y2)],
            fill=color,
        )
        draw.text((x1 + pad, y2 - sh - pad), sub, fill=TEXT_COLOR, font=sub_font)

    return annotated


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Face demographics pipeline client (sends image to LitServe).",
    )
    parser.add_argument("-i", "--image", required=True, help="Path to the input image.")
    parser.add_argument("--url", default=SERVER_URL, help="LitServe endpoint URL.")
    parser.add_argument("--out", default=None,
                        help="Output path for annotated image. Default: <name>_annotated.jpg")
    parser.add_argument("--crop-padding", type=float, default=0.5, help="Extra padding around face bbox.")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        logging.error("File not found: %s", args.image)
        return

    # ── Send request to server ──
    with open(args.image, "rb") as f:
        try:
            response = requests.post(
                args.url,
                files={"request": f},
                headers={"crop-padding": str(args.crop_padding)},
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            logging.error("Could not connect to %s — is the server running?", args.url)
            return

    if response.status_code != 200:
        logging.error("Server error %s: %s", response.status_code, response.text)
        return

    payload = response.json()
    detections = payload["detections"]
    num_faces  = payload.get("num_faces", len(detections))
    logging.info("Received %d detection(s) from the pipeline", num_faces)

    if not detections:
        logging.warning("No faces detected in the image.")
        return

    # ── Print table to console ──
    print(f"\n{'─' * 70}")
    print(f"  {'#':<3} {'Gender':<8} {'Age':>5} {'FaceConf':>9} {'GenConf':>8}  Bbox")
    print(f"  {'-' * 66}")
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox"]
        print(f"  {i:<3} {d['gender']:<8} {d['age']:>4.0f}y "
              f"{d['face_conf']:>9.3f} {d['gender_conf']:>8.3f}  "
              f"({x1:.0f},{y1:.0f})→({x2:.0f},{y2:.0f})")
    print(f"{'─' * 70}\n")

    # Summary stats
    males   = sum(1 for d in detections if d["gender"] == "Male")
    females = sum(1 for d in detections if d["gender"] == "Female")
    avg_age = sum(d["age"] for d in detections) / len(detections)
    print(f"  Summary: {males} male, {females} female, average age {avg_age:.1f} years\n")

    # ── Render and save annotated image ──
    img = Image.open(args.image).convert("RGB")
    annotated = draw_detections(img, detections)

    if args.out:
        out_path = args.out
    else:
        base = os.path.splitext(os.path.basename(args.image))[0]
        out_dir = os.path.dirname(args.image) or "."
        out_path = os.path.join(out_dir, f"{base}_annotated.jpg")

    annotated.save(out_path, quality=95)
    logging.info("Annotated image saved to: %s", out_path)


if __name__ == "__main__":
    main()
