# Face Detection and Demographic Estimation

## Overview

This project implements a two-stage computer vision pipeline for face detection and demographic estimation.

The first stage uses a YOLOv8s model fine-tuned on the WIDER FACE dataset to detect human faces in images. The second stage uses a multi-task ResNet18 model trained on the UTKFace dataset to estimate:

* Age
* Gender

The system is deployed as a client-server application using LitServe and FastAPI.

---

## Model Performance

### Stage 1 – Face Detection (YOLOv8s)

Dataset: WIDER FACE

Results on the test set:

| Metric   | Value |
| -------- | ----- |
| mAP@50   | 0.742 |
| F1 Score | 0.759 |

---

### Stage 2 – Demographic Estimation (ResNet18)

Dataset: UTKFace

Results on the test set:

| Metric            | Value      |
| ----------------- | ---------- |
| Gender Accuracy   | 89.84%     |
| Weighted F1 Score | 0.90       |
| Age MAE           | 5.60 years |

---

## Pipeline Architecture

### Stage 1: Face Detection

Input image → YOLOv8s → Face bounding boxes

### Stage 2: Demographic Estimation

Detected faces → Face cropping → ResNet18 → Age + Gender predictions

---

## Repository Structure

```text
.
├── checkpoints/
│   ├── yolo_face_best.pt
│   └── demographic_resnet18.pt
│
├── client.py
├── server.py
├── FinalProject.ipynb
│
├── classifier_metrics.json
├── yolo_metrics.json
├── training_results.json
│
├── README.md
├── requirements.txt
└── AI_USAGE_STATEMENT.md
```

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd Face-Detection-and-Demographic-Estimation
```

### Create Environment

```bash
python -m venv venv
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Server

Start the inference server:

```bash
python server.py
```

The server will start on:

```text
http://localhost:8000
```

---

## Running the Client

Send an image to the server:

```bash
python client.py -i image.jpg
```

Custom endpoint:

```bash
python client.py \
    -i image.jpg \
    --url http://localhost:8000/predict
```

Specify output path:

```bash
python client.py \
    -i image.jpg \
    --out result.jpg
```

Modify face crop padding:

```bash
python client.py \
    -i image.jpg \
    --crop-padding 0.75
```

---

## API Endpoint

### POST /predict

Input:

* Multipart form-data
* Field name: request

Returns:

```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "face_conf": 0.95,
      "age": 28.4,
      "gender": "Male",
      "gender_conf": 0.93
    }
  ],
  "num_faces": 1
}
```

---

## Training

The complete training and evaluation workflow is available in:

```text
FinalProject.ipynb
```

The notebook contains:

* Data preparation
* WIDER FACE preprocessing
* YOLOv8 fine-tuning
* UTKFace preprocessing
* Multi-task ResNet18 training
* Model evaluation
* Metrics generation

---

## Datasets

### WIDER FACE

Used for face detection training.

https://shuoyang1213.me/WIDERFACE/

### UTKFace

Used for demographic estimation.

https://susanqq.github.io/UTKFace/

---

## Emanuel Serna & Samuel Areiza

Artificial Intelligence - EAFIT

Face Detection and Demographic Estimation

2026
