# Fire & Smoke Threat Detection System

## Overview

This project is a real-time Fire & Smoke Threat Detection System built using YOLOv8 and OpenCV.

The system detects fire and smoke from video streams, calculates threat-related signals, generates a threat score, classifies risk levels, and exports both annotated videos and CSV logs for further analysis.

The project demonstrates end-to-end computer vision and threat assessment capabilities suitable for:

* Smart surveillance systems
* Industrial safety monitoring
* Forest fire monitoring
* Building fire early-warning systems
* AI threat scoring research projects

---

## Features

### Object Detection

* Fire detection
* Smoke detection
* Bounding box visualization
* Confidence score display

### Threat Signal Extraction

The system extracts four key signals from each frame:

#### 1. Flame Ratio

Measures how much of the frame is occupied by fire.

[
flame_ratio = fire_area / total_frame_area
]

#### 2. Smoke Density

Measures the percentage of the frame occupied by smoke.

[
smoke_density = smoke_area / total_frame_area
]

#### 3. Spread Rate

Measures fire growth between consecutive frames.

[
spread_rate = max(0, fire_area - prev_fire_area)
spread_rate = spread_rate / total_frame_area
]

#### 4. Proximity

Approximates how close the fire is to becoming dangerous.

[
proximity = min(1.0, flame_ratio * 2)
]

---

## Threat Score Calculation

The threat score is calculated using a weighted combination of signals.

[
ThreatScore =
40(FlameRatio)
+30(SmokeDensity)
+20(SpreadRate)
+10(Proximity)
]

Maximum score:

[
threat_score = min(100, threat_score)
]

---

## Risk Levels

| Threat Score | Risk Level |
| ------------ | ---------- |
| < 5          | LOW        |
| 5 - 15       | MEDIUM     |
| 15 - 25      | HIGH       |
| > 25         | CRITICAL   |

---

## Project Structure

```text
project/
│
├── data/
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   ├── valid
│   │   ├── images
│   │   └── labels
│   └── 5.mp4
│
├── train.py
├── test.py
├── data.yaml
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Pasinduthennakoon/fire_and_smoke_detection
cd fire_and_smoke_detection
```

Create Virtual environment
```bash
python -m venv venv
cd venv/Scripts
activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---


## Model Training

Execute:

```bash
python train.py
```

After training:

```text
runs/detect/fire_smoke_model/weights/best.pt
```

will contain the best model.

---

## Running Detection

Execute:

```bash
python threat_detection.py
```

The system will:

1. Load the trained model
2. Read the input video
3. Detect fire and smoke
4. Calculate threat signals
5. Generate threat scores
6. Display risk levels
7. Save annotated video
8. Save CSV logs

---

## Outputs

### Annotated Video

```text
outputs/detect/5_output.mp4
```

Contains:

* Fire bounding boxes
* Smoke bounding boxes
* Threat score
* Risk level

### CSV Logs

```text
outputs/detect/5_log.csv
```

Example:

| frame | flame_ratio | smoke_density | spread_rate | proximity | threat_score | risk |
| ----- | ----------- | ------------- | ----------- | --------- | ------------ | ---- |
| 1     | 0.021       | 0.010         | 0.003       | 0.042     | 1.56         | LOW  |
| 2     | 0.041       | 0.020         | 0.008       | 0.082     | 3.52         | LOW  |

---

## Applications

* Industrial fire monitoring
* Warehouse surveillance
* Smart city safety systems
* CCTV analytics
* Forest fire early detection
* Threat scoring research
* Emergency response systems

---

## Future Improvements

* Multi-camera support
* Real-time webcam monitoring
* Temporal threat prediction
* Fire spread forecasting
* SMS/Email alerts
* Dashboard visualization
* DeepSORT object tracking
* Edge deployment using NVIDIA Jetson
* Threat score optimization using machine learning

---

## Technologies Used

* Python
* YOLOv8
* OpenCV
* NumPy
* Pandas
* PyTorch
* Ultralytics


