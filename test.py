import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os

# Load trained model
model = YOLO("./runs/detect/fire_smoke_model/weights/best.pt")

# Input video path
video_path = "./data/5.mp4"

# Extract filename without extension
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Video input
cap = cv2.VideoCapture(video_path)

os.makedirs("outputs/detect", exist_ok=True)

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    f'outputs/detect/{video_name}_output.mp4',
    fourcc,
    20.0,
    (640, 640)
)

# CSV log
logs = []

prev_fire_area = 0
frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 640))

    results = model(frame)

    total_frame_area = 640 * 640

    fire_area = 0
    smoke_area = 0

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            area = (x2 - x1) * (y2 - y1)

            label = ""

            # FIRE
            if cls == 0:
                fire_area += area
                label = f"Fire {conf:.2f}"
                color = (0, 0, 255)

            # SMOKE
            elif cls == 1:
                smoke_area += area
                label = f"Smoke {conf:.2f}"
                color = (128, 128, 128)

            else:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # -----------------------------
    # SIGNALS
    # -----------------------------

    flame_ratio = fire_area / total_frame_area

    smoke_density = smoke_area / total_frame_area

    spread_rate = max(0, fire_area - prev_fire_area)
    spread_rate = spread_rate / total_frame_area

    # simplified proximity
    proximity = min(1.0, flame_ratio * 2)

    prev_fire_area = fire_area

    # -----------------------------
    # THREAT SCORE
    # -----------------------------

    threat_score = (
        40 * flame_ratio +
        30 * smoke_density +
        20 * spread_rate +
        10 * proximity
    )

    threat_score = min(100, threat_score)

    # -----------------------------
    # RISK LEVEL
    # -----------------------------

    if threat_score < 5:
        risk = "LOW"

    elif threat_score < 15:
        risk = "MEDIUM"

    elif threat_score < 25:
        risk = "HIGH"

    else:
        risk = "CRITICAL"

    # -----------------------------
    # DISPLAY
    # -----------------------------

    cv2.putText(
        frame,
        f"Threat Score: {threat_score:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Risk: {risk}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Save logs
    logs.append({
        "frame": frame_id,
        "flame_ratio": flame_ratio,
        "smoke_density": smoke_density,
        "spread_rate": spread_rate,
        "proximity": proximity,
        "threat_score": threat_score,
        "risk": risk
    })

    frame_id += 1

    out.write(frame)

    cv2.imshow("Threat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save CSV
pd.DataFrame(logs).to_csv(
    f"outputs/detect/{video_name}_log.csv",
    index=False
)

print("Processing completed")