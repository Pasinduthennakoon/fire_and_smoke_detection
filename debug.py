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

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

# text log
# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------

os.makedirs("outputs/debug_logs", exist_ok=True)

# Log file path
log_path = f"outputs/debug_logs/{video_name}_log.txt"

# -----------------------------
# VARIABLES
# -----------------------------

prev_fire_area = 0
frame_id = 0

# -----------------------------
# OPEN TXT FILE
# -----------------------------

with open(log_path, "w") as f:
    f.write(f"Debug Log for {video_name}.mp4\n")
    f.write(f"FPS: {fps}\n\n\n")

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

            for box in r.boxes:
            
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)

                # FIRE
                if cls == 0:
                
                    fire_area += area

                    line = (
                        f"[FRAME {frame_id}] "
                        f"FIRE | "
                        f"Conf={conf:.2f} | "
                        f"Fire Area={area}"
                    )

                    print(line)
                    f.write(line + "\n")

                # SMOKE
                elif cls == 1:
                
                    smoke_area += area

                    line = (
                        f"[FRAME {frame_id}] "
                        f"SMOKE | "
                        f"Conf={conf:.2f} | "
                        f"Smoke Area={area}"
                    )

                    print(line)
                    f.write(line + "\n")

                else:
                    continue


        flame_ratio = fire_area / total_frame_area

        smoke_density = smoke_area / total_frame_area

        spread_rate = max(0, fire_area - prev_fire_area)
        spread_rate = spread_rate / total_frame_area

        # simplified proximity
        proximity = min(1.0, flame_ratio * 2)

        prev_fire_area = fire_area

        # # -----------------------------
        # # THREAT SCORE
        # # -----------------------------

        threat_score = (
            40 * flame_ratio +
            30 * smoke_density +
            20 * spread_rate +
            10 * proximity
        )

        threat_score = min(100, threat_score)

        if threat_score < 5:
            risk = "LOW"

        elif threat_score < 15:
            risk = "MEDIUM"

        elif threat_score < 25:
            risk = "HIGH"

        else:
            risk = "CRITICAL"


        signal_text = (
            "\n===== SIGNALS =====\n"
            f"Frame         : {frame_id}\n"
            f"Flame Ratio   : {flame_ratio:.4f}\n"
            f"Smoke Density : {smoke_density:.4f}\n"
            f"Spread Rate   : {spread_rate:.4f}\n"
            f"Proximity     : {proximity:.4f}\n"
            f"Threat Score  : {threat_score:.2f}\n"
            f"Risk Level    : {risk}\n"
            "===================\n\n"
        )

        # Print to console
        print(signal_text)

        # Write to txt file
        f.write(signal_text)

        # Next frame
        frame_id += 1
    
cap.release()

print(f"Log saved: {log_path}")