import cv2
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("./runs/detect/fire_smoke_model/weights/best.pt")

test_image_dir = "./data/test/images"
output_dir     = "./outputs/test_results"
os.makedirs(output_dir, exist_ok=True)

image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

logs = []

for img_file in image_files:
    img_path = os.path.join(test_image_dir, img_file)

    frame = cv2.imread(img_path)
    frame = cv2.resize(frame, (640, 640))
    original = frame.copy()

    results = model(frame, verbose=False)

    total_area = 640 * 640
    fire_area  = 0
    smoke_area = 0
    confidences = []

    for r in results:
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            confidences.append(conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if cls == 0:
                fire_area += area
                label = f"Fire {conf:.2f}"
                color = (0, 0, 255)      # Red
            elif cls == 1:
                smoke_area += area
                label = f"Smoke {conf:.2f}"
                color = (128, 128, 128)  # Gray
            else:
                continue

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Threat Score ──────────────────────────────────────
    avg_conf      = sum(confidences) / len(confidences) if confidences else 0
    flame_ratio   = fire_area  / total_area
    smoke_density = smoke_area / total_area
    spread_rate   = flame_ratio * 0.5
    proximity     = 1.0 if flame_ratio > 0.1 else 0.5

    threat_score = (
        40 * avg_conf     +
        25 * flame_ratio  +
        20 * smoke_density +
        10 * spread_rate  +
        5  * proximity
    )
    threat_score = min(100, threat_score)

    # ── Risk Level ────────────────────────────────────────
    if threat_score < 15:
        risk  = "LOW"
        color = (0, 255, 0)        # Green
    elif threat_score < 25:
        risk  = "MEDIUM"
        color = (0, 165, 255)      # Orange
    elif threat_score < 35:
        risk  = "HIGH"
        color = (0, 0, 255)        # Red
    else:
        risk  = "CRITICAL"
        color = (0, 0, 139)        # Dark Red

    # ── Overlay - Threat Score & Risk ─────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame,
                f"Threat Score: {threat_score:.1f} / 100",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.putText(frame,
                f"Risk: {risk}",
                (15, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # ── Save Output Image ─────────────────────────────────
    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, frame)

    # ── Log ───────────────────────────────────────────────
    logs.append({
        "image":        img_file,
        "avg_confidence": round(avg_conf, 4),
        "flame_ratio":  round(flame_ratio, 4),
        "smoke_density": round(smoke_density, 4),
        "threat_score": round(threat_score, 2),
        "risk":         risk
    })

    print(f"✅ {img_file:<40} Score: {threat_score:.1f}  Risk: {risk}")

# ── Save CSV Log ──────────────────────────────────────────
df = pd.DataFrame(logs)
df.to_csv(os.path.join(output_dir, "test_results_log.csv"), index=False)

# ── Summary Chart ─────────────────────────────────────────
risk_counts = df["risk"].value_counts().reindex(
    ["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0
)

colors = ["#00b300", "#ff8c00", "#cc0000", "#660000"]
plt.figure(figsize=(8, 5))
bars = plt.bar(risk_counts.index, risk_counts.values, color=colors)
plt.title("Risk Level Distribution - Test Set")
plt.xlabel("Risk Level")
plt.ylabel("Number of Images")
plt.bar_label(bars, padding=3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "risk_distribution.png"), dpi=150)

print(f"\n{'='*50}")
print(f"Total images processed : {len(logs)}")
print(f"Output saved to        : {output_dir}")
print(f"CSV log                : {output_dir}/test_results_log.csv")
print(f"Chart                  : {output_dir}/risk_distribution.png")
print(f"{'='*50}")
print(risk_counts.to_string())