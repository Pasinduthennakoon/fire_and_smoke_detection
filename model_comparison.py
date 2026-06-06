import time
import cv2
import pandas as pd
from ultralytics import YOLO

DATA_YAML = "data.yaml"
TEST_VIDEO = "./data/5.mp4"

BASELINE_MODEL_PATH = "./yolov8n.pt"
IMPROVED_MODEL_PATH = "./runs/detect/fire_smoke_model/weights/best.pt"

IMG_SIZE = 640

def measure_fps(model, video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    start = time.time()

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

        _ = model(frame, verbose=False)

        frame_count += 1

    end = time.time()
    cap.release()

    fps = frame_count / (end - start)
    return fps


def evaluate_model(model_path):
    model = YOLO(model_path)
    metrics = model.val(data=DATA_YAML, workers=0, verbose=False)

    return {
        "mAP@0.5": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


baseline_model = YOLO(BASELINE_MODEL_PATH)
improved_model = YOLO(IMPROVED_MODEL_PATH)

baseline_metrics = evaluate_model(BASELINE_MODEL_PATH)
improved_metrics = evaluate_model(IMPROVED_MODEL_PATH)

baseline_fps = measure_fps(baseline_model, TEST_VIDEO)
improved_fps = measure_fps(improved_model, TEST_VIDEO)

results = pd.DataFrame([
    {
        "Model": "Baseline",
        "mAP@0.5": baseline_metrics["mAP@0.5"],
        "Precision": baseline_metrics["precision"],
        "Recall": baseline_metrics["recall"],
        "FPS": baseline_fps
    },
    {
        "Model": "Improved",
        "mAP@0.5": improved_metrics["mAP@0.5"],
        "Precision": improved_metrics["precision"],
        "Recall": improved_metrics["recall"],
        "FPS": improved_fps
    }
])

print("\n=== COMPARISON TABLE ===\n")
print(results)

results.to_csv("outputs/model_comparison.csv", index=False)

print("\nSaved to outputs/model_comparison.csv")