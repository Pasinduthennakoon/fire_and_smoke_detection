from ultralytics import YOLO
import os

model = YOLO("./runs/detect/fire_smoke_model/weights/best.pt")

metrics = model.val(
    data="data.yaml",
    workers=0
    )

map50 = metrics.box.map50
precision = metrics.box.mp   # mean precision (Ultralytics uses mp)
recall = metrics.box.mr      # mean 

os.makedirs("outputs/evaluation", exist_ok=True)

# write to txt file
with open("outputs/evaluation/evaluation_metrics.txt", "w") as f:
    f.write("=== MODEL EVALUATION ===\n")
    f.write(f"mAP@0.5     : {map50:.4f}\n")
    f.write(f"Precision   : {precision:.4f}\n")
    f.write(f"Recall      : {recall:.4f}\n")
    f.write("========================\n\n")
    f.write(str(metrics))

print("Saved to outputs/evaluation/evaluation_metrics.txt")