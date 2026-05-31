from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data = 'data.yaml',
    epochs = 50,
    imgsz = 640,
    batch = 16,
    workers=0,
    name = 'fire_smoke_model'
)