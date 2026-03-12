from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

print(model.names)
