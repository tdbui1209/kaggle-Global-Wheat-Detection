import os
import comet_ml
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(
        data="data.yaml",
        project="global-wheat-detection",
        batch=2,
        save_period=1,
        epochs=100,
        imgsz=320,
        cos_lr=True,
        name="yolov8n 320 aug",
        optimizer="SGD",
        patience=10
    )