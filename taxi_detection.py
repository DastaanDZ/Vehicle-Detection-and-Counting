from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8n.yaml")

    # Use the model
    results = model.train(data="config.yaml", epochs=25 ,imgsz=800 ,plots=True)
