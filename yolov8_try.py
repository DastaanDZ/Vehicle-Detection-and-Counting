from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model(source="G:\\NIT STUDENT PROJECT\\Monday Clip\\192.168.1.100_ch8_20230807120002_20230807160003.asf", show=True, conf=0.4, save=True)
