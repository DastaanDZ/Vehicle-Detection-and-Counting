from IPython import display
display.clear_output()
import cv2
import pandas as pd

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image

# import subprocess

# command = "yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.2 source=data/test/images"

# # Run the command
# subprocess.run(command, shell=True)

# Load a pretrained YOLOv8n model
model = YOLO('A:\\vehicle detection and counting\\runs\\detect\\train\\weights\\best.pt')

# Run inference on 'bus.jpg' with arguments
# model.predict('A:\\vehicle detection and counting\\data\\train\\images\\axi23_jpeg.rf.a8cf8c24a40cabb7a3bef9cd6f025f22.jpg', imgsz=320, conf=0.2)
result = model.track('A:\\vehicle detection and counting\\data\\train\\images\\photo1.png')
a = result[0].boxes.data
px = pd.DataFrame(a.cpu().numpy()).astype("float")

image = cv2.imread('A:\\vehicle detection and counting\\data\\train\\images\\photo1.png')


for index,row in px.iterrows():
    # Given coordinates
    # print(index,row)
    x1, y1 = int(row[0]),int(row[1])  # Top-left corner
    x2, y2 = int(row[2]),int(row[3])  # Bottom-right corner
    print(x1,y1)
    print(x2,y2)
    print("============")

    # Define the color (in BGR format)
    color = (0, 255, 0)  # Green in BGR

    # Draw the rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

    # Display the image
    cv2.imshow('Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
