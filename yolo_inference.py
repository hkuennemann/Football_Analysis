"""File to get to know ultralytics and the YOLO model"""

from ultralytics import YOLO

# model 1
# model = YOLO('yolov8x.pt')

# model 2
model = YOLO('runs/detect/train/weights/best.pt')

results = model.predict('input_videos/08fd33_4.mp4', save = True)
print(results[0])
print('===========================')
for box in results[0].boxes:
    print(box)
