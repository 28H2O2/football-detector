from ultralytics import YOLO 

model = YOLO('models/best_yolov5_50.pt') # 我微调训练之后的模型
# model = YOLO('yolov8l') # 用原来的yolo模型

results = model.predict('input_videos/08fd33_0.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)