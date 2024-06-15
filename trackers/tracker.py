from ultralytics import YOLO
import supervision as sv # supervision库是目标跟踪库

class Tracker:
    """
    追踪器，用于检测视频帧中的目标，并追踪目标
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path) # 通过YOLO加载模型（训练后的pt文件）
        self.tracker = sv.ByteTrack()   # 初始化目标跟踪器

    def detect_frames(self, frames): # 检测视频帧，能够输出检测到的目标的位置
        batch_size = 20 # 每次检测的帧数
        detections = [] # 用于存放检测结果   
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) # 对每一帧进行目标检测， 置信度0.1
            detections += detections_batch
            break # 测试用，仅检测一帧
        return detections
        
    def get_object_trackers(self, frames): # 获取目标跟踪器，通过检测好的帧来追踪目标

        detections = self.detect_frames(frames) # 获取每一帧目标检测的结果

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # 获取检测到的目标的类别
            cls_names_inv = {v:k for k,v in cls_names.items()} # 获取类别的反向映射，用于获取类别的名称

            detections_supervision = sv.Detections.from_ultralytics(detection) # 将ultralytics的检测结果转换为supervision的检测结果
            print(f'frame_num: {frame_num}, detections_supervision: {detections_supervision}')
