# 用于处理视频的工具函数
import cv2

def read_video(video_path):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    # 确保输出文件路径以 .mp4 结尾
    if not output_video_path.endswith('.mp4'):
        output_video_path += '.mp4'

    # 使用 'H264' 编解码器
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    
    # 获取帧的宽度和高度
    frame_height, frame_width = output_video_frames[0].shape[:2]
    
    # 创建 VideoWriter 对象
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (frame_width, frame_height))
    
    # 写入帧
    for frame in output_video_frames:
        out.write(frame)
    
    # 释放 VideoWriter 对象
    out.release()