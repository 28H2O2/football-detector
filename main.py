from utils import save_video, read_video
from trackers import Tracker

def main():
    video_frames = read_video('input_videos/08fd33_0.mp4') # 读取视频

    tracker = Tracker('models/best_yolov5_100.pt')

    object_tracker = tracker.get_object_trackers(video_frames) # 目标追踪器

    save_video(video_frames, 'output_videos/output.avi') # 保存视频

if __name__ == '__main__':
    main()