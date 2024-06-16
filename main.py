# 注意：运行需要修改的参数有：视频路径、模型路径、stub路径、输出视频路径
from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
import cv2

def main(video_name):
    video_frames = read_video(f'input_videos/{video_name}') # 读取视频

    tracker = Tracker('models/best_yolov5_100.pt')

    tracks = tracker.get_object_trackers(video_frames, read_from_stub=True,
                                                 stub_path=f'stubs/track_stubs_{video_name}.pkl') # 目标追踪器

    # # 保存队员的一个裁剪图片
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     # 裁剪图片
    #     cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     # 保存图片
    #     cv2.imwrite('output_images/player_'+str(track_id)+'.png', cropped_img)

    #     break


    # 队伍分配器
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    # 为每个队员分配队伍
    
    for frame_num, player_track in enumerate(tracks['players']):  # 遍历每一帧
        for player_id, track in player_track.items():  # 遍历每个队员
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team  # 为队员分配队伍
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]  # 为队员分配队伍颜色

    # 绘制追踪效果
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    save_video(output_video_frames, f'output_videos/output_{video_name}.avi') # 保存视频

    # 成功保存视频
    print("Video saved successfully")

if __name__ == '__main__':
    video_name = '08fd33_0.mp4'
    main(video_name)