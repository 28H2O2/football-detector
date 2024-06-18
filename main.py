# 注意：运行需要修改的参数有：视频路径、模型路径、stub路径、输出视频路径
from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from visualize_heatmap import HeatmapVisualizer
import cv2
import numpy as np
import os

def main(video_name):
    video_frames = read_video(f'input_videos/{video_name}') # 读取视频

    # if video_name.endswith(".mp4"):
    #     video_name = video_name[:-4]

    tracker = Tracker('models/best_yolov5_100.pt')

    tracks = tracker.get_object_trackers(video_frames, read_from_stub=True,
                                                 stub_path=f'stubs/track_stubs_{video_name}.pkl') # 目标追踪器
 
    tracker.add_position_to_tracks(tracks)  # 将目标的位置添加到跟踪结果中
    
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"]) # 插值球的位置，使得每一帧都有球的位置

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

    # 球员分配器
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    default_team = 0  # 假设 0 表示没有控球队伍或默认控球队伍

    for frame_num, player_track in enumerate(tracks['players']):  # 遍历每一帧
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:  # 如果有队员持球
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:  # 如果没有控球队伍
                team_ball_control.append(default_team)

    team_ball_control = np.array(team_ball_control)  # 将控球队伍转换为numpy数组

    # 绘制追踪效果
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # 确保输出目录存在
    output_dir = f'output_videos/output_{video_name}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存视频
    save_video(output_video_frames, f'{output_dir}/output_{video_name}.mp4') # 保存视频

    # 导出位置数据
    tracker.export_positions(tracks, output_path=f'{output_dir}/positions.csv')

    # 生成热力图

    visualizer = HeatmapVisualizer()
    visualizer.visualize_heatmaps(input_path=f'{output_dir}/positions.csv', output_path = output_dir)

    # 成功保存视频
    print("Video saved successfully")

if __name__ == '__main__':
    video_name = 'e660601b_0.mp4'
    main(video_name)