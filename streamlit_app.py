import streamlit as st
import os
from utils import save_video, read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from visualize_heatmap import HeatmapVisualizer
import numpy as np
import subprocess
import sys

def process_video(video_path):

    video_name = os.path.basename(video_path)
    if video_name.endswith(".mp4"):
        video_name = video_name[:-4]
    video_frames = read_video(video_path)  # 读取视频

    tracker = Tracker('models/best_yolov5_100.pt')

    tracks = tracker.get_object_trackers(video_frames, read_from_stub=True,
                                         stub_path=f'stubs/track_stubs_{video_name}.pkl')  # 目标追踪器

    tracker.add_position_to_tracks(tracks)  # 将目标的位置添加到跟踪结果中

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])  # 插值球的位置，使得每一帧都有球的位置

    # 队伍分配器
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
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
    output_video_path = f'{output_dir}/output_{video_name}.mp4'
    save_video(output_video_frames, output_video_path)  # 保存视频

    # 导出位置数据
    tracker.export_positions(tracks, output_path='positions.csv')

    # 生成热力图
    visualizer = HeatmapVisualizer()
    visualizer.visualize_heatmaps(input_path='positions.csv', output_path=output_dir)

    # 返回输出视频路径和热力图路径
    heatmap_paths = [
        os.path.join(output_dir, 'player_heatmap.png'),
        os.path.join(output_dir, 'referee_heatmap.png'),
        os.path.join(output_dir, 'ball_heatmap.png'),
        os.path.join(output_dir, 'teams_heatmap.png')
    ]
    
    return output_video_path, heatmap_paths

def main():
    st.title("Football Detector")
    st.write("Upload a football match video to detect players, referees, and the ball, and generate heatmaps.")

    example_videos = {
        "Example 1": "input_videos/example1.mp4",
        "Example 2": "input_videos/example2.mp4",
        "Example 3": "input_videos/example3.mp4"
    }

    example_choice = st.selectbox("Select an Example Video", list(example_videos.keys()))
    if example_choice:
        video_path = example_videos[example_choice]
        st.video(video_path)

    uploaded_file = st.file_uploader("Or Upload Your Own Video", type=["mp4", "avi", "mov"])

    if st.button("Process Video"):
        if uploaded_file is not None:
            video_path = uploaded_file.name
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        elif example_choice:
            video_path = example_videos[example_choice]
        
        if video_path:
            output_video_path, heatmap_paths = process_video(video_path)
            
            st.write("### Output Video")
            st.video(output_video_path)

            st.write("### Player Heatmap")
            st.image(heatmap_paths[0])

            st.write("### Referee Heatmap")
            st.image(heatmap_paths[1])

            st.write("### Ball Heatmap")
            st.image(heatmap_paths[2])

            st.write("### Teams Heatmap")
            st.image(heatmap_paths[3])

if __name__ == "__main__":
    # os.system("pip install -r requirements.txt")
    main()
