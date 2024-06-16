from ultralytics import YOLO
import supervision as sv # supervision库是目标跟踪库
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.path.join("../")  # 将上级目录添加到系统路径中
from utils import get_bbox_width, get_center_of_bbox, measure_distance, get_foot_position

class Tracker:
    """
    追踪器，用于检测视频帧中的目标，并追踪目标
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path) # 通过YOLO加载模型（训练后的pt文件）
        self.tracker = sv.ByteTrack()   # 初始化目标跟踪器
        self.previous_distances = []  # 新增变量，存储上一帧的距离信息

    def add_position_to_tracks(sekf,tracks):
        """将目标的位置添加到跟踪结果中"""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        """插值球的位置"""
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames): # 检测视频帧，能够输出检测到的目标的位置
        batch_size = 20 # 每次检测的帧数
        detections = [] # 用于存放检测结果   
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1) # 对每一帧进行目标检测， 置信度0.1
            detections += detections_batch
            # break # 测试用，仅检测一帧
        return detections
        
    def get_object_trackers(self, frames, read_from_stub=False, stub_path=None): # 获取目标跟踪器，通过检测好的帧来追踪目标

        # 从文件中读取跟踪结果
        # stub的作用是为了避免重复计算，如果已经计算过了，就直接从文件中读取
        if(read_from_stub == True and stub_path is not None and os.path.exists(stub_path)):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks

        #  初始化目标跟踪器
        tracks = {
            "players": [],
            "referees": [],
            "ball": [],
        }

        detections = self.detect_frames(frames) # 获取每一帧目标检测的结果

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names # 获取检测到的目标的类别
            cls_names_inv = {v:k for k,v in cls_names.items()} # 获取类别的反向映射，用于获取类别的名称

            detection_supervision = sv.Detections.from_ultralytics(detection) # 将ultralytics的检测结果转换为supervision的检测结果
            # print(f'frame_num: {frame_num}, detections_supervision: {detections_supervision}')

            for object_id, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_id] = cls_names_inv["player"] # 将门将的类别改为普通球员的类别

            # 跟踪目标
            detection_with_trackers = self.tracker.update_with_detections(detection_supervision) # 更新目标跟踪器
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # 将跟踪结果存储到tracks中
            for frame_detection in detection_with_trackers:
                bbox = frame_detection[0].tolist()  # 获取目标的位置
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:  # 如果是球员
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:  # 如果是裁判
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
            for frame_detection in detection_supervision:  # 获取球的位置
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

            # print(f'frame_num: {frame_num}, detection_with_trackers: {detection_with_trackers}')
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)  # 将跟踪结果存储到文件中

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        # 使用椭圆绘制目标的位置
        y2 = int(bbox[3])  # 获取bbox的底部坐标
        x_center, _ = get_center_of_bbox(bbox)  # 获取bbox的中心点
        width = get_bbox_width(bbox)  # 获取bbox的宽度

        # 画椭圆
        cv2.ellipse(
            frame,  # 图像
            center=(x_center,y2),  # 中心坐标
            axes=(int(width), int(0.35*width)),  # 长轴和短轴
            angle=0.0,  # 旋转角度
            startAngle=-45,  # 开始角度
            endAngle=235,  # 结束角度
            color = color,  # 颜色
            thickness=2,  # 线宽
            lineType=cv2.LINE_4  # 线型
        )

        rectangle_width = 40  # 矩形的宽度
        rectangle_height=20  # 矩形的高度
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            # 画矩形
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),  # 左上角坐标
                          (int(x2_rect),int(y2_rect)),  # 右下角坐标
                          color,
                          cv2.FILLED)  # 填充
            
            x1_text = x1_rect+12 # 文本的x坐标
            if track_id > 99: # 如果track_id大于99，就向左移动10个像素
                x1_text -=10
            
            # 画文本
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),  #  文本的位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                0.6,  # 字体大小
                (0,0,0),  # 颜色
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        # 使用三角形绘制目标的位置
        y= int(bbox[1])  # 获取bbox的顶部坐标
        x,_ = get_center_of_bbox(bbox)  # 获取bbox的中心点

        triangle_points = np.array([  # 三角形的三个顶点
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)  # 画三角形
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)  # 画三角形的边框

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """绘制所有追踪效果"""
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # 复制一份视频帧    

            player_dict = tracks["players"][frame_num]
            referee_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # 画出球员的位置
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 255, 0))  # 获取队员的颜色，如果不存在则返回默认值
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball',False):  # 如果队员持球
                    frame = self.draw_traingle(frame, player["bbox"],(0,0,255))

            # 画出裁判的位置
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # 画出球的位置
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # 画出控球者
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            # 计算并绘制裁判与球以及控球者与球之间的距离
            if ball_dict:
                frame = self.draw_distance(frame, player_dict, referee_dict, ball)

            output_video_frames.append(frame)
        
        return output_video_frames

    def draw_distance(self, frame, players, referees, ball):
        """画出两个目标之间的距离"""
        ball_center = get_center_of_bbox(ball['bbox']) if ball else None

        distances = []

        if ball_center:
            for ref_id, referee in referees.items():
                ref_center = get_center_of_bbox(referee['bbox'])
                distance = measure_distance(ball_center, ref_center)
                distances.append(f"Ref Distance: {distance:.2f}")
                cv2.putText(frame, f"Dist: {distance:.2f}", (int(ref_center[0]), int(ref_center[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            for player_id, player in players.items():
                if player.get('has_ball', False):
                    player_center = get_center_of_bbox(player['bbox'])
                    distance = measure_distance(ball_center, player_center)
                    distances.append(f"Control Player Distance: {distance:.2f}")
                    cv2.putText(frame, f"Dist: {distance:.2f}", (int(player_center[0]), int(player_center[1] - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if not distances and self.previous_distances:
            distances = self.previous_distances
        else:
            self.previous_distances = distances

        # 在右下角显示所有距离信息
        text_y = frame.shape[0] - 20
        for dist_text in distances:
            cv2.putText(frame, dist_text, (frame.shape[1] - 300, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            text_y -= 20

        return frame

    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """画出控球者"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # 获取控球队伍的比例
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        total_frames = team_1_num_frames + team_2_num_frames

        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1 = 0
            team_2 = 0

        text_x_position = 1400
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(text_x_position,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(text_x_position,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame
    
    def export_positions(self, tracks, output_path='positions.csv'):
        """导出控球球员、裁判和球的位置"""
        positions = []

        for frame_num, (player_tracks, referee_tracks, ball_tracks) in enumerate(zip(tracks['players'], tracks['referees'], tracks['ball'])):
            for player_id, player in player_tracks.items():
                if player.get('has_ball', False):
                    position = player['position']
                    positions.append([frame_num, player_id, 'player', position[0], position[1], player['team']])
            for referee_id, referee in referee_tracks.items():
                position = referee['position']
                positions.append([frame_num, referee_id, 'referee', position[0], position[1], -1])  # team -1 for referees
            if 1 in ball_tracks:
                ball = ball_tracks[1]
                position = get_center_of_bbox(ball['bbox'])
                positions.append([frame_num, 1, 'ball', position[0], position[1], -1])  # team -1 for ball

        df = pd.DataFrame(positions, columns=['Frame', 'ID', 'Type', 'X', 'Y', 'Team'])
        df.to_csv(output_path, index=False)
        print(f"Positions exported to {output_path}")
