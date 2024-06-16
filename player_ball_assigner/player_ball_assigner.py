import sys 
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    """球员分配器，用于将球分配给球员"""
    def __init__(self):
        self.max_player_ball_distance = 70  # 球员和球之间的最大距离，单位是像素
    
    def assign_ball_to_player(self,players,ball_bbox):  
        """将球分配给球员"""
        ball_position = get_center_of_bbox(ball_bbox)  # 获取球的位置

        miniumum_distance = 99999  # 初始化最小距离
        assigned_player=-1  # 初始化分配的球员

        for player_id, player in players.items():
            """遍历所有球员，找到离球最近的球员"""
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0],player_bbox[-1]),ball_position)
            distance_right = measure_distance((player_bbox[2],player_bbox[-1]),ball_position)
            distance = min(distance_left,distance_right)

            if distance < self.max_player_ball_distance:  # 如果球员和球之间的距离小于最大距离
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player