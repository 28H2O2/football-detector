from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

class HeatmapVisualizer:
    def __init__(self):
        pass

    def visualize_heatmaps(self, input_path='positions.csv', output_path='output_videos'):
        """生成控球球员、裁判、球的位置的热力图"""
        df = pd.read_csv(input_path)

        # 绘制控球球员位置的热力图
        player_positions = df[df['Type'] == 'player']
        plt.figure(figsize=(12, 8))
        sns.kdeplot(x=player_positions['X'], y=player_positions['Y'], cmap='Reds', fill=True, bw_adjust=0.5)
        plt.title('Heatmap of Control Player Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(os.path.join(output_path, 'player_heatmap.png'))
        # plt.show()

        # 绘制裁判位置的热力图
        referee_positions = df[df['Type'] == 'referee']
        plt.figure(figsize=(12, 8))
        sns.kdeplot(x=referee_positions['X'], y=referee_positions['Y'], cmap='Blues', fill=True, bw_adjust=0.5)
        plt.title('Heatmap of Referee Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(os.path.join(output_path, 'referee_heatmap.png'))
        # plt.show()

        # 绘制球位置的热力图
        ball_positions = df[df['Type'] == 'ball']
        plt.figure(figsize=(12, 8))
        sns.kdeplot(x=ball_positions['X'], y=ball_positions['Y'], cmap='Greens', fill=True, bw_adjust=0.5)
        plt.title('Heatmap of Ball Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.savefig(os.path.join(output_path, 'ball_heatmap.png'))
        # plt.show()

        # 绘制两支队伍所有球员的热力图
        team1_positions = player_positions[player_positions['Team'] == 1]
        team2_positions = player_positions[player_positions['Team'] == 2]
        plt.figure(figsize=(12, 8))
        sns.kdeplot(x=team1_positions['X'], y=team1_positions['Y'], cmap='Reds', fill=True, bw_adjust=0.5, label='Team 1')
        sns.kdeplot(x=team2_positions['X'], y=team2_positions['Y'], cmap='Blues', fill=True, bw_adjust=0.5, label='Team 2')
        plt.title('Heatmap of Team Player Positions')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_path, 'teams_heatmap.png'))
        # plt.show()
