import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_tactical_board(frame_size):
    """绘制底色为白色且有黑色边框的战术板"""
    # 创建一个白色背景的图像
    board = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
    
    # 绘制中线（黑色）
    cv2.line(board, (frame_size[1] // 2, 0), (frame_size[1] // 2, frame_size[0]), (0, 0, 0), 1)
    
    # 绘制中圈（黑色）
    cv2.circle(board, (frame_size[1] // 2, frame_size[0] // 2), 75, (0, 0, 0), 1)
    
    # 绘制边框（黑色）
    cv2.rectangle(board, (0, 0), (frame_size[1] - 1, frame_size[0] - 1), (0, 0, 0), 1)
    
    return board

# 示例
frame_size = (500, 800)
board = draw_tactical_board(frame_size)

# 使用 matplotlib 显示图像
plt.imshow(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
plt.title('Tactical Board with White Background and Black Borders')
plt.axis('off')  # 关闭坐标轴
plt.show()

