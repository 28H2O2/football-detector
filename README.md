# football-detector
- 能够对足球比赛视频中的运动员进行识别，并且显示球员热力图、距离等指标
- 使用streamlit搭建前端，yolov5进行目标检测，opencv进行视频处理
- [引用仓库链接](https://github.com/abdullahtarek/football_analysis)

## 如何运行
- 将模型文件`best_yolov5_100.pt` 放入`/models`文件夹内
- 在终端运行命令`streamlit run streamlit_app.py` 