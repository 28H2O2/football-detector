from sklearn.cluster import KMeans

class TeamAssigner:
    """队员分配器"""
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """
        用KMeans获取图片的颜色        
        """
        image_2d = image.reshape(-1, 3)  # 将图片转换为2D数组
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        获取队员的颜色        
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[:int(image.shape[0] / 2), :]  # 取图片的上半部分（队衣）
        
        """以下代码和在color_assignment.ipynb中的代码相同"""
        kmeans = self.get_clustering_model(top_half_image)  # 获取图片的颜色

        labels = kmeans.labels_  # 获取图片的标签

        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])  # 重塑图片

        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]  # 获取角落的颜色
        non_player_clusters = max(set(corner_clusters), key=corner_clusters.count)  # 获取非角落的颜色
        player_cluster = 1 - non_player_clusters  # 获取队员的颜色

        player_color = kmeans.cluster_centers_[player_cluster]  # 获取队员的颜色

        return player_color

    def assign_team_color(self, frame, player_detections):
        """检测两队队员的颜色"""
        player_colors = []
        for _, player_detection in player_detections.items():  # 遍历队员
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)  # 获取队员的颜色
            player_colors.append(player_color)  # 添加队员的颜色

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)  # 对两个队伍进行聚类
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors = {1: kmeans.cluster_centers_[0], 2: kmeans.cluster_centers_[1]}  # 获取两个队伍的颜色

    def get_player_team(self, frame, player_bbox, player_id):
        """获取队员的队伍"""
        if player_id in self.player_team_dict:  # 如果队员已经分配了队伍
            return self.player_team_dict[player_id]  # 返回队员的队伍

        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1 # 预测队伍

        self.player_team_dict[player_id] = team_id

        return team_id
