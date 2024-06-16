# bbox_utils.py
def get_center_of_bbox(bbox):
    # 获取bbox的中心点
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    # 获取bbox的宽度
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    # 计算两点之间的二维距离
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    # 计算两点之间的L1距离
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    # 获取bbox的底部中心点
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)