from collections import Counter
import numpy as np
from math import radians, cos, sin, asin, sqrt
from gaussion_mixture import gaussion, run


# 公式计算两点间距离（m）
def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)
    return distance


# 获取一个用户在某一个中心点的协方差conxy
def one_calculation_user_con(tmp_user_idx, tmp_center_idx, poi_list, train_check_in_data):
    user_check_in_poi_list = [[poi_list[int(x[1])][0], poi_list[int(x[1])][1]] for x in train_check_in_data if
                              int(x[0]) == tmp_user_idx]
    tmp_latitude_list = [float(x[0]) for x in user_check_in_poi_list]
    tmp_longitude_list = [float(x[1]) for x in user_check_in_poi_list]
    tmp_user_con = []
    tmp_latitude_con = calculation_conxy(np.array(tmp_latitude_list), np.array(tmp_latitude_list),
                                         tmp_center_idx[0], tmp_center_idx[0])
    tmp_longitude_con = calculation_conxy(np.array(tmp_longitude_list), np.array(tmp_longitude_list),
                                          tmp_center_idx[1], tmp_center_idx[1])
    tmp_latitude_longitude_con = calculation_conxy(np.array(tmp_latitude_list), np.array(tmp_longitude_list),
                                                   tmp_center_idx[0], tmp_center_idx[1])
    tmp_user_con.append([tmp_latitude_con, tmp_latitude_longitude_con])
    tmp_user_con.append([tmp_latitude_longitude_con, tmp_longitude_con])
    return tmp_user_con


# 计算标准差
def calculation_std(a, x_mean):
    return np.sqrt(((a - x_mean) ** 2).sum() / (len(a) - 1))


# 计算x与y的相关系数
def calculation_conxy(x, y, x_mean, y_mean):
    count = (x - x_mean) * (y - y_mean)
    return sum(count) / (len(x) - 1)


# 寻找中心点与最近的点
# a: 为中心点: 地点的id
# b: 为需要判断的经纬度点: 地点的id
def find_small_distance(a, b, tmp_location_distance_matrix):
    tmp_distance = []
    for x in range(len(a)):
        for y in range(len(b)):
            tmp_distance.append(tmp_location_distance_matrix[int(a[x])][int(b[y])])
    return a[tmp_distance.index(min(tmp_distance)) // len(b)], b[tmp_distance.index(min(tmp_distance)) % len(b)]


# 建立用户的混合高斯，进而获取用户之间的相似度
tmp_train_data = np.loadtxt("dataset/train_data.txt")
train_data = [list(x) for x in tmp_train_data]
data_size = np.loadtxt("dataset/data_size.txt")
user_sum, poi_sum = int(data_size[0]), int(data_size[1])
tmp_poi_coos = np.loadtxt("dataset/poi_coos.txt")
poi_coos = [[x[1], x[2]] for x in tmp_poi_coos]

# 获取所有地点之间的距离
location_distance_matrix = np.zeros((len(poi_coos), len(poi_coos)))
for location_idx in range(len(poi_coos)):
    for _location_idx in range(location_idx + 1, len(poi_coos)):
        location_distance_matrix[location_idx, _location_idx] = geodistance(poi_coos[location_idx][1],
                                                                            poi_coos[location_idx][0],
                                                                            poi_coos[_location_idx][1],
                                                                            poi_coos[_location_idx][0])
        location_distance_matrix[_location_idx, location_idx] = location_distance_matrix[location_idx, _location_idx]

# need_tmp_user_location_list: 需要做判断的用户地点签到列表
# tmp_user_center_dict: 签到次数最多的中心点
tmp_user_center_dict = dict()
need_tmp_user_location_list = []
for user_idx in range(user_sum):
    tmp_center = []
    # 1. 获取用户所有的签到地点
    user_check_in_record = [x[1] for x in train_data if int(x[0]) == user_idx]
    # 2. 获取用户的多个访问中心点
    counter = Counter(user_check_in_record)
    for key, value in counter.items():
        if value == counter.most_common(1)[0][1]:
            tmp_center.append([poi_coos[int(key)][0], poi_coos[int(key)][1]])
    tmp_user_center_dict[user_idx] = tmp_center
    tmp_user_location = [x for x in user_check_in_record if
                         [poi_coos[int(x)][0], poi_coos[int(x)][1]] not in tmp_center]
    need_tmp_user_location_list.append(tmp_user_location)

# all_user_center_list: 所有用户所需要建立的中心点列表
all_user_center_list = []
for user_idx in range(user_sum):
    _user_center_list = tmp_user_center_dict[user_idx]
    user_center_list = [poi_coos.index(x) for x in _user_center_list]
    need_user_location_list = list(set(need_tmp_user_location_list[user_idx]))
    i = len(need_user_location_list)
    while True:
        # center_idx，need_idx为两点之间最近的点
        center_idx, need_idx = find_small_distance(user_center_list, need_user_location_list, location_distance_matrix)
        center_idx, need_idx = int(center_idx), int(need_idx)
        user_idx_conxy = one_calculation_user_con(user_idx, poi_coos[center_idx], poi_coos, train_data)
        need_idx_conxy = one_calculation_user_con(user_idx, poi_coos[need_idx], poi_coos, train_data)
        user_idx_result = gaussion(np.array(poi_coos[need_idx]), np.array(poi_coos[center_idx]), user_idx_conxy)
        need_idx_result = gaussion(np.array(poi_coos[need_idx]), np.array(poi_coos[need_idx]), need_idx_conxy)
        if user_idx_result < need_idx_result:
            user_center_list.append(need_idx)
        need_user_location_list.remove(need_idx)
        i -= 1
        if i == 0:
            break
    all_user_center_list.append(user_center_list)

# 计算每一个中心点的协方差矩阵
tmp_user_con = []
# 将地点id中心点列表转换为经度纬度列表
pro_user_latitude_longitude_list = []
for user_idx in range(len(all_user_center_list)):
    user_latitude_longitude_list = [poi_coos[int(x)] for x in all_user_center_list[user_idx]]
    pro_user_latitude_longitude_list.append(user_latitude_longitude_list)
    tmp_user_list = []
    for _ in user_latitude_longitude_list:
        tmp_user_ = one_calculation_user_con(user_idx, _, poi_coos, train_data)
        tmp_user_list.append(tmp_user_)
    tmp_user_con.append(tmp_user_list)


np.save("pro_user_latitude_longitude_list.npy", np.array(pro_user_latitude_longitude_list))
np.save("tmp_user_con.npy", np.array(tmp_user_con))




