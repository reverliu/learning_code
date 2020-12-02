from math import radians, cos, sin, asin, sqrt
import numpy as np


# 计算每个区域的面积
def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000


if __name__ == '__main__':
    poi_coos_list = np.loadtxt("dataset/poi_coos.txt")
    latitude_list = list(map(lambda x: x[1], poi_coos_list))
    longitude_list = list(map(lambda x: x[2], poi_coos_list))
    average_lat = (max(latitude_list) - min(latitude_list)) / 1000
    average_lgt = (max(longitude_list) - min(longitude_list)) / 1000
    print(min(latitude_list))
    print(min(longitude_list))
    print(average_lat)
    print(average_lgt)
    print((40.7651 - min(latitude_list)) / average_lat)
    print((-73.9865 - min(longitude_list)) / average_lgt)
    # print(min(latitude_list) + average_lat)
    # print(min(longitude_list) + average_lgt)
    # x1 = haversine(min(longitude_list), min(latitude_list), min(longitude_list), min(latitude_list) + average_lat)
    # y1 = haversine(min(longitude_list) + average_lgt, min(latitude_list), min(longitude_list), min(latitude_list))
    # print(x1)
    # print(y1)
    # print(x1 * y1)
