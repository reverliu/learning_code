import numpy as np


# 划分区域
class DivideArea:
    def __init__(self, tmp_latitude_longitude_list, x_line, y_line):
        self.tmp_latitude_longitude_list = tmp_latitude_longitude_list
        self.x_line = x_line
        self.y_line = y_line
        tmp_latitude_list = [float(x[0]) for x in tmp_latitude_longitude_list]
        tmp_longitude_list = [float(x[1]) for x in tmp_latitude_longitude_list]
        self.min_latitude = min(tmp_latitude_list)
        self.max_latitude = max(tmp_latitude_list)
        self.min_longitude = min(tmp_longitude_list)
        self.max_longitude = max(tmp_longitude_list)
        self.area = None

    def divide(self):
        #  1. 求每一个区域的长度
        average_latitude = (self.max_latitude - self.min_latitude) / self.x_line
        average_longitude = (self.max_longitude - self.min_longitude) / self.y_line
        # 2. 均匀划分长度，产生列表
        tmp_list_latitude = list(np.arange(self.min_latitude, self.max_latitude, average_latitude))
        tmp_list_latitude.append(self.max_latitude)
        tmp_list_longitude = list(np.arange(self.min_longitude, self.max_longitude, average_longitude))
        tmp_list_longitude.append(self.max_longitude)
        # 3. 分区域
        area = dict()
        for x in range(self.x_line):
            for y in range(self.y_line):
                area[(x, y)] = [(tmp_list_latitude[x] + tmp_list_latitude[x + 1]) / 2.0,
                                (tmp_list_longitude[y] + tmp_list_longitude[y + 1]) / 2.0]
        self.area = area


if __name__ == '__main__':
    poi_coos = np.loadtxt("dataset/poi_coos.txt")
    latitude_longitude_list = [[x[1], x[2]] for x in poi_coos]
    divideArea = DivideArea(latitude_longitude_list, 10000, 10000)
    divideArea.divide()
    d = divideArea.area
    print(d)
