# 计算区域的概率然后进行归一化，计算两个用户的矩阵的地理相似度
import numpy as np
from numpy.linalg import det, inv
from divide_area import DivideArea
import time

# 多维高斯分布
def new_gaussion(x, mu, sigma):
    dim = len(x)
    constant = (2.0 * np.pi) ** (-dim / 2.0) * det(sigma) ** (-0.5)
    return constant * np.exp(-0.5 * (x - mu).dot(inv(sigma)).dot(x - mu))


# 高斯混合分布 pi：每个高斯分布的权重列表
def new_gaussion_mixture(x, pi, mu, sigma):
    z = 0
    for idx in range(len(pi)):
        z += pi[idx] * new_gaussion(x, mu[idx], sigma[idx])
    return z


if __name__ == '__main__':
    poi_coos = np.loadtxt("dataset/poi_coos.txt")
    latitude_longitude_list = [[x[1], x[2]] for x in poi_coos]
    divideArea = DivideArea(latitude_longitude_list, 1000, 1000)
    divideArea.divide()
    map_set = []
    for _ in divideArea.area:
        map_set.append(divideArea.area[_])
    pi_center_conxy = np.load("new_result/pi_center_conxy_result.npy", allow_pickle=True)
    user_area_matrix_set = []
    for user_idx in range(0, 100):
        print("===========")
        print(user_idx)
        tmp = []
        for _ in map_set:
            area_matrix = new_gaussion_mixture(np.array(_), np.array(pi_center_conxy[user_idx][0]),
                                               np.array(pi_center_conxy[user_idx][1]),
                                               np.array(pi_center_conxy[user_idx][2]))
            tmp.append(area_matrix)
        user_area_matrix_set.append(tmp)
    np.save("user_area_matrix_0_100.npy", np.array(user_area_matrix_set))
