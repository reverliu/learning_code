import numpy as np
from gaussion_mixture import run

pro_user_latitude_longitude_list = list(np.load("pro_user_latitude_longitude_list.npy", allow_pickle=True))
tmp_user_con = list(np.load("tmp_user_con.npy", allow_pickle=True))

# a = [5, 7, 11, 12, 20, 27, 29, 25, 38, 42, 44, 50, 51, 52, 54, 69, 72, 86, 90, 96, 97, 103, 105, 106, 109, 115, 120,
#      121, 123, 118]
# 高斯分布拟合
# a = [27, 42, 105, 96, 120, 121, 126, 149, 154, 171, 174, 252, 296, 305, 312, 320, 13]
a = [26, 39, 92, 100, 114, 118, 127, 140, 144, 160, 239, 282, 290, 304]
gaussion_list = []
for user_idx in range(280, len(tmp_user_con)):
    if user_idx not in a:
        print(user_idx)
        new_pi, tmp_center_list, tmp_con_list = run(pro_user_latitude_longitude_list[user_idx], tmp_user_con[user_idx])
        gaussion_list.append([new_pi.tolist(), tmp_center_list.tolist(), tmp_con_list.tolist()])

# 划分区域后，计算每个区域的概率
# divideArea = DivideArea(poi_coos, 1000, 1000)
# divideArea.divide()
np.save("new_result/pi_center_conxy_280_307.npy", np.array(gaussion_list))
