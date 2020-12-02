from collections import Counter

import numpy as np

from gaussion_mixture import gaussion_mixture

train_data = np.loadtxt("dataset/train_data.txt")
c = np.load("new_result/pi_center_conxy_0_10.npy", allow_pickle=True)
poi_coos = np.loadtxt("dataset/poi_coos.txt")
user_one = list(filter(lambda x: int(x[0]) == 2, train_data))
location_ids = [int(x[1]) for x in user_one]
poi_information = [[x[1], x[2]] for x in poi_coos if x[0] in location_ids]
poi_informations = [[poi_coos[x][1], poi_coos[x][2]] for x in location_ids]

pi_ = np.array(c[2][0])
center = np.array(c[2][1])
conxy = np.array(c[2][2])
dic_ = []
for _ in poi_information:
    if _ not in dic_:
        dic_.append(_)
num_count={}
for i in poi_informations:
    if (i[0], i[1]) not in num_count:
        num_count[(i[0], i[1])] = 1
    else:
        num_count[(i[0], i[1])] += 1
new_dict = dict()
for _ in dic_:
    new_dict[(_[0], _[1])] = gaussion_mixture(np.array(_), pi_, center, conxy)
new_num_count = sorted(num_count.items(), key=lambda x: -x[1])
print(new_num_count)
new_new_dict = sorted(new_dict.items(), key=lambda x: -x[1])
print(new_new_dict)