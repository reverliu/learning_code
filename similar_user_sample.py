from similar_calculation import cosSim, eculidDisSim, jaccardDisSim, perasonrSim, manhattanDisSim
import numpy as np


class Sample:
    def __init__(self, k, tmp_user_sum, tmp_poi_sum, tmp_train_data):
        self.k = k
        self.tmp_user_sum = tmp_user_sum
        self.tmp_poi_sum = tmp_poi_sum
        self.tmp_train_data = tmp_train_data
        self.cosSim_user_matrix = None
        self.eculidDisSim_user_matrix = None
        self.jaccardDisSim_user_matrix = None
        self.perasonrSim_user_matrix = None
        self.manhattanDisSim_matrix = None

    # 构建用户地点签到矩阵
    def construct_user_poi_matrix(self):
        tmp_check_in_matrix = np.zeros((self.tmp_user_sum, self.tmp_poi_sum))
        for _ in self.tmp_train_data:
            tmp_check_in_matrix[int(_[0]), int(_[1])] += 1
        return tmp_check_in_matrix

    # 获取用户与用户之间相似度矩阵
    def calculate_user_similar(self):
        tmp_check_in_matrix = self.construct_user_poi_matrix()
        tmp_cosSim_user_matrix = np.zeros((self.tmp_user_sum, self.tmp_user_sum))
        tmp_eculidDisSim_user_matrix = tmp_cosSim_user_matrix.copy()
        tmp_jaccardDisSim_user_matrix = tmp_cosSim_user_matrix.copy()
        tmp_perasonrSim_user_matrix = tmp_cosSim_user_matrix.copy()
        tmp_manhattanDisSim_matrix = tmp_cosSim_user_matrix.copy()
        for user_idx in range(self.tmp_user_sum):
            for user_jdx in range(user_idx + 1, self.tmp_user_sum):
                tmp_cosSim_user_matrix[user_idx, user_jdx] = cosSim(tmp_check_in_matrix[user_idx],
                                                                    tmp_check_in_matrix[user_jdx])
                tmp_cosSim_user_matrix[user_jdx, user_idx] = tmp_cosSim_user_matrix[user_idx, user_jdx]

                tmp_eculidDisSim_user_matrix[user_idx, user_jdx] = eculidDisSim(tmp_check_in_matrix[user_idx],
                                                                                tmp_check_in_matrix[user_jdx])
                tmp_eculidDisSim_user_matrix[user_jdx, user_idx] = tmp_eculidDisSim_user_matrix[user_idx, user_jdx]

                tmp_jaccardDisSim_user_matrix[user_idx, user_jdx] = jaccardDisSim(tmp_check_in_matrix[user_idx],
                                                                                  tmp_check_in_matrix[user_jdx])
                tmp_jaccardDisSim_user_matrix[user_jdx, user_idx] = tmp_jaccardDisSim_user_matrix[user_idx, user_jdx]

                tmp_perasonrSim_user_matrix[user_idx, user_jdx] = perasonrSim(tmp_check_in_matrix[user_idx],
                                                                              tmp_check_in_matrix[user_jdx])
                tmp_perasonrSim_user_matrix[user_jdx, user_idx] = tmp_perasonrSim_user_matrix[user_idx, user_jdx]

                tmp_manhattanDisSim_matrix[user_idx, user_jdx] = manhattanDisSim(tmp_check_in_matrix[user_idx],
                                                                                 tmp_check_in_matrix[user_jdx])
                tmp_manhattanDisSim_matrix[user_jdx, user_idx] = tmp_manhattanDisSim_matrix[user_idx, user_jdx]
        self.cosSim_user_matrix = tmp_cosSim_user_matrix
        self.eculidDisSim_user_matrix = tmp_eculidDisSim_user_matrix
        self.jaccardDisSim_user_matrix = tmp_jaccardDisSim_user_matrix
        self.perasonrSim_user_matrix = tmp_perasonrSim_user_matrix
        self.manhattanDisSim_matrix = tmp_manhattanDisSim_matrix

    # 对相似度列表进行排序，避免相似度相等的情况出现
    def sort_sequence(self, sequence_x):
        set_a = list(set(map(lambda x: x, sequence_x)))
        set_a.sort(reverse=True)
        grouped_a = dict()
        for idx in set_a:
            tmp_a = []
            for _ in range(len(sequence_x)):
                if sequence_x[_] == idx and sequence_x[_] != 0:
                    tmp_a.append(_)
                    sequence_x[_] = 0
            grouped_a[idx] = tmp_a
        new_sequence = []
        for _ in grouped_a.values():
            new_sequence.extend(_)
        return new_sequence

    # 获取所有用户的采样用户集
    def sample_user(self):
        tmp_all_user_sample_list = []
        for idx in range(self.tmp_user_sum):
            tmp_user_sample_list = []
            cosSim_re1 = self.sort_sequence(list(self.cosSim_user_matrix[idx]))
            eculidDisSim_re1 = self.sort_sequence(list(self.eculidDisSim_user_matrix[idx]))
            jaccardDisSim_re1 = self.sort_sequence(list(self.jaccardDisSim_user_matrix[idx]))
            perasonrSim_re1 = self.sort_sequence(list(self.perasonrSim_user_matrix[idx]))
            manhattanDisSim_re1 = self.sort_sequence(list(self.manhattanDisSim_matrix[idx]))
            tmp_user_sample_list.append(
                [cosSim_re1[:self.k], eculidDisSim_re1[:self.k], jaccardDisSim_re1[:self.k], perasonrSim_re1[:self.k],
                 manhattanDisSim_re1[:self.k]])
            tmp_all_user_sample_list.append(tmp_user_sample_list[0])
        return tmp_all_user_sample_list
