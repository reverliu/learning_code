# 根据用户编号id和地点编号id提取数据
import numpy as np

from data_pre_solving import dataSolving


class NaiveBayes:
    def __init__(self, train_data, users_sum, location_sum):
        self.train_data = train_data
        self.users_sum = users_sum
        self.location_sum = location_sum
        self.user_probability = None
        self.item_prior = None
        self.user_item_condition_probability = None

    def userProbability(self):
        '''
        按照用户的编号顺序存储： P(user)
        :return: 每个用户的概率列表
        '''
        user_probability = []
        for _ in range(self.users_sum):
            user_data_list = list(filter(lambda x: x[0] == _, self.train_data))
            user_probability.append(len(user_data_list) / len(self.train_data))
        return user_probability

    def priorAndconditionalProbability(self):
        '''
        使用拉普拉斯修正进行平滑
        按照项目的编号进行存储:P(item) P(user | item)
        :return: 每个项目的概率列表, 以及项目用户的条件概率列表: 项目-用户
        '''
        item_prior = []
        user_item_condition_probability = []
        for item in range(self.location_sum):
            record = list(filter(lambda x: x[1] == item, self.train_data))
            item_prior.append((len(record) + 1) / (len(self.train_data) + self.location_sum))

            _item = []
            for user in range(self.users_sum):
                _record = list(filter(lambda x: x[0] == user, record))
                _item.append((len(_record) + 1) / (len(record) + self.users_sum))
            user_item_condition_probability.append(_item)
        return item_prior, user_item_condition_probability


def train_score(sample_list, target_item, user_item_condition_probability, user_probability, item_prior):
    '''
    训练得出抽样样本在某一位置的签到概率
    :param sample_list:  样本列表
    :param user_probability: 用户概率列表
    :param item_prior: 先验概率
    :param user_item_condition_probability:  条件概率
    :param target_item:  目标项目
    :return:采样样本集对应计算出来的概率列表
            样本与概率一一对应
    '''
    # sample列表:用户编号
    # P(loc | U1, U2, U3) = (P(U1 | loc) * P(U2 | loc) * P(U3 | loc) * P(loc)) / (P(U1) * P(U2) * P(U3))
    result_probability = []
    for sample in sample_list:
        tmp_one, tmp_two = 1, 1
        for p in sample:
            tmp_one *= user_item_condition_probability[target_item][p]
            tmp_two *= user_probability[p]
        result_probability.append(
            (tmp_one * item_prior[target_item]) / tmp_two)
    return result_probability


if __name__ == '__main__':
    _data = dataSolving("dataset/check_in.txt", "dataset/test_data.txt", "dataset/check_in.txt",
                        "dataset/poi_coos.txt", "dataset/data_size.txt")

    naive_model = NaiveBayes(_data.train_data, len(_data.get_user_list()), len(_data.get_location_list()))
    user_probability = naive_model.userProbability()
    item_prior, user_item_condition_probability = naive_model.priorAndconditionalProbability()

    np.save("1_baye_file/user_probability.npy", user_probability)
    np.save("1_baye_file/item_prior.npy", item_prior)
    np.save("1_baye_file/user_item_condition_probability.npy", user_item_condition_probability)
