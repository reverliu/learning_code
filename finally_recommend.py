import numpy as np

from baye_model import train_score
from data_pre_solving import dataSolving
from metrics import precisionk, recallk, ndcgk, mapk
from similar_user_sample import Sample


def read_train_data(filename):
    train_data = open(filename, 'r').readlines()
    training_tuples = set()
    for eachline in train_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid, = int(uid), int(lid)
        training_tuples.add((uid, lid))
    return training_tuples


if __name__ == '__main__':
    training_tuple = read_train_data("dataset/train_data.txt")
    print("------实验开始-----")
    _data = dataSolving("dataset/train_data.txt", "dataset/test_data.txt", "dataset/check_in.txt",
                        "dataset/poi_coos.txt", "dataset/data_size.txt")
    data_size = np.loadtxt("dataset/data_size.txt")
    user_sum, poi_sum = int(data_size[0]), int(data_size[1])
    tmp_poi_coos = np.loadtxt("dataset/poi_coos.txt")
    poi_coos_list = [list(x) for x in tmp_poi_coos]
    tmp_train_data = np.loadtxt("dataset/train_data.txt")
    train_data = [list(x) for x in tmp_train_data]
    tmp_test_data = np.loadtxt("dataset/test_data.txt")
    test_data = [list(x) for x in tmp_test_data]
    print("-------数据加载完成-------")
    sample = Sample(3, user_sum, poi_sum, train_data)
    sample.calculate_user_similar()
    all_user_sample_list = sample.sample_user()
    # print(all_user_sample_list)
    print("------所有用户采样完成-------")
    user_probability = list(np.load("1_baye_file/user_probability.npy", allow_pickle=True))
    item_prior = np.load("1_baye_file/item_prior.npy", allow_pickle=True)
    _user_item_condition_probability = np.load("1_baye_file/user_item_condition_probability.npy", allow_pickle=True)
    user_item_condition_probability = list(map(lambda x: list(x), _user_item_condition_probability))
    print("-----数据加载完成-----")
    all_user_location_score = []
    for _sample_user in all_user_sample_list:
        user_location_score = []
        for _ in _data.get_location_list():
            # train_score(_sample_user, _, user_item_condition_probability, user_probability, item_prior):
            # 计算得出一个用户在一个地点处的得分列表，长度为采样数
            # user_location_score： 计算得出用户在每一个地点处的得分列表，长度为地点数目
            user_location_score.append(
                train_score(_sample_user, _, user_item_condition_probability, user_probability, item_prior))
        # 总的信息：长度为用户总数，小列表长度为地点数目
        all_user_location_score.append(user_location_score)
    print("-----各子分类器的结果产生完毕-----")
    # print(all_user_location_score)
    user_similarity = np.load("user_area/user_similarity.npy", allow_pickle=True)
    # 每个用户总的签到次数列表
    user_count_check_in_record = []
    for user_idx in range(len(all_user_sample_list)):
        _check_in_record = list(filter(lambda x: int(x[0]) == user_idx, train_data))
        user_count_check_in_record.append(len(_check_in_record))

    # check_in_count_list为每个用户的子用户集的总签到次数列表
    check_in_count_list = []
    # 签到次数与相似度结合
    combine_num_similarity = []
    for user_idx in range(len(all_user_sample_list)):
        # 一个用户总的签到次数列表
        tmp_check_in_count = []
        _combine_num_similarity = []
        for _user_idx in range(len(all_user_sample_list[user_idx])):
            _check_in_count = list(filter(lambda x: int(x[0]) in all_user_sample_list[user_idx][_user_idx], train_data))
            tmp_check_in_count.append(len(_check_in_count))
            c = 0
            for __user_idx in all_user_sample_list[user_idx][_user_idx]:
                c += user_similarity[user_idx][__user_idx] * user_count_check_in_record[__user_idx]
            _combine_num_similarity.append(c)
        check_in_count_list.append(tmp_check_in_count)
        combine_num_similarity.append(_combine_num_similarity)

    weights = []
    for user_idx in range(len(all_user_sample_list)):
        _weights = []
        for _ in range(len(all_user_sample_list[user_idx])):
            new_result = combine_num_similarity[user_idx][_] / check_in_count_list[user_idx][_]
            _weights.append(new_result)
        weights.append(_weights)
    print("------权重计算完成-----------")
    all_user_check_in_score_list = []
    for user_idx in range(len(weights)):
        a_user_check_in_score_list = []
        for location_idx in range(poi_sum):
            result_ = np.multiply(np.array(all_user_location_score[user_idx][location_idx]),
                                  np.array(weights[user_idx]))
            a_user_check_in_score_list.append(sum(result_))
        all_user_check_in_score_list.append(a_user_check_in_score_list)
    print("------所有用户在所有地点的得分计算完成-----------")
    print("===========开始推荐================")
    test_user = list(set(map(lambda x: int(x[0]), _data.test_data)))

    ground_truth = dict()
    for idx in range(len(_data.get_user_list())):
        if idx in test_user:
            ground_truth[idx] = [x[1] for x in _data.test_data if x[0] == idx]

    rec_list = open("./result/reclist_top_" + str(100) + ".txt", 'w')
    result_5 = open("./result/result_top_" + str(5) + ".txt", 'w')
    result_10 = open("./result/result_top_" + str(10) + ".txt", 'w')
    result_20 = open("./result/result_top_" + str(20) + ".txt", 'w')

    all_uids = list(range(len(_data.get_user_list())))
    all_lids = list(range(len(_data.get_location_list())))
    np.random.shuffle(all_uids)
    # list for different ks
    precision_5, recall_5, nDCG_5, MAP_5 = [], [], [], []
    precision_10, recall_10, nDCG_10, MAP_10 = [], [], [], []
    precision_20, recall_20, nDCG_20, MAP_20 = [], [], [], []
    for cnt, uid in enumerate(all_uids):
        if uid in test_user:
            overall_scores = [all_user_check_in_score_list[uid][lid]
                              if (uid, lid) not in training_tuple else -1
                              for lid in all_lids]

            overall_scores = np.array(overall_scores)

            predicted = list(reversed(overall_scores.argsort()))[:100]
            actual = ground_truth[uid]

            # calculate the average of different k
            precision_5.append(precisionk(actual, predicted[:15]))
            recall_5.append(recallk(actual, predicted[:15]))
            nDCG_5.append(ndcgk(actual, predicted[:15]))
            MAP_5.append(mapk(actual, predicted[:15], 15))

            precision_10.append(precisionk(actual, predicted[:25]))
            recall_10.append(recallk(actual, predicted[:25]))
            nDCG_10.append(ndcgk(actual, predicted[:25]))
            MAP_10.append(mapk(actual, predicted[:25], 25))

            precision_20.append(precisionk(actual, predicted[:30]))
            recall_20.append(recallk(actual, predicted[:30]))
            nDCG_20.append(ndcgk(actual, predicted[:30]))
            MAP_20.append(mapk(actual, predicted[:30], 30))

            print(cnt, uid, "pre@10:", np.mean(precision_10), "rec@10:", np.mean(recall_10))

            rec_list.write('\t'.join([
                str(cnt),
                str(uid),
                ','.join([str(lid) for lid in predicted])
            ]) + '\n')

            # write the different ks
            result_5.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_5)), str(np.mean(recall_5)),
                                      str(np.mean(nDCG_5)), str(np.mean(MAP_5))]) + '\n')
            result_10.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_10)), str(np.mean(recall_10)),
                                       str(np.mean(nDCG_10)), str(np.mean(MAP_10))]) + '\n')
            result_20.write('\t'.join([str(cnt), str(uid), str(np.mean(precision_20)), str(np.mean(recall_20)),
                                       str(np.mean(nDCG_20)), str(np.mean(MAP_20))]) + '\n')

    print("<< Task Finished >>")