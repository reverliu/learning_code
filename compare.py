from collections import defaultdict

import numpy as np

from metrics import precisionk, recallk, ndcgk, mapk


def read_training_data(train_file, user_count, poi_count):
    train_data = open(train_file, 'r').readlines()
    train_matrix = np.zeros((int(user_count), int(poi_count)))
    training_tuples = set()
    for eachline in train_data:
        uid, lid, freq = eachline.strip().split()
        uid, lid, freq = int(uid), int(lid), int(freq)
        train_matrix[uid, lid] = freq
        training_tuples.add((uid, lid))
    return train_matrix, training_tuples


def read_ground_truth(test_file):
    ground_truth = defaultdict(set)  # value type is set
    truth_data = open(test_file, 'r').readlines()
    for eachline in truth_data:
        uid, lid, _ = eachline.strip().split()
        uid, lid = int(uid), int(lid)
        ground_truth[uid].add(lid)
    return ground_truth


user_similarity = np.load("user_area/user_similarity.npy", allow_pickle=True)
for uid in range(len(user_similarity)):
    for _ in range(len(user_similarity[0])):
        if uid == _:
            user_similarity[uid, _] = 0
user_num, poi_num = open("dataset/data_size.txt", 'r').readlines()[0].strip('\n').split()
train_matrix, training_tuples = read_training_data("dataset/train_data.txt", user_num, poi_num)
score_matrix = np.zeros((int(user_num), int(poi_num)))
for uid in range(int(user_num)):
    for lid in range(int(poi_num)):
        score_matrix[uid, lid] = sum(np.multiply(np.array(user_similarity[uid]), np.array(train_matrix[:, lid]))) / sum(
            user_similarity[uid])
print(score_matrix)
ground_truth = read_ground_truth("dataset/test_data.txt")
rec_list = open("result_/reclist_top_" + str(100) + ".txt", 'w')
result_5 = open("result_/result_top_" + str(5) + ".txt", 'w')
result_10 = open("result_/result_top_" + str(10) + ".txt", 'w')
result_20 = open("result_/result_top_" + str(20) + ".txt", 'w')

all_uids = list(range(int(user_num)))
all_lids = list(range(int(poi_num)))
np.random.shuffle(all_uids)
# list for different ks
precision_5, recall_5, nDCG_5, MAP_5 = [], [], [], []
precision_10, recall_10, nDCG_10, MAP_10 = [], [], [], []
precision_20, recall_20, nDCG_20, MAP_20 = [], [], [], []
for cnt, uid in enumerate(all_uids):
    if uid in ground_truth:
        overall_scores = [score_matrix[uid, lid]
                          if (uid, lid) not in training_tuples else -1
                          for lid in all_lids]

        overall_scores = np.array(overall_scores)

        predicted = list(reversed(overall_scores.argsort()))[:100]
        actual = ground_truth[uid]

        # calculate the average of different k
        precision_5.append(precisionk(actual, predicted[:5]))
        recall_5.append(recallk(actual, predicted[:5]))
        nDCG_5.append(ndcgk(actual, predicted[:5]))
        MAP_5.append(mapk(actual, predicted[:5], 5))

        precision_10.append(precisionk(actual, predicted[:10]))
        recall_10.append(recallk(actual, predicted[:10]))
        nDCG_10.append(ndcgk(actual, predicted[:10]))
        MAP_10.append(mapk(actual, predicted[:10], 10))

        precision_20.append(precisionk(actual, predicted[:20]))
        recall_20.append(recallk(actual, predicted[:20]))
        nDCG_20.append(ndcgk(actual, predicted[:20]))
        MAP_20.append(mapk(actual, predicted[:20], 20))

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
