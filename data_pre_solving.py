import numpy as np


class dataSolving:
    def __init__(self, train_data_path, test_data_path, check_in_path, poi_coos_path, data_size_path):
        self.train_data = np.loadtxt(train_data_path, delimiter="\t")
        self.test_data = np.loadtxt(test_data_path, delimiter="\t")
        self.check_in = np.loadtxt(check_in_path, delimiter="\t")
        self.poi_coos = np.loadtxt(poi_coos_path, delimiter="\t")
        self.data_size = np.loadtxt(data_size_path, delimiter="\t")

    def get_user_list(self):
        return list(set(map(lambda x: int(x[0]), self.train_data)))

    def get_user_count_and_location_count(self):
        return list(map(lambda x: int(x), self.poi_coos))

    def get_location_list(self):
        return list(set(map(lambda x: int(x[1]), self.train_data)))
