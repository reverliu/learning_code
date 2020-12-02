import math
import numpy as np
from scipy.stats import pearsonr


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return round(tmp / float(non), 8)


def eculidDisSim(x, y):
    '''
    欧几里得相似度
    '''
    return math.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def jaccardDisSim(x, y):
    '''
    杰卡德相似度
    '''
    res = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return res / float(union_cardinality)


def perasonrSim(x, y):
    '''
    皮尔森相似度
    '''
    return pearsonr(x, y)[0]


def manhattanDisSim(x, y):
    '''
    曼哈顿距离计算相似度
    '''
    return sum(abs(a - b) for a, b in zip(x, y))


