import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt


# 多维高斯分布
def gaussion(x, mu, sigma):
    dim = len(x)
    constant = (2.0 * np.pi) ** (-dim / 2.0) * det(sigma) ** (-0.5)
    return constant * np.exp(-0.5 * (x - mu).dot(inv(sigma)).dot(x - mu))


# 高斯混合分布 pi：每个高斯分布的权重列表
def gaussion_mixture(x, pi, mu, sigma):
    z = 0
    for idx in range(len(pi)):
        z += pi[idx] * gaussion(x, mu[idx], sigma[idx])
    return z


def EM_step(x, pi, mu, sigma):
    N = len(x)
    K = len(pi)
    gamma = np.zeros((N, K))

    # E-step
    for n in range(N):
        p_xn = 0
        for k in range(K):
            t = pi[k] * gaussion(x[n], mu[k], sigma[k])
            p_xn += t
            # 计算每一个地点在每一个高斯分布上的概率和权重的相乘，同时进行归一化
            gamma[n, k] = t
        gamma[n] /= p_xn

    # M-step
    for k in range(K):
        # _mu:表示每一个高斯分布的中心点
        _mu = np.zeros(mu[k].shape)
        # _sigma:表示每一个高斯分布的方差
        _sigma = np.zeros(sigma[k].shape)
        N_k = np.sum(gamma[:, k])

        # 更新均值
        for n in range(N):
            # n: 表示第几个地点 x[n]表示地点的精度纬度
            _mu += gamma[n, k] * x[n]
        mu[k] = _mu / N_k

        # 更新方差
        for n in range(N):
            delta = np.asmatrix(x[n] - mu[k]).T
            _sigma += gamma[n, k] * np.array(delta.dot(delta.T))
        sigma[k] = _sigma / N_k

        # 更新权重
        pi[k] = N_k / N
    return pi, mu, sigma


# 随机采样样本
def sampling(pi, mean, cov, N):
    samples = np.array([])
    for idx in range(len(pi)):
        _sample = np.random.multivariate_normal(mean[idx], cov[idx], int(N * pi[idx]))
        samples = np.append(samples, _sample)
    return samples.reshape((-1, mean[0].shape[0]))


# 初始化权重
# center_list: 中心点列表
def init_weight(center_list, con_list):
    # 1. 初始化权重
    tmp_center_sum = len(center_list)
    pi = [round(1.0 / tmp_center_sum, 2) for i in range(tmp_center_sum - 1)]
    pi.append(1.0 - sum(pi))
    pi = np.array(pi)
    # 2. 初始化中心点(均值)
    mu = np.array(center_list)
    sigma = np.array(con_list)
    samples = sampling(pi, mu, sigma, 200)
    return samples, pi


# 绘制混合高斯分布等高线图
def plot_gaussion(Pi, mu, Sigma):
    x = np.linspace(40.55, 40.98, 200)
    y = np.linspace(-74.18, -73.8, 200)
    x, y = np.meshgrid(x, y)
    X = np.array([x.ravel(), y.ravel()]).T
    z = [gaussion_mixture(x, Pi, mu, Sigma) for x in X]
    z = np.array(z).reshape(x.shape)
    return plt.contour(x, y, z)


def run(tmp_center_list, tmp_con_list):
    tmp_samples, pi = init_weight(tmp_center_list, tmp_con_list)
    n_iter = 100
    _pi = pi
    tmp_center_list = np.array(tmp_center_list)
    tmp_con_list = np.array(tmp_con_list)
    for i in range(n_iter):
        _pi, tmp_center_list, tmp_con_list = EM_step(tmp_samples, _pi, tmp_center_list, tmp_con_list)
    return _pi, tmp_center_list, tmp_con_list


if __name__ == '__main__':
    pro_user_latitude_longitude_list = np.load("pro_user_latitude_longitude_list.npy", allow_pickle=True)
    tmp_user_con = np.load("tmp_user_con.npy", allow_pickle=True)

    tmp_center_list = pro_user_latitude_longitude_list[4]
    tmp_con_list = tmp_user_con[4]

    tmp_center_list = np.array(tmp_center_list)
    tmp_con_list = np.array(tmp_con_list)

    _pi, tmp_center_list, tmp_con_list = run(tmp_center_list, tmp_con_list)
    print(_pi)
    print(tmp_center_list)
    print(tmp_con_list)
