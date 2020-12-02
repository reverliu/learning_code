import numpy as np
a = [21, 222, 89, 225, 18, 278, 36, 272, 114, 50]
b = [28, 34, 24, 22, 29, 53, 29, 30, 34, 29]
user_si = np.load("user_area/user_similarity.npy", allow_pickle=True)
c = 0

for i in range(len(a)):
    c += user_si[1][a[i]] * b[i]
print(c / sum(b))
