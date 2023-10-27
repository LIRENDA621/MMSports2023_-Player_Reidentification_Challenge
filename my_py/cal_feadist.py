import pickle
import torch
import numpy as np
from sklearn.metrics import pairwise_distances

with open('ckp/query_fea.pkl', 'rb') as f:
    data = pickle.load(f)   

assert isinstance(data, torch.Tensor)
assert data.dim() == 2

dist_matrix = pairwise_distances(data, metric='euclidean')  # 使用欧几里得距离，也可以使用其他距离度量

# 获取类别数
class_num = data.size(0)

# 计算每个类别的平均距离
mean_distances = []
for i in range(class_num):
    mean_distance = dist_matrix[i, :].sum() / (class_num - 1)  # 除去自身的距离
    mean_distances.append(mean_distance)

average_mean_distance = np.mean(mean_distances)

print(f"类别特征类别间的分散程度指标（平均距离）：{average_mean_distance}")
