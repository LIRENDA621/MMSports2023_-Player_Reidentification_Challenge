import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import torch

with open('ckp/test_fea_idloss_dic2.pkl', 'rb') as f: # saved feature pkl file
    feature_dict = pickle.load(f)

features = []
labels = []
for key, value in feature_dict.items():
    features.append(value.numpy())  # 将PyTorch张量转换为NumPy数组
    label = key.split('_')[0]  # 提取第一个数字作为类别
    labels.append(label)

features = np.array(features)

tsne = TSNE(n_components=2, random_state=42)
embedded_features = tsne.fit_transform(features)

unique_labels = list(set(labels))
num_labels = len(unique_labels)
colors = plt.cm.rainbow(np.linspace(0, 1, num_labels))

label_color_map = defaultdict(lambda: 'gray')
for label, color in zip(unique_labels, colors):
    label_color_map[label] = color

# 绘制散点图
plt.figure(figsize=(8, 8))
for i, (x, y) in enumerate(embedded_features):
    plt.scatter(x, y, color=label_color_map[labels[i]], label=labels[i], alpha=0.7)

# 添加图例
legend_labels = [f'Class {label}' for label in unique_labels]

plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

plt.legend().set_visible(False)

plt.title('Distance Visualization of Test Set Features')

plt.savefig('test_fea_idloss2.png')

