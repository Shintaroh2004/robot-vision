import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score

# データセットの読み込み
digits = load_digits()

# KMeansクラスタリング
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
comp, clusters, centers = cv2.kmeans(digits.data.astype(np.float32), 10, None, criteria, 10, flags)

# クラスタ中心のプロット
plt.style.use('ggplot')
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
centers = centers.reshape(10, 8, 8)
for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

# 実際のラベルのモードに基づいてラベルを割り当て
labels = np.zeros_like(clusters.ravel())
for i in range(10):
    mask = (clusters.ravel() == i)
    labels[mask] = mode(digits.target[mask])[0]

# 各クラスタに属する最初の10個のサンプルを表示
fig, ax = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    cluster_indices = np.where(clusters.ravel() == i)[0]
    for j in range(10):
        ax[i, j].imshow(digits.images[cluster_indices[j]], cmap='gray')
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

plt.show()

# 精度の計算
accuracy = accuracy_score(digits.target, labels)
print(accuracy)  # => 0.7417918753478019