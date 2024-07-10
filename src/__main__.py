import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color, io
from skimage.segmentation import active_contour

img_path = "amp.jpg" # 画像ファイルのパス
img = cv2.imread(img_path)
img = cv2.resize(img,(300,300))
img = cv2.bitwise_not(img)
img=img[80:300,0:300]
img_g = color.rgb2gray(img)

# 初期輪郭の生成（ 楕円）
s = np.linspace(0, 2 * np.pi, 400)
x = 147 + 70 * np.cos(s)
y = 113 + 105 * np.sin(s)
init = np.array([x, y]).T

# スネークアルゴリズムの適用
snk = active_contour(img_g, init, alpha=0.015, beta=10, gamma=0.001)
# 結果の表示
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 0], init[:, 1], "--r", lw=3, label="Initial contour")
ax.plot(snk[:, 0], snk[:, 1], "-b", lw=3, label="Final contour")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="best")

plt.show()