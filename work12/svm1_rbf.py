
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# データのロードと前処理
def load_images_from_folder(folder):
    imgs = []
    lbls = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # 画像を64x64にリサイズ
            imgs.append(img)
            lbl = filename.split('_')[0]  # ファイル名の先頭部分をラベルとする
            lbls.append(lbl)
    return imgs, lbls

# 画像データとラベルの取得
imgs, lbls = load_images_from_folder('./data01')

# 画像データをnumpy配列に変換
imgs = np.array(imgs)
lbls = np.array(lbls)

# RGB値を特徴量とする関数
def extract_rgb_features(images):
    feats = []
    for img in images:
        img_feats = img.reshape(-1, 3)  # 画像をピクセルごとにRGB値に変換
        feats.append(img_feats)
    return np.array(feats)

# 特徴量の抽出
feats = extract_rgb_features(imgs)

# ラベルのエンコーディング
le = LabelEncoder()
lbls = le.fit_transform(lbls)

# データの分割
X_train, X_test, y_train, y_test, imgs_train, imgs_test = train_test_split(
    feats, lbls, imgs, test_size=0.3, random_state=42
)

# SVMモデルの訓練（RGB値を特徴量とする場合）
# 特徴量を適切に変形
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 特徴量の標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled = scaler.transform(X_test_flat)

# SVMの訓練
svm = SVC(kernel='rbf', degree=3)
svm.fit(X_train_scaled, y_train)

# テストデータでの予測
y_pred = svm.predict(X_test_scaled)

# 精度の計算
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc * 100:.2f}%')

# 主成分分析（PCA）による次元削減
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 特徴空間でのプロット
def plot_feature_space(X_pca, y, title):
    plt.figure(figsize=(12, 8))
    for label in np.unique(y):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=le.inverse_transform([label])[0])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.legend()
    plt.show()

# 訓練データの特徴空間プロット
plot_feature_space(X_train_pca, y_train, 'Train Feature Space')

# テストデータの特徴空間プロット
plot_feature_space(X_test_pca, y_test, 'Test Feature Space')

# 5×10のタイル状に分類結果を表示する関数
def display_classification_results(images, true_labels, pred_labels, label_encoder, num_rows=10, num_cols=5):
    plt.figure(figsize=(15, 30))
    for i in range(num_rows * num_cols):
        if i >= len(images):
            break
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(f'True: {label_encoder.inverse_transform([true_labels[i]])[0]}\nPred: {label_encoder.inverse_transform([pred_labels[i]])[0]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# 分類結果を5×10のタイル状に表示
display_classification_results(imgs_test, y_test, y_pred, le, num_rows=10, num_cols=5)
