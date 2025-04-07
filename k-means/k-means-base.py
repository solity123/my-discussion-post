import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters  # 聚类数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.centroids = None  # 聚类中心

    def _initialize_centroids(self, X):
        """K-means++初始化"""
        centroids = [X[np.random.randint(X.shape[0])]]
        for _ in range(1, self.n_clusters):
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            probabilities = distances / distances.sum()
            centroids.append(X[np.random.choice(X.shape[0], p=probabilities)])
        return np.array(centroids)

    def _assign_clusters(self, X):
        """分配每个样本到最近的聚类中心"""
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """更新聚类中心为簇内均值"""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            new_centroids[k] = np.mean(X[labels == k], axis=0)
        return new_centroids

    def fit(self, X):
        """训练K-means模型"""
        self.centroids = self._initialize_centroids(X)
        for _ in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            # 检查是否收敛（中心点移动距离小于阈值）
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
        return labels

    def predict(self, X):
        """预测样本所属簇"""
        return self._assign_clusters(X)


def kmeans_image_segmentation(image_path, k=3):
    # 1. 读取图像并转换为RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 2. 将图像像素转换为二维数组 (N_samples, 3)
    pixels = image.reshape(-1, 3).astype(np.float32)

    # 3. 使用自定义K-means聚类
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit(pixels)

    # 4. 用聚类中心颜色替换原始像素
    segmented = kmeans.centroids[labels].reshape(h, w, 3).astype(np.uint8)

    # 5. 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(image), plt.title("原始图像")
    plt.subplot(122), plt.imshow(segmented), plt.title(f"K-means分割 (K={k})")
    plt.show()

    return segmented


# 使用示例
segmented_image = kmeans_image_segmentation(".\matteo-catanese-4KrQq8Z6Y5c-unsplash.jpg",
                                            k=3)
