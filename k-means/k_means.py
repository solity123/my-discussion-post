import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def preprocess_image(image, blur_kernel=(7, 7), contrast_alpha=1.3, contrast_beta=15):
    """
    图像预处理函数

    参数:
        image: 输入图像(RGB格式)
        blur_kernel: 高斯模糊核大小
        contrast_alpha: 对比度增强系数
        contrast_beta: 亮度调整值

    返回:
        processed_image: 预处理后的图像(Lab颜色空间)
    """
    # 1. 高斯模糊
    blurred = cv2.GaussianBlur(image, blur_kernel, 0)

    # 2. 对比度增强
    contrasted = cv2.convertScaleAbs(blurred, alpha=contrast_alpha, beta=contrast_beta)

    # 3. 转换为Lab颜色空间
    lab_image = cv2.cvtColor(contrasted, cv2.COLOR_RGB2LAB)

    return lab_image


def kmeans_color_segmentation(image_path, k=4, preprocessing=True, display=True):
    """
    K-means彩色图像分割(不自动保存结果)

    参数:
        image_path: 输入图像路径
        k: 聚类数量
        preprocessing: 是否预处理
        display: 是否显示结果

    返回:
        segmented_image: 分割后的RGB图像
    """
    # 1. 读取图像(带异常处理)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2. 应用预处理
    if preprocessing:
        processed_image = preprocess_image(image)
    else:
        processed_image = image.copy()

    # 3. 准备聚类数据
    h, w = processed_image.shape[:2]
    pixel_values = processed_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # 4. 执行K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # 5. 生成分割图像
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape((h, w, 3))

    if preprocessing:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB)

    # 6. 显示结果
    if display:
        plt.figure(figsize=(18, 12))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title('原始图像')
        plt.axis('off')

        if preprocessing:
            plt.subplot(1, 3, 2)
            preprocessed_display = cv2.cvtColor(processed_image, cv2.COLOR_LAB2RGB)
            plt.imshow(preprocessed_display)
            plt.title('预处理后图像')
            plt.axis('off')
            plt.subplot(1, 3, 3)
        else:
            plt.subplot(1, 3, 2)

        plt.imshow(segmented_image)
        plt.title(f'K-means分割 (k={k})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return segmented_image


if __name__ == "__main__":
    try:
        image_path = r"C:\Users\solity\Desktop\photo\matteo-catanese-4KrQq8Z6Y5c-unsplash.jpg"

        # 执行分割(不自动保存)
        segmented = kmeans_color_segmentation(
            image_path,
            k=3,
            preprocessing=True,
            display=True
        )

        # 如需手动保存，可以取消下面代码的注释
        # cv2.imwrite("手动保存结果.png", cv2.cvtColor(segmented, cv2.COLOR_RGB2BGR))

    except Exception as e:
        print(f"发生错误: {e}")
