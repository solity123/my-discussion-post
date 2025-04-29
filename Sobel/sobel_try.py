import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# 设置中文字体（使用系统已安装的中文字体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def visualize_sobel_results(image_path):
    """
    执行Sobel边缘检测并可视化结果

    参数:
        image_path: 输入图像路径
    """
    # 1. 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 2. 执行Sobel边缘检测
    magnitude, direction = sobel_torch(image_path, device=device)

    # 3. 准备可视化
    plt.figure(figsize=(15, 5))

    # 3.1 原始图像
    plt.subplot(1, 3, 1)
    original_img = Image.open(image_path).convert('L')
    plt.imshow(original_img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    # 3.2 边缘强度
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude, cmap='gray')
    plt.title('边缘强度')
    plt.axis('off')

    # 3.3 边缘方向 (使用HSV色彩空间)
    plt.subplot(1, 3, 3)

    # 将方向归一化到[0,1]作为色相(Hue)
    h = (direction + np.pi) / (2 * np.pi)  # 方向映射到[0,1]
    s = np.ones_like(h)  # 饱和度设为1
    v = magnitude / 255.0  # 亮度使用归一化的边缘强度

    # 创建HSV图像并转换为RGB
    hsv_image = np.stack([h, s, v], axis=-1)
    rgb_image = hsv_to_rgb(hsv_image)

    plt.imshow(rgb_image)
    plt.title('边缘方向(颜色)和强度(亮度)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def sobel_torch(image_path, convert_to_grayscale=True, device='cuda'):
    """
    使用PyTorch的Sobel边缘检测

    参数:
        image_path: 输入图像路径
        convert_to_grayscale: 是否转换为灰度图像
        device: 使用 'cuda' 或 'cpu'

    返回:
        edge_magnitude: 边缘强度图像 (NumPy数组, uint8)
        edge_direction: 边缘方向图像 (NumPy数组, 弧度制)
    """
    # 1. 加载图像并转换为张量
    image = Image.open(image_path)
    if convert_to_grayscale:
        image = image.convert('L')

    img_tensor = torch.from_numpy(np.array(image)).float()
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, H, W)
    img_tensor = img_tensor.to(device)

    # 2. 定义Sobel算子核
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 3. 计算梯度
    gx = F.conv2d(img_tensor, sobel_x, padding=1)
    gy = F.conv2d(img_tensor, sobel_y, padding=1)

    # 4. 计算幅值和方向
    edge_magnitude = torch.sqrt(gx ** 2 + gy ** 2)
    edge_direction = torch.atan2(gy, gx)

    # 5. 转换为NumPy数组并归一化幅值
    edge_magnitude = (edge_magnitude / edge_magnitude.max()) * 255
    edge_magnitude = edge_magnitude.squeeze().cpu().numpy().astype(np.uint8)
    edge_direction = edge_direction.squeeze().cpu().numpy()  # 方向范围: [-π, π]

    return edge_magnitude, edge_direction


# 使用示例
if __name__ == "__main__":
    input_image = ".\matteo-catanese-4KrQq8Z6Y5c-unsplash.jpg"  # 替换为你的图像路径
    visualize_sobel_results(input_image)