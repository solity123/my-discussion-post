import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from multiprocessing import freeze_support

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 超参数
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 3  # 背景、宠物、轮廓
IMAGE_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# 图像 transform
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 掩码 transform
def target_transform(mask):
    mask = transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST)(mask)
    mask = np.array(mask, dtype=np.int64)

    # 映射标签
    mask[mask == 3] = 0
    mask[mask == 1] = 1
    mask[mask == 2] = 2

    return torch.as_tensor(mask, dtype=torch.long)

# 定义 FCN 模型
class FCN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(FCN, self).__init__()

        # 使用预训练的ResNet50作为特征提取器
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 移除最后的全连接层和平均池化层
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # 输出大小: 2048 x 7 x 7 (输入为224x224)

        # 1x1卷积调整通道数
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

        # 上采样模块：总共上采样32倍
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # 提取特征
        x = self.features(x)

        # 通道调整
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 上采样至原始图像大小
        x = self.upsample(x)

        # 断言检查
        assert x.shape[-2:] == (224, 224), f"Output size mismatch: got {x.shape[-2:]}, expected (224, 224)"

        return x


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct_pixels += (predicted == masks).sum().item()
        total_pixels += masks.numel()

    return running_loss / len(dataloader.dataset), correct_pixels / total_pixels

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_pixels += (predicted == masks).sum().item()
            total_pixels += masks.numel()

    return running_loss / len(dataloader.dataset), correct_pixels / total_pixels

def main():
    # 加载数据集
    train_dataset = OxfordIIITPet(
        root='./data',
        split='trainval',
        target_types='segmentation',
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    test_dataset = OxfordIIITPet(
        root='./data',
        split='test',
        target_types='segmentation',
        download=True,
        transform=transform,
        target_transform=target_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    model = FCN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    best_val_loss = float('inf')
    best_model_weights = None

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            torch.save(best_model_weights, 'best_fcn_model.pth')

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print('-' * 50)

    total_time = time.time() - start_time
    print(f'Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s')

    # 绘制损失与准确率曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()
