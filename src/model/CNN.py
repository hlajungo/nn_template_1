import torch.nn as nn

class CNN(nn.Module):
  """For MNIST, 28x28 單色灰階"""
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Conv2d(1, 16, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 5 * 5, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        )

  def forward(self, x):
    return self.net(x)


class PlantCNN(nn.Module):
  """224x224"""
  def __init__(self, num_classes=6):
    super().__init__()
    # 特徵擷取層 (Feature Extractor)
    # 透過 5 層的 Conv + Pool，將 224x224 逐步降維成 7x7
    self.features = nn.Sequential(
        # Block 1: 224x224 -> 112x112
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 2: 112x112 -> 56x56
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 3: 56x56 -> 28x28
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 4: 28x28 -> 14x14
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        # Block 5: 14x14 -> 7x7
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        )

    # 分類層 (Classifier)
    self.classifier = nn.Sequential(
        # AdaptiveAvgPool2d 也就是 GAP：直接把 7x7 的特徵圖壓縮成 1x1
        # 這樣不論前面出來的尺寸是多少，到這邊都會統一，且能大幅減少 Linear 層的參數量
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        # 丟棄 50% 的神經元，強迫模型學習更健全的特徵，防止死背資料
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
        )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x

class PlantCNN_v2(nn.Module):
  """224x224, 加寬版 (Wider CNN) 以提升特徵學習能力"""
  def __init__(self, num_classes=6):
    super().__init__()
    # 特徵擷取層 (通道數全面翻倍)
    self.features = nn.Sequential(
      # Block 1: 3 -> 64
      nn.Conv2d(3, 64, kernel_size=3, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),

      # Block 2: 64 -> 128
      nn.Conv2d(64, 128, kernel_size=3, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),

      # Block 3: 128 -> 256
      nn.Conv2d(128, 256, kernel_size=3, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),

      # Block 4: 256 -> 512
      nn.Conv2d(256, 512, kernel_size=3, padding=1),
      nn.BatchNorm2d(512),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),

      # Block 5: 512 -> 1024
      nn.Conv2d(512, 1024, kernel_size=3, padding=1),
      nn.BatchNorm2d(1024),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(2),
    )

    # 分類層
    self.classifier = nn.Sequential(
      nn.AdaptiveAvgPool2d((1, 1)),
      nn.Flatten(),
      nn.Dropout(p=0.5), # 保持 0.5 防過擬合
      nn.Linear(1024, num_classes) # 輸入維度配合最後一層改為 1024
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
