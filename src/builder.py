import torch
import torch.nn as nn
import torchvision.models as models
from config.BaseConfig import BaseConfig
from src.model.CNN import CNN
from src.model.CNN import PlantCNN, PlantCNN_v2
from src.model.Perceptron import Perceptron


def build_model(config, device):
  """
  使用 torchvision 內建的 ResNet18 結構，但不使用預訓練權重。
  完全從零開始訓練 (Training from scratch)。
  """
  # 關鍵：weights=None 代表隨機初始化權重
  model = models.resnet18(weights=None)

  # 修改最後的全連接層以符合你的類別數
  num_ftrs = model.fc.in_features

  # 加入 Dropout 防過擬合，因為從零訓練特別容易死記硬背
  model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_ftrs, config.num_classes)
  )

  return model.to(device)

# def build_model(config: BaseConfig, device: torch.device):
  # """根據 config 初始化模型並處理特定的硬體加速設定"""
  # # 其他模型（如 LocallyConnectedNet）也可以循相同模式加入
  # if config.model_type == 'CNN':
    # model = CNN_Net().to(device)
    # if device.type == 'cuda':
      # torch.backends.cudnn.benchmark = True # CNN cuda 加入
  # elif config.model_type == 'Perceptron':
    # model = Perceptron().to(device)
  # elif config.model_type == 'PlantCNN':
    # model = PlantCNN().to(device)
  # elif config.model_type == 'PlantCNN_v2':
    # model = PlantCNN_v2().to(device)
  # else:
    # raise ValueError(f"不支援的模型類型: {config.model_type}")
  # return model
