import torch
from config.BaseConfig import BaseConfig
from src.model.CNN import CNN
from src.model.CNN import PlantCNN, PlantCNN_v2
from src.model.Perceptron import Perceptron

def build_model(config: BaseConfig, device: torch.device):
  """根據 config 初始化模型並處理特定的硬體加速設定"""
  # 其他模型（如 LocallyConnectedNet）也可以循相同模式加入
  if config.model_type == 'CNN':
    model = CNN_Net().to(device)
    if device.type == 'cuda':
      torch.backends.cudnn.benchmark = True # CNN cuda 加入
  elif config.model_type == 'Perceptron':
    model = Perceptron().to(device)
  elif config.model_type == 'PlantCNN':
    model = PlantCNN().to(device)
  elif config.model_type == 'PlantCNN_v2':
    model = PlantCNN_v2().to(device)
  else:
    raise ValueError(f"不支援的模型類型: {config.model_type}")
  return model
