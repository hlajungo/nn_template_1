import torch
import matplotlib.pyplot as plt
import random
import numpy as np

def set_all_seeds(seed=42):
  """鎖死所有隨機性，確保實驗可重現"""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed) # 如果有多張 GPU
  
  # 針對 CuDNN 進行底層設定
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

def get_device():
  """Choose cuda or cpu"""
  if torch.cuda.is_available():
    return torch.device('cuda')
  return torch.device('cpu')

def plot_loss(loss_history):
  plt.figure(figsize=(10, 5))
  plt.plot(loss_history, linewidth=2, color='blue', label='Training Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss Curve')
  plt.grid(True)
  plt.legend()
  plt.show()
