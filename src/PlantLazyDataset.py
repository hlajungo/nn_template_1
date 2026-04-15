import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class PlantLazyDataset(Dataset):
  def __init__(self, root_dir, target_size=224, samples_per_image=50, green_threshold=0.05):
    self.target_size = target_size
    self.samples_per_image = samples_per_image
    self.green_threshold = green_threshold
    self.scales = [target_size, target_size * 2, target_size * 3] # 1x, 2x, 3x 視野

    self.all_samples = [] # 儲存 (圖片路徑, 標籤索引)
    self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

    # 掃描目錄
    for cls in self.class_names:
      cls_dir = os.path.join(root_dir, cls)
      for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
          self.all_samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))

  def __len__(self):
    # 總樣本數 = 原始圖片張數 * 每張圖想抽樣的次數
    return len(self.all_samples) * self.samples_per_image

  
  def _is_green_enough(self, patch):
    # 轉為 HSV 通道
    hsv_img = np.array(patch.convert('HSV'))
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 綠色區間大約在 35-90 (PIL HSV scale 是 0-255)
    # 綠色大約在 60 (黃綠) 到 130 (青綠) 之間
    # green_mask = (h >= 35) & (h <= 130) & (s > 20) & (v > 20)
    green_mask = (h >= 35) & (h <= 130)
    return np.mean(green_mask) >= self.green_threshold

  # def _is_green_enough(self, patch):
    # hsv_img = np.array(patch.convert('HSV'))
    # h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    # # 寬鬆範圍：涵蓋你的直方圖高峰
    # green_mask = (h >= 25) & (h <= 105) & (s > 20) & (v > 20)
    # actual_ratio = np.mean(green_mask)

    # # 使用初始化時傳入的 threshold
    # return actual_ratio >= self.green_threshold

  # def _is_green_enough(self, patch):
    # # 快速檢查綠色比率 (HSV 空間)
    # hsv_img = np.array(patch.convert('HSV'))
    # hue = hsv_img[:, :, 0]
    # # 綠色區間大約在 35-90 (PIL HSV scale 是 0-255)
    # green_mask = (hue > 35) & (hue < 95)
    # return np.mean(green_mask) >= self.green_threshold
  def __getitem__(self, idx):
    # 根據 idx 找到對應的大圖
    img_path, label = self.all_samples[idx % len(self.all_samples)]

    # Lazy Loading: 只有在需要時才打開圖
    with Image.open(img_path) as full_img:
      full_img = full_img.convert('RGB')
      w, h = full_img.size

      # 嘗試採樣，直到符合綠色閾值或達到上限
      for _ in range(15):
        scale = random.choice(self.scales)
        if w < scale or h < scale:
          scale = min(w, h)

        x = random.randint(0, w - scale)
        y = random.randint(0, h - scale)
        patch = full_img.crop((x, y, x + scale, y + scale))

        if self._is_green_enough(patch):
          # 縮放到目標尺寸並轉為 Tensor
          patch = TF.resize(patch, [self.target_size, self.target_size])
          # 回傳: Tensor, Label, True(代表合格)
          return TF.to_tensor(patch), torch.tensor(label), True

      # 如果試了多次都失敗，就直接壓縮整張圖作為保底
      # 回傳: Tensor, Label, False(代表不合格，使用保底)
      fallback_img = TF.resize(full_img, [self.target_size, self.target_size])
      return TF.to_tensor(fallback_img), torch.tensor(label), False

  # def __getitem__(self, idx):
    # # 根據 idx 找到對應的大圖
    # img_path, label = self.all_samples[idx % len(self.all_samples)]

    # # Lazy Loading: 只有在需要時才打開圖
    # with Image.open(img_path) as full_img:
      # full_img = full_img.convert('RGB')
      # w, h = full_img.size

      # # 嘗試採樣，直到符合綠色閾值或達到上限
      # for _ in range(15):
        # scale = random.choice(self.scales)
        # if w < scale or h < scale:
          # scale = min(w, h)

        # x = random.randint(0, w - scale)
        # y = random.randint(0, h - scale)
        # patch = full_img.crop((x, y, x + scale, y + scale))

        # if self._is_green_enough(patch):
          # # 縮放到目標尺寸並轉為 Tensor
          # patch = TF.resize(patch, [self.target_size, self.target_size])
          # return TF.to_tensor(patch), torch.tensor(label)

      # # 如果試了多次都失敗，就直接壓縮整張圖作為保底
      # return TF.to_tensor(TF.resize(full_img, [self.target_size, self.target_size])), torch.tensor(label)
