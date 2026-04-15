import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from config.MyConfig import MyConfig
from src.builder import build_model 
from src.utils import get_device

class KaggleUnifiedDataset(Dataset):
  def __init__(self, path, transform=None):
    self.path = path
    self.transform = transform or transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])

    if os.path.isfile(path) and path.endswith('.csv'):
      self.mode = 'csv'
      self.data_df = pd.read_csv(path)
    elif os.path.isdir(path):
      self.mode = 'folder'
      self.image_files = [
          os.path.join(path, f) for f in os.listdir(path) 
          if f.lower().endswith(('.png', '.jpg', '.jpeg'))
          ]
      print(f"  [診斷] 目錄 {path} 找到 {len(self.image_files)} 張圖片。")
      if len(self.image_files) > 0:
        print(f"  [診斷] 前 3 個讀取到的檔案: {[os.path.basename(f) for f in self.image_files[:3]]}")
    else:
      raise ValueError(f"不支援的路徑類型: {path}")

  def __len__(self):
    return len(self.data_df) if self.mode == 'csv' else len(self.image_files)

  def __getitem__(self, idx):
    if self.mode == 'csv':
      row = self.data_df.iloc[idx]
      features = row.values[1:].astype(np.float32)
      x_tensor = torch.tensor(features).view(1, 28, 28) 
      file_id = str(row.values[0])
      return x_tensor, file_id
    else:
      img_path = self.image_files[idx]
      image = Image.open(img_path).convert("RGB")
      tensor = self.transform(image)
      file_id = os.path.splitext(os.path.basename(img_path))[0]
      return tensor, file_id

def get_kaggle_loader(path, batch_size, num_workers=4):
  dataset = KaggleUnifiedDataset(path=path)
  return DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
      )

def main():
  config = MyConfig()

  config.model_type = 'convnext_tiny'

  device = get_device()
  print(f"使用運算設備: {device}")

  print("\n--- [診斷] 類別設定檢查 ---")
  print("目前 config.class_names 假設的順序為：")
  for i, name in enumerate(config.class_names):
    print(f"  Index {i} -> {name}")
  print("強烈建議：請確認這與訓練時 torchvision ImageFolder 產生的 class_to_idx 完全一致！\n")

  print("繞過 builder，直接建立模型架構 (ConvNeXt-Tiny)...")
  model = models.convnext_tiny(weights=None)

  num_ftrs = model.classifier[2].in_features
  model.classifier[2] = nn.Sequential(
      nn.Dropout(p=0.6),
      nn.Linear(num_ftrs, config.num_classes)
      )

  model = model.to(device)

  weight_path = "/media/hlajungo/D/linux/repo_my/1142_nn/1142_nn_mid/checkpoint/best_model_0415.pth"
  print(f"載入 PyTorch 權重: {weight_path}")

  model.load_state_dict(torch.load(weight_path, map_location=device))
  model.eval()

  all_indices = []
  all_ids = []

  # 建立從 PyTorch 模型索引，對應回 Kaggle 官方索引的翻譯字典
  # 格式: {PyTorch_Index: Kaggle_Index}
  idx_mapping = {
      0: 0,  # 台灣欒樹 -> 台灣欒樹
      1: 2,  # 松 -> 松
      2: 4,  # 桂花 -> 桂花
      3: 5,  # 楓 -> 楓
      4: 1,  # 羊蹄甲 -> 羊蹄甲
      5: 3   # 苦楝 -> 苦楝
      }

  for path in config.predict_dirs:
    print(f"\n正在處理路徑: {path}")
    loader = get_kaggle_loader(
        path=path,
        batch_size=config.test_batch_size
        )

    path_indices = []
    path_ids = []

    batch_count = 0
    with torch.no_grad():
      for images, ids in loader:
        images = images.to(device)
        outputs = model(images)

        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        # 將 PyTorch 的預測索引轉換為 Kaggle 的索引
        raw_preds = preds.cpu().tolist()
        mapped_preds = [idx_mapping[p] for p in raw_preds]

        if batch_count == 0:
          print("\n=== 第一批次 (First Batch) 深度診斷 ===")
          print(f"Tensor Shape: {images.shape}")
          print(f"Tensor 數值範圍: min={images.min().item():.3f}, max={images.max().item():.3f}, mean={images.mean().item():.3f}")
          print("前 3 筆影像的預測結果細節：")

          limit = min(3, len(ids))
          for i in range(limit):
            p_str = ", ".join([f"{p:.3f}" for p in probs[i].cpu().tolist()])
            pred_idx = raw_preds[i]
            mapped_idx = mapped_preds[i]
            # 這裡的 pred_name 只是基於 PyTorch 的 index 抓字串，方便你核對
            pred_name = config.class_names[mapped_idx] if mapped_idx < len(config.class_names) else "Unknown"

            print(f"  [影像 ID: {ids[i]}]")
            print(f"    各類別機率: [{p_str}]")
            print(f"    PyTorch 原始輸出: {pred_idx} -> 轉換為 Kaggle Index: {mapped_idx} ({pred_name})")
          print("========================================\n")

        path_indices.extend(mapped_preds)
        path_ids.extend(ids)
        batch_count += 1

    print(f"路徑 {path} 預測完成，共 {len(path_indices)} 筆資料。")
    all_indices.extend(path_indices)
    all_ids.extend(path_ids)

  print(f"\n全部預測完成！總計 {len(all_indices)} 筆資料。")

  submission_df = pd.DataFrame({
    'ID': all_ids,
    'Target_Index': all_indices
    })

  try:
    submission_df['ID'] = submission_df['ID'].astype(int)
    submission_df = submission_df.sort_values('ID')
  except ValueError:
    pass 

  submission_df.to_csv(config.submission_path, index=False)
  print(f"結果已儲存至: {config.submission_path} (附帶轉換字典版本)")

if __name__ == '__main__':
  main()
