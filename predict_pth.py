import torch
import torch.nn as nn
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
  
  # 強制覆寫模型設定，確保 build_model 吐出正確的架構
  config.model_type = 'convnext_tiny'
  
  device = get_device()
  print(f"使用運算設備: {device}")
  print("建立模型架構 (ConvNeXt-Tiny)...")
  
  model = build_model(config, device)
  
  # 鎖定你剛剛找到的救命權重
  weight_path = "/media/hlajungo/D/linux/repo_my/1142_nn/1142_nn_mid/checkpoint/best_model.pth"
  print(f"載入 PyTorch 權重: {weight_path}")
  
  model.load_state_dict(torch.load(weight_path, map_location=device))
  model.eval()

  all_indices = []
  all_ids = []

  for path in config.predict_dirs:
    print(f"正在處理路徑: {path}")
    loader = get_kaggle_loader(
      path=path,
      batch_size=config.test_batch_size
    )

    path_indices = []
    path_ids = []
    
    with torch.no_grad():
      for images, ids in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        path_indices.extend(preds.cpu().tolist())
        path_ids.extend(ids)

    print(f"路徑 {path} 預測完成，共 {len(path_indices)} 筆資料。")
    all_indices.extend(path_indices)
    all_ids.extend(path_ids)

  print(f"全部預測完成！總計 {len(all_indices)} 筆資料。")

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
  print(f"結果已儲存至: {config.submission_path} (原生 PyTorch 版本)")

if __name__ == '__main__':
  main()
