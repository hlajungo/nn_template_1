import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
from config.MyConfig import MyConfig
from src.utils import get_device

class KaggleTTADataset(Dataset):
  def __init__(self, path):
    self.path = path
    self.mode = 'folder'
    
    if os.path.isdir(path):
      self.image_files = [
        os.path.join(path, f) for f in os.listdir(path) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
      ]
    else:
      raise ValueError(f"TTA 腳本目前僅支援圖片資料夾路徑: {path}")

    # 定義三種不同的 TTA 視角
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 1. 標準視角
    self.t1 = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      normalize
    ])
    
    # 2. 鏡像視角 (放大一點點後水平翻轉)
    self.t2 = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.CenterCrop(224),
      transforms.RandomHorizontalFlip(p=1.0), # 強制翻轉
      transforms.ToTensor(),
      normalize
    ])

    # 3. 隨機裁切視角
    self.t3 = transforms.Compose([
      transforms.Resize((256, 256)),
      transforms.RandomCrop(224),
      transforms.ToTensor(),
      normalize
    ])

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_path = self.image_files[idx]
    image = Image.open(img_path).convert("RGB")
    
    # 產生三種變體
    tensor1 = self.t1(image)
    tensor2 = self.t2(image)
    tensor3 = self.t3(image)
    
    # 將三張圖疊合成形狀 (3, C, H, W) 的 Tensor
    stacked_tensors = torch.stack([tensor1, tensor2, tensor3])
    file_id = os.path.splitext(os.path.basename(img_path))[0]
    
    return stacked_tensors, file_id

def get_kaggle_loader(path, batch_size, num_workers=4):
  dataset = KaggleTTADataset(path=path)
  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
  )

def main():
  config = MyConfig()
  device = get_device()
  print(f"使用運算設備: {device}")
  
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

  # 確保這是你最終查出的正確 mapping
  idx_mapping = {
    0: 0,
    1: 2,
    2: 4,
    3: 5,
    4: 1,
    5: 3
  }

  all_indices = []
  all_ids = []

  for path in config.predict_dirs:
    print(f"\n正在處理路徑: {path} (啟用 3x TTA 擴增)")
    # 因為每張圖會變成 3 張，為了避免 VRAM 爆掉，我們將 Batch Size 減半
    safe_batch_size = max(1, config.test_batch_size // 2)
    loader = get_kaggle_loader(path=path, batch_size=safe_batch_size)

    path_indices = []
    path_ids = []
    
    with torch.no_grad():
      for images, ids in loader:
        # images shape: (Batch_Size, 3_Crops, 3_Channels, 224, 224)
        batch_size, n_crops, c, h, w = images.size()
        
        # 攤平以便送入模型: (Batch_Size * 3, 3, 224, 224)
        images = images.view(-1, c, h, w).to(device)
        
        # 模型推論
        outputs = model(images)
        
        # 將輸出重組回: (Batch_Size, 3_Crops, Num_Classes)
        outputs = outputs.view(batch_size, n_crops, -1)
        
        # 核心 TTA 邏輯：將 3 種視角的機率 logits 加總平均
        avg_outputs = torch.mean(outputs, dim=1)
        
        # 針對平均後的結果取最大值
        _, preds = torch.max(avg_outputs, 1)
        
        raw_preds = preds.cpu().tolist()
        mapped_preds = [idx_mapping[p] for p in raw_preds]
        
        path_indices.extend(mapped_preds)
        path_ids.extend(ids)

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

  tta_path = config.submission_path.replace('.csv', '_TTA.csv')
  submission_df.to_csv(tta_path, index=False)
  print(f"結果已儲存至: {tta_path}")

if __name__ == '__main__':
  main()
