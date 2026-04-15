import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime as ort
from config.MyConfig import MyConfig

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
      # 刪除 self.image_files.sort()，我們不再依賴陣列順序
    else:
      raise ValueError(f"不支援的路徑類型: {path}")

  def __len__(self):
    return len(self.data_df) if self.mode == 'csv' else len(self.image_files)

  def __getitem__(self, idx):
    if self.mode == 'csv':
      row = self.data_df.iloc[idx]
      features = row.values[1:].astype(np.float32)
      x_tensor = torch.tensor(features).view(1, 28, 28) 
      # 抓取 CSV 第一欄作為 ID
      file_id = str(row.values[0])
      return x_tensor, file_id
    else:
      img_path = self.image_files[idx]
      image = Image.open(img_path).convert("RGB")
      tensor = self.transform(image)
      # 核心修復：直接從檔名切出 ID (例如 "12.jpg" -> "12")
      file_id = os.path.splitext(os.path.basename(img_path))[0]
      return tensor, file_id

def get_kaggle_loader(path, batch_size, num_workers=0):
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

  print(f"載入 ONNX 模型: {config.onnx_path}")
  ort_session = ort.InferenceSession(
    "/media/hlajungo/D/linux/repo_my/1142_nn/1142_nn_mid/checkpoint/mnist_cnn.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
  )

  all_indices = []
  all_ids = [] # 新增：用來儲存真正的檔案 ID

  for path in config.predict_dirs:
    print(f"正在處理路徑: {path}")
    loader = get_kaggle_loader(
      path=path,
      batch_size=config.test_batch_size
    )

    path_indices = []
    path_ids = []
    
    # DataLoader 現在會同時吐出影像與對應的 ID
    for images, ids in loader:
      x_numpy = images.numpy()
      ort_inputs = {'input': x_numpy}
      ort_outputs = ort_session.run(None, ort_inputs)

      predicted_logits = ort_outputs[0]
      preds = np.argmax(predicted_logits, axis=1)
      
      path_indices.extend(preds.tolist())
      path_ids.extend(ids) # 把真正的 ID 存起來

    print(f"路徑 {path} 預測完成，共 {len(path_indices)} 筆資料。")
    all_indices.extend(path_indices)
    all_ids.extend(path_ids)

  print(f"全部預測完成！總計 {len(all_indices)} 筆資料。")

  # 將真正的 ID 與預測結果對齊寫入
  submission_df = pd.DataFrame({
    'ID': all_ids,
    'Target_Index': all_indices
  })
  
  # 友善的格式化：將 ID 轉回數字並照順序排好，方便你在 Kaggle 上檢查
  try:
    submission_df['ID'] = submission_df['ID'].astype(int)
    submission_df = submission_df.sort_values('ID')
  except ValueError:
    pass 

  submission_df.to_csv(config.submission_path, index=False)
  print(f"結果已儲存至: {config.submission_path}")

if __name__ == '__main__':
  main()
