import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset

# 共用的 Transform 設定
def _get_transforms():
  # 針對「從零訓練」設計的強效資料增強
  train_transform = transforms.Compose([
    transforms.Resize((256, 256)),               # 先放大一點點
    transforms.RandomCrop(224),                  # 再隨機裁切回 224
    transforms.RandomHorizontalFlip(p=0.5),      # 50% 機率水平翻轉
    transforms.RandomRotation(degrees=15),       # 隨機旋轉正負 15 度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 顏色與光照擾動
    transforms.ToTensor(),
    # 雖然是從零訓練，但標準化還是建議做，這組均值與標準差對 RGB 圖片依然適用
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  # 驗證集 / 測試集「絕對不可以」做隨機增強，只需 Resize 與標準化
  eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),               # 注意：這裡必須 Resize 回 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  return train_transform, eval_transform

def _build_combined_dataset(data_dirs, transform):
  """
  內部輔助函數：讀取多個目錄的資料並合併。
  """
  datasets_list = []
  for d_dir in data_dirs:
    datasets_list.append(datasets.ImageFolder(root=d_dir, transform=transform))

  # 使用 ConcatDataset 將多個 ImageFolder 合併
  return ConcatDataset(datasets_list)

def get_train_vaild_test_loader_from_dirs(data_dirs, batch_size=32, eval_batch_size=256, seed=42, split_ratio=(0.8, 0.1, 0.1)):
  """
  將多個資料目錄 (data_dirs) 內的資料合併後，切分為 Train / Valid / Test
  預設切割比例為 80% / 10% / 10%
  """
  assert sum(split_ratio) == 1.0, "切割比例的總和必須為 1.0"
  assert isinstance(data_dirs, list), "data_dirs 必須是一個列表 (list)"

  train_transform, eval_transform = _get_transforms()

  # 建立合併後的基礎資料集
  base_dataset_train = _build_combined_dataset(data_dirs, transform=train_transform)
  base_dataset_eval = _build_combined_dataset(data_dirs, transform=eval_transform)

  total_size = len(base_dataset_train)
  train_size = int(total_size * split_ratio[0])
  valid_size = int(total_size * split_ratio[1])
  # 測試集大小用相減的，確保不會因為浮點數小數點捨去而少漏掉圖片
  test_size = total_size - train_size - valid_size

  generator = torch.Generator().manual_seed(seed)
  train_sub_temp, valid_sub_temp, test_sub_temp = random_split(
      base_dataset_train, [train_size, valid_size, test_size], generator=generator
      )

  # 將對應的 indices 抽出來，並套用正確的 transform (Valid/Test 必須用 eval_transform)
  train_data = Subset(base_dataset_train, train_sub_temp.indices)
  valid_data = Subset(base_dataset_eval, valid_sub_temp.indices)
  test_data = Subset(base_dataset_eval, test_sub_temp.indices)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  valid_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)
  test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return train_loader, valid_loader, test_loader

def get_train_valid_loader_from_dirs(data_dirs, batch_size=32, eval_batch_size=256, seed=42, split_ratio=(0.9, 0.1)):
  """
  將多個資料目錄 (data_dirs) 內的資料合併後，切分為 Train / Valid
  預設切割比例為 90% / 10%
  """
  assert sum(split_ratio) == 1.0, "切割比例的總和必須為 1.0"
  assert isinstance(data_dirs, list), "data_dirs 必須是一個列表 (list)"

  train_transform, eval_transform = _get_transforms()

  base_dataset_train = _build_combined_dataset(data_dirs, transform=train_transform)
  base_dataset_eval = _build_combined_dataset(data_dirs, transform=eval_transform)

  total_size = len(base_dataset_train)
  train_size = int(total_size * split_ratio[0])
  valid_size = total_size - train_size

  generator = torch.Generator().manual_seed(seed)
  train_sub_temp, valid_sub_temp = random_split(
      base_dataset_train, [train_size, valid_size], generator=generator
      )

  train_data = Subset(base_dataset_train, train_sub_temp.indices)
  valid_data = Subset(base_dataset_eval, valid_sub_temp.indices)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  valid_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return train_loader, valid_loader

def get_test_loader_from_dirs(data_dirs, eval_batch_size=256):
  """
  將多個資料目錄 (data_dirs) 內的資料合併為單一 Test DataLoader
  """
  assert isinstance(data_dirs, list), "data_dirs 必須是一個列表 (list)"

  _, eval_transform = _get_transforms()

  test_data = _build_combined_dataset(data_dirs, transform=eval_transform)
  test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return test_loader
