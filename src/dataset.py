import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset, WeightedRandomSampler

# 共用的 Transform 設定
def _get_transforms():
  train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

  return train_transform, eval_transform

def _build_combined_dataset(data_dirs, transform):
  datasets_list = []
  for d_dir in data_dirs:
    datasets_list.append(datasets.ImageFolder(root=d_dir, transform=transform))
  return ConcatDataset(datasets_list)

def _get_global_targets(concat_dataset):
  """輔助函數：從 ConcatDataset 中提取所有 ImageFolder 的原始標籤"""
  targets = []
  for ds in concat_dataset.datasets:
    targets.extend(ds.targets)
  return targets

def _create_balanced_sampler(global_targets, indices):
  """輔助函數：根據 Subset 的索引，計算類別權重並建立 Sampler"""
  # 取出該 Subset 對應的標籤
  subset_targets = [global_targets[i] for i in indices]
  subset_targets_tensor = torch.tensor(subset_targets)

  # 計算各類別數量
  class_counts = torch.bincount(subset_targets_tensor)

  # 計算權重：數量越少，權重越高 (加上 1e-6 避免除以零)
  class_weights = 1.0 / (class_counts.float() + 1e-6)

  # 將權重映射回每一個樣本
  sample_weights = class_weights[subset_targets_tensor]

  # 建立並回傳 Sampler
  sampler = WeightedRandomSampler(
      weights=sample_weights,
      num_samples=len(sample_weights),
      replacement=True
      )
  return sampler

def get_train_vaild_test_loader_from_dirs(data_dirs, batch_size=32, eval_batch_size=256, seed=42, split_ratio=(0.8, 0.1, 0.1)):
  """
  將多個資料目錄合併後，切分為 Train / Valid / Test (預設 80/10/10)
  並為 Train 啟用類別平衡抽樣 (Balanced Sampling)
  """
  assert sum(split_ratio) == 1.0, "切割比例的總和必須為 1.0"
  assert isinstance(data_dirs, list), "data_dirs 必須是一個列表"

  train_transform, eval_transform = _get_transforms()

  base_dataset_train = _build_combined_dataset(data_dirs, transform=train_transform)
  base_dataset_eval = _build_combined_dataset(data_dirs, transform=eval_transform)

  # 提取全局標籤，供後續計算抽樣權重使用
  global_targets = _get_global_targets(base_dataset_train)

  total_size = len(base_dataset_train)
  train_size = int(total_size * split_ratio[0])
  valid_size = int(total_size * split_ratio[1])
  test_size = total_size - train_size - valid_size

  generator = torch.Generator().manual_seed(seed)
  train_sub_temp, valid_sub_temp, test_sub_temp = random_split(
      base_dataset_train, [train_size, valid_size, test_size], generator=generator
      )

  train_data = Subset(base_dataset_train, train_sub_temp.indices)
  valid_data = Subset(base_dataset_eval, valid_sub_temp.indices)
  test_data = Subset(base_dataset_eval, test_sub_temp.indices)

  # 為訓練集建立平衡抽樣器
  train_sampler = _create_balanced_sampler(global_targets, train_sub_temp.indices)

  # 注意：啟用 sampler 時，shuffle 必須強制為 False
  train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=4)
  valid_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=4)
  test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=4)

  return train_loader, valid_loader, test_loader

def get_train_valid_loader_from_dirs(data_dirs, batch_size=32, eval_batch_size=256, seed=42, split_ratio=(0.9, 0.1)):
  """
  雙切分版本 (預設 90/10)，同樣為 Train 啟用平衡抽樣
  """
  assert sum(split_ratio) == 1.0, "切割比例的總和必須為 1.0"
  assert isinstance(data_dirs, list), "data_dirs 必須是一個列表"

  train_transform, eval_transform = _get_transforms()

  base_dataset_train = _build_combined_dataset(data_dirs, transform=train_transform)
  base_dataset_eval = _build_combined_dataset(data_dirs, transform=eval_transform)

  global_targets = _get_global_targets(base_dataset_train)

  total_size = len(base_dataset_train)
  train_size = int(total_size * split_ratio[0])
  valid_size = total_size - train_size

  generator = torch.Generator().manual_seed(seed)
  train_sub_temp, valid_sub_temp = random_split(
      base_dataset_train, [train_size, valid_size], generator=generator
      )

  train_data = Subset(base_dataset_train, train_sub_temp.indices)
  valid_data = Subset(base_dataset_eval, valid_sub_temp.indices)

  train_sampler = _create_balanced_sampler(global_targets, train_sub_temp.indices)

  train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=2)
  valid_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return train_loader, valid_loader

def get_test_loader_from_dirs(data_dirs, eval_batch_size=256):
  assert isinstance(data_dirs, list), "data_dirs 必須是一個列表"
  _, eval_transform = _get_transforms()
  test_data = _build_combined_dataset(data_dirs, transform=eval_transform)
  test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)
  return test_loader
