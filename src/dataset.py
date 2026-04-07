import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# 共用的 Transform 設定
def _get_transforms():
  train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    # Mean and Standard(標準差) from ImageNet, from [0, 1] -> [-2, 2]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return train_transform, eval_transform

def get_train_vaild_test_loader_from_dir(data_dir, batch_size=32, eval_batch_size=256, seed=42, split_ratio=(0.8, 0.1, 0.1)):
  """
  將 data_dir 內的資料切分為 Train / Valid / Test
  預設切割比例為 80% / 10% / 10%
  """
  assert sum(split_ratio) == 1.0, "切割比例的總和必須為 1.0"

  train_transform, eval_transform = _get_transforms()

  base_dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
  base_dataset_eval = datasets.ImageFolder(root=data_dir, transform=eval_transform)

  total_size = len(base_dataset_train)
  train_size = int(total_size * split_ratio[0])
  valid_size = int(total_size * split_ratio[1])
  # 測試集大小用相減的，確保不會因為浮點數小數點捨去而少漏掉圖片
  test_size = total_size - train_size - valid_size

  generator = torch.Generator().manual_seed(seed)
  train_sub_temp, valid_sub_temp, test_sub_temp = random_split(
    base_dataset_train, [train_size, valid_size, test_size], generator=generator
  )

  train_data = Subset(base_dataset_train, train_sub_temp.indices)
  valid_data = Subset(base_dataset_eval, valid_sub_temp.indices)
  test_data = Subset(base_dataset_eval, test_sub_temp.indices)

  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  valid_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)
  test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return train_loader, valid_loader, test_loader

def get_train_valid_loader_from_dir(data_dir, batch_size=32, eval_batch_size=256, seed=42, split_ratio=(0.9, 0.1)):
  """
  將 data_dir 內的資料切分為 Train / Valid
  預設切割比例為 90% / 10%
  """
  assert sum(split_ratio) == 1.0, "切割比例的總和必須為 1.0"

  train_transform, eval_transform = _get_transforms()

  base_dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
  base_dataset_eval = datasets.ImageFolder(root=data_dir, transform=eval_transform)

  total_size = len(base_dataset_train)
  train_size = int(total_size * split_ratio[0])
  # 驗證集大小同樣用相減的確保精確
  valid_size = total_size - train_size

  generator = torch.Generator().manual_seed(seed)
  train_sub_temp, valid_sub_temp = random_split(
    base_dataset_train, [train_size, valid_size], generator=generator
  )

  train_data = Subset(base_dataset_train, train_sub_temp.indices)
  valid_data = Subset(base_dataset_eval, valid_sub_temp.indices)

  # 建立 DataLoaders
  train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  valid_loader = DataLoader(valid_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return train_loader, valid_loader

def get_test_loader_from_dir(data_dir, eval_batch_size=256):
  """
  將 data_dir 內的資料變為 Test
  """
  _, eval_transform = _get_transforms()
  test_data = datasets.ImageFolder(root=data_dir, transform=eval_transform)
  test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=False, num_workers=2)

  return test_loader
