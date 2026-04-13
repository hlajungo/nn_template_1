import bisect
import torch
import wandb
import copy
from torch.utils.data import Subset, ConcatDataset

def _get_image_path(dataset, idx):
  """遞迴解析 Dataset，找出該 index 圖片的真實檔案路徑"""
  if isinstance(dataset, Subset):
    # 如果是 Subset，就把當前的相對 index 轉換成原始的絕對 index
    return _get_image_path(dataset.dataset, dataset.indices[idx])

  elif isinstance(dataset, ConcatDataset):
    # 如果是 ConcatDataset，計算這個 index 屬於哪一個子 dataset
    dataset_idx = bisect.bisect_right(dataset.cumulative_sizes, idx)
    if dataset_idx == 0:
      sample_idx = idx
    else:
      sample_idx = idx - dataset.cumulative_sizes[dataset_idx - 1]
    return _get_image_path(dataset.datasets[dataset_idx], sample_idx)

  else:
    # 已經剝開到最底層的 ImageFolder
    return dataset.imgs[idx][0]


def train_model_with_early_stopping(
    model, train_loader, valid_loader, criterion, optimizer, device,
    epochs=100, patience=10, model_path='checkpoint/best_model.pth'
    ):
  """
  包含早停機制的訓練函數
  patience: 容忍多少個 Epoch 沒有進步
  """
  best_valid_loss = float('inf')
  best_model_wts = copy.deepcopy(model.state_dict()) # 用來暫存史上最強的權重
  early_stop_counter = 0

  print(f"開始訓練，最多執行 {epochs} Epochs，早停耐心值為 {patience}...")

  for epoch in range(epochs):
    # --- 1. 訓練階段 ---
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- 2. 驗證階段 ---
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_valid_loss = valid_loss / len(valid_loader)
    avg_valid_acc = 100 * correct / total

    # --- 3. 核心邏輯：判斷是否進步 ---
    # 我們通常監控 Valid Loss，因為 Loss 的波動比 Accuracy 更細膩
    if avg_valid_loss < best_valid_loss:
      best_valid_loss = avg_valid_loss
      best_model_wts = copy.deepcopy(model.state_dict()) # 刷新最強紀錄
      early_stop_counter = 0 # 重置耐心計數器

      # 物理儲存最佳模型權重
      torch.save(model.state_dict(), model_path)
      print(f"Epoch {epoch+1:03d}: 🌟 發現更低的 Valid Loss ({avg_valid_loss:.4f})，已更新權重！")
    else:
      early_stop_counter += 1
      print(f"Epoch {epoch+1:03d}: ⚠️  表現未提升 Vaild Loss ({avg_valid_loss:.4f})，已連續 {early_stop_counter}/{patience} 回合沒有進步。")

    # --- 4. 記錄指標到 W&B ---
    wandb.log({
      "epoch": epoch + 1,
      "train_loss": avg_train_loss,
      "valid_loss": avg_valid_loss,
      "valid_acc": avg_valid_acc,
      "patience_counter": early_stop_counter
      })

    # --- 5. 觸發早停 ---
    if early_stop_counter >= patience:
      print(f"\n🛑 [Early Stopping] 偵測到模型已停止進步，訓練提早結束！")
      break

  # 訓練結束後，將模型載回史上最強的那一版，而不是最後一版
  model.load_state_dict(best_model_wts)
  print(f"訓練結束。目前模型已載入 Best Valid Loss = {best_valid_loss:.4f} 的權重。")
  return model

def train_model(model, train_loader, criterion, optimizer, device, epochs):
  model.train()
  loss_history = []

  for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

    # 將每一輪的 Training Loss 傳送給 W&B，它會自動幫你畫成折線圖
    wandb.log({"Train Loss": epoch_loss, "Epoch": epoch+1})

  return loss_history

def evaluate_model(model, test_loader, device, class_names=None):
  model.eval()
  correct = 0
  total = 0
  all_true_labels = []
  all_pred_labels = []

  # 1. 抓出所有測試圖片的真實檔案路徑
  dataset = test_loader.dataset

  # 使用支援 Subset, ConcatDataset 與單純 ImageFolder 的萬用取路徑法
  paths = [_get_image_path(dataset, i) for i in range(len(dataset))]

  misclassified_info = [] # 用來存錯誤圖片的清單
  current_idx = 0 # 追蹤目前跑到第幾張照片

  with torch.no_grad():
    for inputs, labels in test_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)

      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      all_true_labels.extend(labels.cpu().numpy())
      all_pred_labels.extend(predicted.cpu().numpy())

      # 2. 揪出預測錯誤的圖片
      for i in range(len(labels)):
        if predicted[i] != labels[i]:
          filepath = paths[current_idx + i]
          true_class = class_names[labels[i].item()] if class_names else str(labels[i].item())
          pred_class = class_names[predicted[i].item()] if class_names else str(predicted[i].item())

          # 記錄起來
          misclassified_info.append({
            'path': filepath,
            'true': true_class,
            'pred': pred_class
            })
      current_idx += len(labels)

  accuracy = 100 * correct / total
  print(f"Test Accuracy: {accuracy:.2f}%")

  # 多回傳一個 misclassified_info
  return accuracy, all_true_labels, all_pred_labels, misclassified_info
