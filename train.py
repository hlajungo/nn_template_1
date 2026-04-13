import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from config.MyConfig import MyConfig

# 確保從 src.dataset 引入新的 API
from src.dataset import (
    get_train_valid_loader_from_dirs, 
    get_test_loader_from_dirs,
    get_train_vaild_test_loader_from_dirs
    )
from src.builder import build_model
from src.engine import train_model_with_early_stopping, evaluate_model
from src.utils import get_device, set_all_seeds
from sklearn.metrics import confusion_matrix, classification_report

class PlantTrainer:
  """負責管理整個模型訓練、評估與匯出生命週期的類別"""

  def __init__(self, config: MyConfig):
    self.config = config

    # 1. 基礎環境設定
    set_all_seeds(self.config.seed)
    self.device = get_device()
    print(f"目前使用的運算設備: {self.device}")

    # 2. 準備類別成員變數
    self.model = None
    self.train_loader = None
    self.valid_loader = None
    self.test_loader = None
    self.criterion = None
    self.optimizer = None

  def setup_wandb(self):
    """初始化 Weights & Biases 監控"""
    wandb.init(
        project="plant-classification",
        name=f"CNN_{self.config.batch_size}_{self.config.learning_rate}",
        config=vars(self.config)
        )

  def prepare_data(self):
    """根據 test_dirs 是否為空，動態決定切分策略"""
    print("\n--- 正在準備資料集 ---")

    if self.config.test_dirs:
      print("偵測到獨立測試集，將 train_dirs 切分 (90% Train / 10% Valid)")
      self.train_loader, self.valid_loader = get_train_valid_loader_from_dirs(
          data_dirs=self.config.train_dirs,
          batch_size=self.config.batch_size,
          eval_batch_size=self.config.test_batch_size,
          seed=self.config.seed,
          split_ratio=(0.9, 0.1)
          )

      self.test_loader = get_test_loader_from_dirs(
          data_dirs=self.config.test_dirs,
          eval_batch_size=self.config.test_batch_size
          )
    else:
      print("未偵測到獨立測試集，將 train_dirs 切分 (80% Train / 10% Valid / 10% Test)")
      self.train_loader, self.valid_loader, self.test_loader = get_train_vaild_test_loader_from_dirs(
          data_dirs=self.config.train_dirs,
          batch_size=self.config.batch_size,
          eval_batch_size=self.config.test_batch_size,
          seed=self.config.seed,
          split_ratio=(0.8, 0.1, 0.1)
          )

  def build_system(self):
    """建立神經網路、損失函數與優化器"""
    self.model = build_model(self.config, self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(
        self.model.parameters(),
        lr=self.config.learning_rate,
        weight_decay=1e-4
        )

    params = sum(p.numel() for p in self.model.parameters())
    print(f"模型總參數數量: {params:,}")

  def train(self):
    """執行包含早停機制的訓練迴圈"""
    print("\n--- 開始訓練模型 ---")
    train_model_with_early_stopping(
        model=self.model,
        train_loader=self.train_loader,
        valid_loader=self.valid_loader,
        criterion=self.criterion,
        optimizer=self.optimizer,
        device=self.device,
        epochs=self.config.epochs,
        patience=self.config.patience
        )

  def evaluate_and_log(self):
    """使用外部測試集評估模型，並上傳報表至 W&B"""
    # 確保 test_loader 存在，避免發生空指標錯誤
    if not self.test_loader:
      print("沒有可用的測試集，跳過最終評估。")
      return

    print("\n--- 執行最終外部測試集評估 ---")
    accuracy, true_labels, pred_labels, misclassified_info = evaluate_model(
        self.model, self.test_loader, self.device, self.config.class_names
        )

    self._print_detailed_metrics(true_labels, pred_labels)

    error_table = wandb.Table(columns=["預覽圖", "檔案路徑", "正確答案", "模型猜測"])
    print("\n=== 預測錯誤的圖片清單 ===")
    count = 0
    for item in misclassified_info:
      if count == 20:
        print(f"... (Total {len(misclassified_info)})")
        break
      filename = item['path'].split('/')[-1]
      print(f"檔案: {filename} | 正確: {item['true']:<4} -> 錯認為: {item['pred']}")
      count += 1

    for item in misclassified_info:
      error_table.add_data(
          wandb.Image(item['path']), item['path'], item['true'], item['pred']
          )

    wandb.log({
      "Test Accuracy (Ext)": accuracy,
      "Misclassified Images": error_table,
      "Confusion Matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=true_labels,
        preds=pred_labels,
        class_names=self.config.class_names
        )
      })

  def export_models(self):
    """匯出 PyTorch 權重與 ONNX 模型"""
    print("\n--- 儲存與匯出模型 ---")
    os.makedirs(os.path.dirname(self.config.model_weight_path), exist_ok=True)

    torch.save(self.model.state_dict(), self.config.model_weight_path)
    print(f"模型權重已儲存至: {self.config.model_weight_path}")

    self.model.eval()
    dummy_input = torch.randn(
        1, self.config.img_channels, self.config.img_size, self.config.img_size
        ).to(self.device)

    torch.onnx.export(
        self.model, dummy_input, self.config.onnx_path,
        export_params=True, opset_version=18, do_constant_folding=True,
        input_names=['input'], output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
    print(f"ONNX 模型已成功匯出至: {self.config.onnx_path}")

  def _print_detailed_metrics(self, true_labels, pred_labels):
    """(私有方法) 印出詳細的分類指標"""
    print("\n=== 分類報告 (Classification Report) ===")
    print(classification_report(true_labels, pred_labels, target_names=self.config.class_names))

    print("=== 各類別的 TP / FP / TN / FN ===")
    cm = confusion_matrix(true_labels, pred_labels)
    for i, name in enumerate(self.config.class_names):
      TP = cm[i, i]
      FP = cm[:, i].sum() - TP
      FN = cm[i, :].sum() - TP
      TN = cm.sum() - (TP + FP + FN)
      print(f"[{name}]\n  TP: {TP}\n  FP: {FP}\n  FN: {FN}\n  TN: {TN}")
      print("-" * 30)

  def run(self):
    """執行完整的生命週期"""
    self.setup_wandb()
    self.prepare_data()
    self.build_system()
    self.train()
    self.evaluate_and_log()
    self.export_models()
    wandb.finish()


def main():
  config = MyConfig()
  trainer = PlantTrainer(config)
  trainer.run()

if __name__ == '__main__':
  main()
