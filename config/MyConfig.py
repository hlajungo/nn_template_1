from config.BaseConfig import BaseConfig
from dataclasses import dataclass, field
from typing import List

@dataclass
class MyConfig(BaseConfig):
  """存放真正的超參數，覆寫 BaseConfig 的屬性"""
  model_type: str = 'PlantCNN_v2'  # ['Perceptron', 'CNN', 'PlantCNN']
  batch_size: int = 64
  test_batch_size: int = 64
  # 從頭開始訓練通常需要多一點 Epoch，這裡先幫你預設調高到 50，可依訓練狀況增減
  epochs: int = 9999
  learning_rate: float = 1e-4
  patience = 10

  # --- 新增：圖片維度資訊 ---
  img_channels: int = 3
  img_size: int = 224

  # --- 資料集路徑 ---
  # 因為我們使用動態切分，只需要指定父目錄即可
  class_names: List[str] = field(default_factory=lambda: [
    "台灣欒樹", "松", "桂花", "楓", "羊蹄甲", "苦楝"
  ])
  data_dir: str = './data/process'
  test_dir: str = './data/ext_test/process'
