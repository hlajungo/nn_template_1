from config.BaseConfig import BaseConfig
from dataclasses import dataclass, field
from typing import List
import string
import os

@dataclass
class MyConfig(BaseConfig):
  """存放真正的超參數，覆寫 BaseConfig 的屬性"""

  # --- 自動定位專案根目錄 ---
  # 取得本檔案 (MyConfig.py) 的絕對路徑，假設它在 ./config/ 下
  # 所以它的上層 (..) 就是專案根目錄
  _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

  model_type: str = 'convnext_tiny'
  batch_size: int = 64
  test_batch_size: int = 64
  epochs: int = 9999
  learning_rate: float = 3e-4
  patience = 20

  # --- 圖片維度資訊 ---
  img_channels: int = 3
  img_size: int = 224

  class_names: List[str] = field(default_factory=lambda: [
    "台灣欒樹", "羊蹄甲", "松", "苦楝", "桂花", "楓",
    ])

  # --- 資料集路徑 (使用 _BASE_DIR 組合絕對路徑) ---

  raw_data_dirs: List[str] = field(default_factory=lambda: [
    os.path.join(MyConfig._BASE_DIR, 'data', 'raw')
    ])

  # base_train_data_dir: str = s.path.join(MyConfig._BASE_DIR, 'data')
  train_data_dirs: List[str] = field(default_factory=lambda: [
    os.path.join(MyConfig._BASE_DIR, 'data', 'filtered_data'),
    ])

  test_dirs: List[str] = field(default_factory=lambda: [
    # 保持空列表
    ])

  predict_dirs: List[str] = field(default_factory=lambda: [
    os.path.join(MyConfig._BASE_DIR, 'data', 'Datasets')
    ])

  submission_path: str = field(default_factory=lambda:
                               os.path.join(MyConfig._BASE_DIR, "Submission.csv")
                               )

  @property
  def num_classes(self) -> int:
    """動態取得類別數量"""
    return len(self.class_names)
