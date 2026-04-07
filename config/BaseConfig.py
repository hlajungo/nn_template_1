from dataclasses import dataclass

@dataclass
class BaseConfig:
  """定義設定檔的結構與預設值，不包含具體實驗的超參數"""
  # General
  seed: int = 42
  device: str = 'cuda'

  # Data
  test_csv_path: str = './data/process/test.csv'
  batch_size: int = 32
  test_batch_size: int = 256

  # Base Model & Training
  model_type: str = 'CNN'
  learning_rate: float = 1e-3
  epochs: int = 5

  # Base Outputs
  model_weight_path: str = 'checkpoint/mnist_latest.pth'
  onnx_path: str = 'checkpoint/mnist_cnn.onnx'
  submission_path: str = 'submission.csv'
