import pandas as pd
import numpy as np
import onnxruntime as ort
from config.MyConfig import MyConfig
from src.dataset import get_kaggle_test_loader

def main():
  # 載入設定
  config = MyConfig()

  print(f"載入 ONNX 模型: {config.onnx_path}")
  # 啟動 ONNX 推論會話 (可加入 providers=['CUDAExecutionProvider'] 開啟 GPU)
  ort_session = ort.InferenceSession(config.onnx_path, providers=['CUDAExecutionProvider'])

  print(f"讀取 Kaggle 測試集: {config.test_csv_path}")
  kaggle_test_loader = get_kaggle_test_loader(
      csv_path=config.test_csv_path,
      batch_size=config.test_batch_size
      )

  predictions = []

  # 執行推論
  for images in kaggle_test_loader:
    # ONNX Runtime 預期接收 NumPy Array 而非 PyTorch Tensor
    x_numpy = images.numpy()

    # 執行預測，'input' 必須對應匯出時的 input_names
    ort_inputs = {'input': x_numpy}
    ort_outputs = ort_session.run(None, ort_inputs)

    # 取得預測機率的 logits 並抓出最大值索引
    predicted_logits = ort_outputs[0]
    preds = np.argmax(predicted_logits, axis=1)
    predictions.extend(preds.tolist())

  print(f"成功預測 {len(predictions)} 筆資料！")

  # 整理為 Kaggle 要求的格式並存檔
  submission_df = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
    })
  submission_df.to_csv(config.submission_path, index=False)
  print(f"{config.submission_path} 已生成！")

if __name__ == '__main__':
  main()
