import torch
import os
from config.MyConfig import MyConfig
from src.builder import build_model
from src.utils import get_device

def main():
  config = MyConfig()
  device = get_device()
  
  print("建立模型架構...")
  model = build_model(config, device)
  
  # 強制指定我們剛剛找到的救命權重
  weight_path = os.path.join(MyConfig._BASE_DIR, 'checkpoint', 'best_model.pth')
  print(f"載入最強權重: {weight_path}")
  
  # 載入權重並切換到評估模式
  model.load_state_dict(torch.load(weight_path, map_location=device))
  model.eval()

  print(f"開始匯出 ONNX 模型至: {config.onnx_path}")
  # 準備一組假的輸入讓 ONNX 追蹤計算圖
  dummy_input = torch.randn(
    1, config.img_channels, config.img_size, config.img_size
  ).to(device)

  # 執行匯出
  torch.onnx.export(
    model, dummy_input, config.onnx_path,
    export_params=True, 
    opset_version=18, 
    do_constant_folding=True,
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
  )
  print("ONNX 匯出成功！你可以準備跑 Kaggle 預測了。")

if __name__ == '__main__':
  main()
