import os
import sys
from PIL import Image
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
  sys.path.insert(0, str(root_path))
from config.MyConfig import MyConfig


def prepare_data(data_dir, target_size, dest_dir, log_file="processed_log.txt"):
  """
  將大圖按多尺度網格切割，支援增量處理(Incremental Processing)。
  """
  # 1. 讀取已處理過的原圖清單
  processed_registry = set()
  log_path = os.path.join(dest_dir, log_file)

  if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
      processed_registry = set(line.strip() for line in f)

  newly_processed_count = 0

  for root, _, files in os.walk(data_dir):
    print(f"Working on {os.path.basename(root)}")
    for file in files:
      if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

      # 取得原始檔案路徑與相對路徑
      src_path = os.path.join(root, file)
      rel_path = os.path.relpath(root, data_dir)

      # 2. 檢查此原圖是否已經處理過 (使用 相對路徑+檔名 作為唯一識別)
      file_identifier = os.path.join(rel_path, file)
      if file_identifier in processed_registry:
        continue  # 已經處理過，直接跳過！

      # 建立目標目錄
      target_dir = os.path.join(dest_dir, rel_path)
      os.makedirs(target_dir, exist_ok=True)

      filename_no_ext = os.path.splitext(file)[0]
      ext = os.path.splitext(file)[1]

      try:
        with Image.open(src_path) as img:
          img = img.convert('RGB')
          w, h = img.size

          # --- 步驟 1: 原圖裁剪成方形後壓縮 ---
          min_dim = min(w, h)
          left = (w - min_dim) // 2
          top = (h - min_dim) // 2
          sq_img = img.crop((left, top, left + min_dim, top + min_dim))
          sq_img = sq_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

          sq_name = f"{filename_no_ext}_square{ext}"
          sq_img.save(os.path.join(target_dir, sq_name))

          # --- 步驟 2~4: 多尺度網格切割 (邊緣捨棄) ---
          scale = target_size
          while scale <= min(w, h):
            for x in range(0, w - scale + 1, scale):
              for y in range(0, h - scale + 1, scale):
                patch = img.crop((x, y, x + scale, y + scale))
                if scale != target_size:
                  patch = patch.resize((target_size, target_size), Image.Resampling.LANCZOS)

                patch_name = f"{filename_no_ext}_{scale}_{x}_{y}{ext}"
                patch.save(os.path.join(target_dir, patch_name))

            scale += target_size

        # 3. 處理成功後，將該檔案加入 Registry 並即時寫入 Log
        processed_registry.add(file_identifier)
        with open(log_path, 'a', encoding='utf-8') as f:
          f.write(f"{file_identifier}\n")

        newly_processed_count += 1

      except Exception as e:
        print(f"處理圖片失敗 {src_path}: {e}")

  print(f"增量處理完成！本次共新增處理了 {newly_processed_count} 張大圖。")

# 使用範例:
config = MyConfig()
prepare_data(config.raw_data_dirs[0], 224, './data/processed_all')
