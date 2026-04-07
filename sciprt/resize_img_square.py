import os
import sys
from PIL import Image

def resize_img_square(input_dir, output_dir, size=224):
  if not os.path.exists(input_dir):
    print(f"錯誤：找不到輸入目錄 {input_dir}")
    return

  # os.walk 會走訪 input_dir 下所有的資料夾與檔案
  for root, dirs, files in os.walk(input_dir):
    # 計算出當前所在目錄相對於 input_dir 的相對路徑
    rel_path = os.path.relpath(root, input_dir)

    # 組合出對應的輸出目錄路徑
    target_dir = os.path.join(output_dir, rel_path)

    # 如果該輸出目錄不存在，則建立它（維持樹狀結構）
    if not os.path.exists(target_dir):
      os.makedirs(target_dir)

    for filename in files:
      if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        in_path = os.path.join(root, filename)
        out_path = os.path.join(target_dir, filename)

        try:
          # 讀取圖片，轉為 RGB 以防 PNG 的透明通道報錯
          img = Image.open(in_path).convert("RGB")

          # 1. 根據中間點，剪切一個正方形，多餘的邊邊刪掉
          width, height = img.size
          new_dim = min(width, height)

          # 計算裁切框的左、上、右、下座標 (以中心點向外推算)
          left = (width - new_dim) / 2
          top = (height - new_dim) / 2
          right = (width + new_dim) / 2
          bottom = (height + new_dim) / 2

          img_cropped = img.crop((left, top, right, bottom))

          # 2. 縮放到指定大小 (預設 224x224)
          # 使用 Image.Resampling.LANCZOS 確保縮放時的畫質
          img_resized = img_cropped.resize((size, size), Image.Resampling.LANCZOS)

          # 儲存到新的資料夾中，格式統一是 JPEG，壓縮品質 85，且不修改原始檔名
          img_resized.save(out_path, "JPEG", quality=85)
          print(f"已處理並儲存: {out_path}")
        except Exception as e:
          print(f"❌ 處理 {in_path} 時發生錯誤: {e}")

  print("\n✅ 所有圖片處理與壓縮完成！")

if __name__ == "__main__":
  # 檢查命令列參數
  if len(sys.argv) != 3:
    print("用法: python resize_img_square.py <輸入目錄> <輸出目錄>")
    print("範例: python resize_img_square.py ./data_raw ./data")
    sys.exit(1)

  in_dir = sys.argv[1]
  out_dir = sys.argv[2]

  resize_img_square(in_dir, out_dir)
