import os
import sys
from PIL import Image

def cut_img_square(input_dir, output_dir, size=224):
  if not os.path.exists(input_dir):
    print(f"錯誤：找不到輸入目錄 {input_dir}")
    return

  for root, dirs, files in os.walk(input_dir):
    rel_path = os.path.relpath(root, input_dir)
    target_dir = os.path.join(output_dir, rel_path)

    if not os.path.exists(target_dir):
      os.makedirs(target_dir)

    for filename in files:
      if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        in_path = os.path.join(root, filename)

        # 取得不含副檔名的檔名，以便後續編號
        base_name = os.path.splitext(filename)[0]

        try:
          img = Image.open(in_path).convert("RGB")
          width, height = img.size

          # 計算可以切出多少個完整的小塊
          rows = height // size
          cols = width // size

          if rows == 0 or cols == 0:
            print(f"跳過 {filename}: 尺寸不足 {size}x{size}")
            continue

          for r in range(rows):
            for c in range(cols):
              # 計算裁切座標 (left, top, right, bottom)
              left = c * size
              top = r * size
              right = left + size
              bottom = top + size

              img_cropped = img.crop((left, top, right, bottom))

              # 輸出檔名加入行列編號，例如: image_0_0.jpg, image_0_1.jpg
              out_filename = f"{base_name}_{r}_{c}.jpg"
              out_path = os.path.join(target_dir, out_filename)

              img_cropped.save(out_path, "JPEG", quality=85)

          print(f"已切割並儲存來自 {filename} 的 {rows * cols} 張小圖")

        except Exception as e:
          print(f"❌ 處理 {in_path} 時發生錯誤: {e}")

  print("\n✅ 所有圖片切割完成！")

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("用法: python cut_img_square.py <輸入目錄> <輸出目錄>")
    sys.exit(1)

  in_dir = sys.argv[1]
  out_dir = sys.argv[2]

  cut_img_square(in_dir, out_dir)
