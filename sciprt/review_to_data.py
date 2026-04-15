import os
import shutil
import glob

def review_to_data_all(src_pattern, data_dir):
  """
  掃描符合 src_pattern (例如 './data/need_review_semantic*') 的所有資料夾，
  將人工審查後保留的圖片，移回正式訓練資料夾。
  """
  total_moved = 0

  # 直接使用傳入的完整 pattern 進行搜尋
  review_dirs = [d for d in glob.glob(src_pattern) if os.path.isdir(d)]

  if not review_dirs:
    print(f"未找到符合 '{src_pattern}' 的資料夾。")
    return

  for review_dir in review_dirs:
    print(f"開始處理審查目錄: {review_dir}")
    moved_count = 0

    # 針對單一 review_dir 進行遍歷
    for root, dirs, files in os.walk(review_dir, topdown=False):
      for file in files:
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
          continue

        src_path = os.path.join(root, file)

        # 取得相對於當前 review_dir 的相對路徑 (例如: 台灣欒樹/)
        rel_path = os.path.relpath(root, review_dir)

        # 目標目錄 (正式訓練資料夾)
        target_dir = os.path.join(data_dir, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, file)
        shutil.move(src_path, target_path)
        moved_count += 1
        total_moved += 1

      # 選配功能：移動完檔案後，刪除空資料夾保持環境整潔
      try:
        if not os.listdir(root):
          os.rmdir(root)
      except OSError:
        pass

    print(f" - 結算: 從 {os.path.basename(review_dir)} 移回了 {moved_count} 張圖片。")

    # 如果最外層的 review_dir (例如 need_review_semantic_grass) 也空了，一併刪除
    try:
      if not os.listdir(review_dir):
        os.rmdir(review_dir)
        print(f" - 目錄 {os.path.basename(review_dir)} 已淨空並刪除。")
    except OSError:
      pass

  print("-" * 30)
  print(f"全部審查完成！總共將 {total_moved} 張圖片移回正式訓練集 {data_dir}。")

# ==========================================
# 使用範例
# ==========================================

# 1. 如果你只想把 CLIP (semantic) 挑出來的圖移回去：
# review_to_data_all('./data/need_review_semantic*', './data/processed_all')

# 2. 如果你想把所有 need_review 開頭的 (包含 blurry, nogreen) 全部移回去：
review_to_data_all('./data/need_review*', './data/processed_all')
