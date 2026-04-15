import os
import random
import argparse
from pathlib import Path

def keep_n_random_files(directory_path, n):
  """
  在指定資料夾中隨機刪除檔案，直到恰好剩下 n 個檔案。
  """
  target_dir = Path(directory_path)

  # 檢查資料夾是否存在
  if not target_dir.is_dir():
    print(f"錯誤：找不到資料夾 '{directory_path}'")
    return

  # 取得資料夾下的所有「檔案」（自動排除子資料夾）
  all_files = [f for f in target_dir.iterdir() if f.is_file()]
  total_files = len(all_files)

  print(f"資料夾 '{target_dir.name}' 中目前共有 {total_files} 個檔案。")

  # 如果檔案數量已經小於或等於 n，則不需要刪除
  if total_files <= n:
    print(f"目標數量為 {n}，目前檔案數已達標或更少，無需刪除。")
    return

  # 計算需要刪除的檔案數量
  delete_count = total_files - n
  print(f"準備隨機刪除 {delete_count} 個檔案...")

  # 隨機選出要被刪除的檔案列表
  files_to_delete = random.sample(all_files, delete_count)

  # 執行刪除操作
  deleted_count = 0
  for file_path in files_to_delete:
    try:
      file_path.unlink()
      deleted_count += 1
      # 若想查看被詳細刪除的檔案名稱，可取消下方註解
      # print(f"已刪除: {file_path.name}")
    except Exception as e:
      print(f"刪除檔案 {file_path.name} 時發生錯誤: {e}")

  print(f"清理完成！成功刪除了 {deleted_count} 個檔案。目前剩餘檔案數：{total_files - deleted_count}。")

if __name__ == "__main__":
  # 建立參數解析器
  parser = argparse.ArgumentParser(
    description="隨機刪除指定目錄中的檔案，直到剩下 n 個檔案。"
  )
  
  # 加入參數
  parser.add_argument(
    "directory", 
    type=str, 
    help="目標資料夾的路徑"
  )
  parser.add_argument(
    "n", 
    type=int, 
    help="希望保留的檔案數量 (必須是大於或等於 0 的整數)"
  )

  # 解析終端機傳入的參數
  args = parser.parse_args()

  # 防呆檢查：確保 n 不是負數
  if args.n < 0:
    print("錯誤：保留的檔案數量 'n' 不能為負數。")
  else:
    # 執行主要功能
    keep_n_random_files(args.directory, args.n)
