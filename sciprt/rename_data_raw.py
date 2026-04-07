import os
import sys

def rename_images(base_dir):
    if not os.path.exists(base_dir):
        print(f"錯誤：找不到目錄 {base_dir}")
        return

    # 取得 base_dir 下所有的第一層資料夾（也就是你的植物類別名稱）
    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for category in categories:
        cat_dir = os.path.join(base_dir, category)
        count = 1

        # 使用 os.walk 遞迴往下找，這樣即使有 20260322 這種子資料夾也能處理到
        for root, dirs, files in os.walk(cat_dir):
            # 排序確保每次處理順序一致
            files.sort()
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    old_path = os.path.join(root, filename)
                    ext = os.path.splitext(filename)[1].lower()

                    # 建立新檔名，例如：羊蹄甲_0001.jpg
                    new_name = f"{category}_{count:04d}{ext}"
                    new_path = os.path.join(root, new_name)

                    # 避免重複命名造成的覆蓋問題
                    if old_path != new_path:
                        # 如果目標檔名已存在，為了避免覆蓋，可以先加個暫時後綴 (雖然通常不會發生)
                        if os.path.exists(new_path):
                            temp_path = os.path.join(root, f"temp_{new_name}")
                            os.rename(old_path, temp_path)
                            os.rename(temp_path, new_path)
                        else:
                            os.rename(old_path, new_path)
                        print(f"重新命名: {os.path.basename(old_path)} -> {new_name}")

                    count += 1
    print("\n✅ 所有圖片重新命名完成！")

if __name__ == "__main__":
    # 檢查命令列參數
    if len(sys.argv) != 2:
        print("用法: python rename_data_raw.py <輸入目錄>")
        print("範例: python rename_data_raw.py ./data_raw")
        sys.exit(1)

    target_dir = sys.argv[1]
    rename_images(target_dir)
